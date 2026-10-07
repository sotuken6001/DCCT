"""画像ソースの抽象化（ディレクトリ / ZIP直読み）。

96GB級のZIPを展開せずに扱えるよう、ZIP内の画像を遅延読み出しする。
DBの `images.filepath` には以下の形式で記録する（04_DB設計書のスキーマ変更は不要）:

    ディレクトリ: /Volumes/Extreme Pro/BigGAN/train/ai/xxx.png
    ZIP内:        /Volumes/Extreme Pro/genimage-...zip!train/ai/xxx.png

`open_image_bytes()` はどちらの形式のパスも透過的に読める。学習・評価時の
DataLoaderからはワーカーごとに `ZipSource` を開き直すこと（ZipFileオブジェクトは
プロセス間・スレッド間で共有しない）。
"""

from __future__ import annotations

import os
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

# ZIPパスと内部パスの区切り（Java の jar URL 表記に倣う）
ZIP_SEPARATOR = "!"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# OS・ファイルシステムが作る管理ファイル。画像として数えると枚数が水増しされ、
# 中身は画像ではないので「破損」に分類されてしまう。
# 特に macOS が exFAT 等へコピーする際に作る AppleDouble（`._<元のファイル名>`）は
# 元画像と1対1で作られるため、放置すると走査枚数がちょうど倍になる。
JUNK_DIRECTORIES = {
    "__MACOSX",
    "System Volume Information",
    ".Spotlight-V100",
    ".Trashes",
    ".fseventsd",
    ".TemporaryItems",
    "$RECYCLE.BIN",
}
JUNK_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


def is_junk_entry(relpath: str) -> bool:
    """OS管理ファイル（AppleDouble等）なら True。"""
    parts = Path(relpath).parts
    if any(part in JUNK_DIRECTORIES for part in parts):
        return True

    name = parts[-1] if parts else relpath
    return name.startswith("._") or name in JUNK_FILENAMES


@dataclass(frozen=True)
class ImageEntry:
    """ソース内の1画像。"""

    relpath: str          # ソースルートからの相対パス（例: "train/ai/xxx.png"）
    size_bytes: int
    filepath: str         # DBに記録する一意なパス


class ImageSource(ABC):
    """ディレクトリとZIPを同じインタフェースで扱うための基底クラス。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    @abstractmethod
    def iter_entries(self) -> Iterator[ImageEntry]:
        """ソース内の画像ファイルを列挙する。"""

    @abstractmethod
    def open(self, entry: ImageEntry) -> IO[bytes]:
        """画像のバイトストリームを開く。"""

    def close(self) -> None:
        """保持しているハンドルを解放する。"""

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DirectorySource(ImageSource):
    """展開済みディレクトリを走査するソース。

    `Path.rglob` ではなく `os.scandir` を**遅延的に**使う。GenImageは1ディレクトリに
    10万件以上のファイルが入るため、ツリー全体をリスト化してソートすると、1件目を
    返すまでに数十秒待たされる（外部SSD・exFATでは特に顕著）。ここではディレクトリを
    読みながら随時返し、サブディレクトリだけを名前順に辿る。

    ファイルの返却順はファイルシステム依存になるが、split割当はパスのハッシュで決まる
    （順序に依存しない）ため再現性には影響しない。`--limit` はスモークテスト用であり、
    どのファイルが選ばれるかは環境依存になる。
    """

    def __init__(self, root: Path, collect_size: bool = True) -> None:
        super().__init__(root)
        # サイズ取得はファイルごとに stat が要る。不要なら省いて走査を速くする
        self.collect_size = collect_size

    def iter_entries(self) -> Iterator[ImageEntry]:
        yield from self._walk(self.root)

    def _walk(self, directory: Path) -> Iterator[ImageEntry]:
        subdirectories: list[Path] = []
        try:
            scanner = os.scandir(directory)
        except OSError:
            return

        with scanner:
            for entry in scanner:
                if entry.name in JUNK_DIRECTORIES or entry.name.startswith("._"):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        subdirectories.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if Path(entry.name).suffix.lower() not in IMAGE_SUFFIXES or entry.name in JUNK_FILENAMES:
                    continue

                size = 0
                if self.collect_size:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0

                yield ImageEntry(
                    relpath=Path(entry.path).relative_to(self.root).as_posix(),
                    size_bytes=size,
                    filepath=entry.path,
                )

        # サブディレクトリ数は少ないので、こちらは名前順に辿る
        for subdirectory in sorted(subdirectories):
            yield from self._walk(subdirectory)

    def open(self, entry: ImageEntry) -> IO[bytes]:
        return (self.root / entry.relpath).open("rb")


class ZipSource(ImageSource):
    """ZIPを展開せずに走査・読み出しするソース。"""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._zf: zipfile.ZipFile | None = None

    @property
    def zipfile_handle(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.root)
        return self._zf

    def iter_entries(self) -> Iterator[ImageEntry]:
        # 中央ディレクトリを読むだけなので展開は発生しない
        for info in self.zipfile_handle.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # macOSで作られたZIPに混ざるリソースフォークなどを除外
            if is_junk_entry(name):
                continue
            yield ImageEntry(
                relpath=name,
                size_bytes=info.file_size,
                filepath=f"{self.root}{ZIP_SEPARATOR}{name}",
            )

    def open(self, entry: ImageEntry) -> IO[bytes]:
        return self.zipfile_handle.open(entry.relpath)

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None


def open_source(path: str | Path, source_type: str = "auto", collect_size: bool = True) -> ImageSource:
    """パスとタイプ指定から適切な `ImageSource` を返す。"""
    p = Path(path)
    if source_type == "auto":
        source_type = "zip" if p.suffix.lower() == ".zip" else "dir"
    if source_type == "zip":
        return ZipSource(p)
    if source_type == "dir":
        return DirectorySource(p, collect_size=collect_size)
    raise ValueError(f"未知のソースタイプです: {source_type!r}")


def split_zip_path(filepath: str) -> tuple[str, str] | None:
    """`...zip!inner/path.png` を (zipパス, 内部パス) に分解する。ZIP形式でなければ None。"""
    marker = f".zip{ZIP_SEPARATOR}"
    idx = filepath.lower().find(marker)
    if idx == -1:
        return None
    boundary = idx + len(marker)
    return filepath[: boundary - 1], filepath[boundary:]


def open_image_bytes(filepath: str, max_bytes: int | None = None) -> bytes:
    """DBに記録された `filepath` から画像バイト列を読む（ディレクトリ／ZIP両対応）。

    `max_bytes` を指定すると先頭その分だけ読む（ヘッダのみ読みたい場合に使う）。
    """
    parts = split_zip_path(filepath)
    if parts is None:
        with open(filepath, "rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    zip_path, inner = parts
    with zipfile.ZipFile(zip_path) as zf, zf.open(inner) as f:
        return f.read() if max_bytes is None else f.read(max_bytes)


# ---------------------------------------------------------------------------
# 学習・評価時の読み出し（ワーカーごとにZIPハンドルをキャッシュする）
# ---------------------------------------------------------------------------

# プロセスごとに独立したキャッシュ。DataLoaderの各ワーカーが自分のハンドルを持つ。
_ZIP_CACHE: dict[str, zipfile.ZipFile] = {}


def open_image_bytes_cached(filepath: str) -> bytes:
    """`open_image_bytes` と同じだが、ZIPハンドルをプロセス内で使い回す。

    1枚ごとにZIPを開き直すと中央ディレクトリの読み直しが発生するため、
    学習ループではこちらを使う。`ZipFile` はスレッド安全ではないので、
    DataLoaderのワーカー数は増やしてもワーカー内は単一スレッドで読むこと。
    """
    parts = split_zip_path(filepath)
    if parts is None:
        with open(filepath, "rb") as f:
            return f.read()

    zip_path, inner = parts
    handle = _ZIP_CACHE.get(zip_path)
    if handle is None:
        handle = zipfile.ZipFile(zip_path)
        _ZIP_CACHE[zip_path] = handle
    with handle.open(inner) as f:
        return f.read()


def close_cached_zips() -> None:
    """キャッシュしているZIPハンドルをすべて閉じる。"""
    for handle in _ZIP_CACHE.values():
        handle.close()
    _ZIP_CACHE.clear()

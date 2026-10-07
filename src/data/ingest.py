"""データ取り込みバッチ（FR-1 / 02_基本設計書 `data.ingest`）。

外部SSD上のGenImageを走査し、`datasets` / `generators` / `images` へ登録する。
ZIPは展開せずに中央ディレクトリを読み、画像ヘッダのみを復号して width/height を得る。

GenImageの構成は `<生成器>/{train,val}/{ai,nature}/*.png` を基本とするが、配布物によっては
`imagenet_ai_0419_sdv4/train/ai/...` のように上位フォルダが1段挟まる。このため
相対パスの末尾から「クラス階層＝親ディレクトリ」「split階層＝その親」として判定し、
階層の深さに依存しないようにしている。
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from ..experiment.config import Config
from .repository import MetadataRepository
from .sources import ImageEntry, ImageSource, is_junk_entry, open_source

logger = logging.getLogger(__name__)

# 画像ヘッダを読むために復号する先頭バイト数。PNG/JPEGのサイズ情報はここに収まる。
_HEADER_PROBE_BYTES = 65536

# `images` への一括INSERTの単位
_INSERT_BATCH_SIZE = 2000

# 実写画像を共通の1生成器にまとめる場合の名前（04_DB設計書 3.2 の例に合わせる）
_SHARED_REAL_GENERATOR = "ImageNet(real)"


@dataclass
class IngestStats:
    """生成器1件あたりの取り込み結果。"""

    generator: str
    scanned: int = 0
    inserted: int = 0
    invalid: int = 0
    too_small: int = 0
    unclassified: int = 0
    by_split: dict[str, int] = field(default_factory=dict)
    skipped_reason: str | None = None

    dry_run: bool = False

    @property
    def already_registered(self) -> int:
        """既にDBにある（＝今回挿入されなかった）枚数。dry-run時は判定できないので0。"""
        if self.dry_run:
            return 0
        return self.scanned - self.inserted - self.unclassified


def _alias_lookup(aliases: dict[str, list[str]]) -> dict[str, str]:
    """{正規名: [別名...]} を {別名(小文字): 正規名} に反転する。"""
    table: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            table[str(name).lower()] = canonical
    return table


def classify_entry(relpath: str, alias_table: dict[str, str]) -> tuple[str, str] | None:
    """相対パスから (split階層, クラス階層) を判定する。判定できなければ None。

    末尾から数えて親ディレクトリをクラス（ai / real）、その親をsplit（train / val）とみなす。
    """
    parts = Path(relpath).parts
    if len(parts) < 3:
        return None

    label_key = alias_table.get(parts[-2].lower())
    split_key = alias_table.get(parts[-3].lower())
    if label_key not in ("ai", "real") or split_key not in ("train", "val"):
        return None
    return split_key, label_key


def assign_split(source_split: str, filepath: str, seed: int, val_ratio: float, source_val_as: str) -> str:
    """GenImageの train/val を DBの train/val/test に割り当てる。

    - GenImage の val/ は `source_val_as`（既定 test）へ。
    - GenImage の train/ は、seedとファイルパスのハッシュで決まる比率 `val_ratio` 分だけ val へ。
      ハッシュベースなので再実行しても同じ画像が同じsplitに入る（NFR-1 再現性）。
    """
    if source_split == "val":
        return source_val_as
    if val_ratio <= 0:
        return "train"

    digest = hashlib.blake2b(f"{seed}:{filepath}".encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") / float(1 << 64)
    return "val" if bucket < val_ratio else "train"


def _probe_image(source: ImageSource, entry: ImageEntry) -> tuple[int | None, int | None, str]:
    """画像ヘッダを読んで (width, height, status) を返す。読めなければ status='invalid'。"""
    try:
        with source.open(entry) as stream:
            head = stream.read(_HEADER_PROBE_BYTES)
            try:
                with Image.open(io.BytesIO(head)) as img:
                    return img.width, img.height, "valid"
            except Exception:
                # ヘッダが先頭64KBに収まらない形式のために全体を読み直す
                body = head + stream.read()
        with Image.open(io.BytesIO(body)) as img:
            return img.width, img.height, "valid"
    except Exception as exc:  # 破損・読込不可（03_詳細設計書 6節）
        logger.debug("画像を読めませんでした (%s): %s", entry.filepath, exc)
        return None, None, "invalid"


def _iter_rows(
    source: ImageSource,
    *,
    dataset_id: int,
    ai_generator_id: int,
    real_generator_id: int,
    alias_table: dict[str, str],
    seed: int,
    val_ratio: float,
    source_val_as: str,
    patch_size: int,
    limit: int | None,
    probe_header: bool,
    stats: IngestStats,
) -> Iterator[dict[str, Any]]:
    for entry in source.iter_entries():
        if limit is not None and stats.scanned >= limit:
            break

        classified = classify_entry(entry.relpath, alias_table)
        if classified is None:
            stats.unclassified += 1
            continue
        source_split, label = classified
        stats.scanned += 1

        if probe_header:
            width, height, status = _probe_image(source, entry)
        else:
            # 画像を開かずパスとサイズだけで登録する。ディレクトリ走査が遅い環境向け。
            # width/height が不明なのでクロップ可否は学習時のフォールバックに任せる。
            width, height, status = None, None, "valid"

        if status == "invalid":
            stats.invalid += 1
            is_croppable = 0
        else:
            is_croppable = 1 if width is None else int(width >= patch_size and height >= patch_size)
            if not is_croppable:
                stats.too_small += 1

        split = assign_split(source_split, entry.filepath, seed, val_ratio, source_val_as)
        stats.by_split[split] = stats.by_split.get(split, 0) + 1

        yield {
            "dataset_id": dataset_id,
            "generator_id": ai_generator_id if label == "ai" else real_generator_id,
            "filepath": entry.filepath,
            "label": label,
            "split": split,
            "width": width,
            "height": height,
            "filesize_bytes": entry.size_bytes,
            "checksum": None,  # ingest時は算出しない（全データ読み直しになるため）
            "status": status,
            "is_croppable": is_croppable,
        }


def ingest(
    config: Config,
    *,
    only_generators: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
    probe_header: bool | None = None,
) -> list[IngestStats]:
    """configに従って外部SSD上のGenImageを走査し、DBへ登録する。

    生成器（ソース）単位でトランザクションを分けるため、途中で失敗しても
    成功済みの生成器はDBに残り、失敗した生成器のみ再実行できる（03_詳細設計書 6節）。

    `probe_header` は画像ヘッダを読んで width/height と破損を判定するかどうか。
    `None`（既定）ではソース種別ごとに決める:

    - ZIP: 有効。中央ディレクトリを一括で読むため1枚あたりのコストが小さい
      （実測 1,200〜3,000 img/s）
    - ディレクトリ: 無効。exFATの外部SSDでは、30万件規模のディレクトリに対して
      1ファイルずつ stat すると後ろへ行くほど遅くなり（実測で 54 img/s → 0.4 img/s）
      現実的な時間で終わらない。ディレクトリを読むだけなら数万 img/s 出る

    ソースごとに `probe: true/false` を書けば個別に上書きでき、CLIの
    `--probe` / `--no-probe` はすべてのソースに対して強制する。
    """
    dataset_root = Path(config.get("paths.dataset_root")).expanduser()
    dataset_name = config.get("dataset.name", "GenImage")
    excluded = set(config.get("dataset.excluded_generators", []))
    sources_cfg = config.get("dataset.sources", [])
    alias_table = _alias_lookup(config.get("dataset.directory_aliases", {}))
    seed = int(config.get("project.seed", 42))
    val_ratio = float(config.get("dataset.split.val_ratio_from_train", 0.0))
    source_val_as = config.get("dataset.split.source_val_as", "test")
    real_naming = config.get("dataset.real_generator_naming", "per_source")
    patch_size = int(config.get("preprocess.patch_size", 64))

    results: list[IngestStats] = []
    repo = MetadataRepository(config.get("paths.db_path"))
    try:
        dataset_id = repo.get_or_create_dataset(
            dataset_name, str(dataset_root), description="GenImage benchmark (外部SSD上のパス参照)"
        )

        for spec in sources_cfg:
            name = spec["name"]
            stats = IngestStats(generator=name, dry_run=dry_run)

            if only_generators and name not in only_generators:
                continue

            is_excluded = name in excluded
            source_path = dataset_root / spec["path"] if spec.get("path") else None
            missing = source_path is None or not source_path.exists()

            # 除外・未取得の生成器も、マスタには notes 付きで登録しておく（【OI-1】対応）
            notes = None
            if is_excluded:
                notes = "configのexcluded_generatorsにより除外中"
            if missing:
                notes = (notes + " / " if notes else "") + "データ未取得（パス未確定または不在）"

            ai_generator_id = repo.get_or_create_generator(
                dataset_id, name, spec.get("category", "diffusion"), is_real=False, notes=notes
            )

            if is_excluded:
                stats.skipped_reason = "excluded_generators により除外"
                results.append(stats)
                logger.info("スキップ（除外設定）: %s", name)
                continue
            if missing:
                stats.skipped_reason = f"パスが見つかりません: {source_path}"
                results.append(stats)
                logger.warning("スキップ（パス不在）: %s -> %s", name, source_path)
                continue

            real_name = _SHARED_REAL_GENERATOR if real_naming == "shared" else f"{_SHARED_REAL_GENERATOR}@{name}"
            real_generator_id = repo.get_or_create_generator(
                dataset_id,
                real_name,
                "real",
                is_real=True,
                notes=None if real_naming == "shared" else f"{name} サブセットの実写画像（nature/）",
            )

            source_type = spec.get("type", "auto")
            is_zip = source_type == "zip" or (
                source_type == "auto" and source_path.suffix.lower() == ".zip"
            )
            # 明示指定 > ソース個別の設定 > 種別ごとの既定
            probe = probe_header if probe_header is not None else bool(spec.get("probe", is_zip))

            logger.info(
                "走査開始: %s -> %s（画像ヘッダの読み取り: %s）",
                name,
                source_path,
                "あり" if probe else "なし",
            )
            # probe しないときはファイルごとの stat も省く（ディレクトリ走査の主コスト）
            with open_source(source_path, source_type, collect_size=probe) as source:
                rows = _iter_rows(
                    source,
                    dataset_id=dataset_id,
                    ai_generator_id=ai_generator_id,
                    real_generator_id=real_generator_id,
                    alias_table=alias_table,
                    seed=seed,
                    val_ratio=val_ratio,
                    source_val_as=source_val_as,
                    patch_size=patch_size,
                    limit=limit,
                    probe_header=probe,
                    stats=stats,
                )
                if progress:
                    try:
                        from tqdm import tqdm

                        rows = tqdm(rows, desc=name, unit="img")
                    except ImportError:  # pragma: no cover
                        pass

                batch: list[dict[str, Any]] = []
                for row in rows:
                    batch.append(row)
                    if len(batch) >= _INSERT_BATCH_SIZE:
                        if not dry_run:
                            stats.inserted += repo.register_images(batch)
                        batch.clear()
                if batch and not dry_run:
                    stats.inserted += repo.register_images(batch)

            results.append(stats)
            logger.info(
                "走査完了: %s 走査=%d 新規登録=%d 破損=%d 小さすぎ=%d",
                name,
                stats.scanned,
                stats.inserted,
                stats.invalid,
                stats.too_small,
            )
    finally:
        repo.close()

    return results


def cleanup_junk(config: Config, dry_run: bool = False) -> tuple[int, list[str]]:
    """既にDBへ登録されてしまったOS管理ファイル（AppleDouble等）の行を削除する。

    AppleDouble（`._<元のファイル名>`）は画像ではないため `status='invalid'` として
    登録される。学習・評価からは元々除外されるが、枚数の集計を大きく歪めるので削除する。

    戻り値: (該当件数, 例として最大5件のパス)
    """
    repo = MetadataRepository(config.get("paths.db_path"))
    try:
        rows = repo.conn.execute("SELECT image_id, filepath FROM images").fetchall()
        junk_ids = [row["image_id"] for row in rows if is_junk_entry(row["filepath"])]
        samples = [row["filepath"] for row in rows if is_junk_entry(row["filepath"])][:5]

        if junk_ids and not dry_run:
            with repo.conn:
                for start in range(0, len(junk_ids), 500):
                    chunk = junk_ids[start : start + 500]
                    placeholders = ", ".join("?" for _ in chunk)
                    repo.conn.execute(f"DELETE FROM images WHERE image_id IN ({placeholders})", chunk)
            repo.conn.execute("VACUUM")
    finally:
        repo.close()

    return len(junk_ids), samples

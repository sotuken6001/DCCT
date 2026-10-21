"""本物の JPEG（ImageNet）の作りを数え、AI画像にかけている PIL の JPEG と比べる（案ク、学習なし）。

案カで、pθ と qφ は学習で同じ「何か」を覚えたとわかった。案キで、それは模様の多さではなかった。
pθ にとっても JPEG96 の AI画像のほうが本物より予測しやすい（案ウ、AUC .397）ので、
「本物の JPEG と PIL の JPEG の作りの違い」が候補になる。DCCT は色どうしのつながりを見るので、
とくに色の間引き方（4:4:4 / 4:2:2 / 4:2:0）の違いは直接効きうる。

1枚ごとに、形式・色の持ち方（RGB / 白黒など）・色の間引き方・圧縮の表から推定した品質・
表が libjpeg の標準の表（品質から決まる表）と一致するかを調べ、まとまりごとに数える。
"""

from __future__ import annotations

import csv
import io
import logging
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, JpegImagePlugin

from ..data.repository import MetadataRepository
from ..data.sources import open_image_bytes_cached
from ..experiment.config import Config
from ..experiment.paths import resolve
from ..preprocess.patch_sampler import jpeg_bytes, load_image_tensor

logger = logging.getLogger(__name__)

_REAL_GENERATOR_PREFIX = "ImageNet(real)"
_SAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
PIL_GROUP = "PIL JPEG{quality}（AI画像にかけているもの）"
AI_GROUP = "AI画像（元のまま）"


@lru_cache(maxsize=None)
def _reference_tables() -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    """PIL（libjpeg）が品質 1〜100 で使う標準の圧縮の表（明るさ・色）。"""
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    tables = {}
    for quality in range(1, 101):
        buf = io.BytesIO()
        Image.fromarray(array).save(buf, format="JPEG", quality=quality)
        with Image.open(io.BytesIO(buf.getvalue())) as img:
            q = img.quantization
            tables[quality] = (tuple(q[0]), tuple(q.get(1, q[0])))
    return tables


def _estimate_quality(quantization: dict[int, Sequence[int]]) -> tuple[int, bool]:
    """圧縮の表から品質を推定する。戻り値（品質, 標準の表と完全に一致したか）。

    白黒（表が1つ）は明るさの表だけで比べる。一致しなければ、差の合計がいちばん小さい品質を返す。
    """
    luma = np.asarray(quantization[0], dtype=np.int64)
    chroma = np.asarray(quantization[1], dtype=np.int64) if 1 in quantization else None
    best, best_diff = 0, None
    for quality, (ref_luma, ref_chroma) in _reference_tables().items():
        diff = np.abs(luma - np.asarray(ref_luma)).sum()
        if chroma is not None:
            diff += np.abs(chroma - np.asarray(ref_chroma)).sum()
        if best_diff is None or diff < best_diff:
            best, best_diff = quality, diff
    return best, bool(best_diff == 0)


def inspect_image(data: bytes) -> dict[str, Any]:
    """画像1枚の形式・色の持ち方・色の間引き方・推定品質を返す。JPEG でなければ品質などは None。"""
    with Image.open(io.BytesIO(data)) as img:
        info: dict[str, Any] = {"format": img.format, "mode": img.mode,
                                "subsampling": None, "quality": None, "standard_tables": None}
        if img.format != "JPEG":
            return info
        if img.mode == "L":
            info["subsampling"] = "なし（白黒）"
        else:
            info["subsampling"] = _SAMPLING.get(JpegImagePlugin.get_sampling(img), "その他")
        if img.quantization:
            info["quality"], info["standard_tables"] = _estimate_quality(img.quantization)
    return info


def summarize(group: str, infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """まとまりごとの割合を出す。割合は JPEG の枚数に対して（jpeg_share だけは全体に対して）。"""
    total = len(infos)
    jpegs = [i for i in infos if i["format"] == "JPEG"]
    n = len(jpegs)

    def share(predicate) -> float | None:
        return sum(1 for i in jpegs if predicate(i)) / n if n else None

    qualities = Counter(i["quality"] for i in jpegs if i["quality"] is not None)
    modes = Counter(i["mode"] for i in infos)
    return {
        "group": group,
        "num_images": total,
        "jpeg_share": n / total if total else None,
        "share_444": share(lambda i: i["subsampling"] == "4:4:4"),
        "share_422": share(lambda i: i["subsampling"] == "4:2:2"),
        "share_420": share(lambda i: i["subsampling"] == "4:2:0"),
        "share_gray": share(lambda i: i["subsampling"] == "なし（白黒）"),
        "share_q96": share(lambda i: i["quality"] == 96),
        "share_standard": share(lambda i: i["standard_tables"] is True),
        "median_quality": float(np.median(list(qualities.elements()))) if qualities else None,
        "top_qualities": " ".join(f"{q}:{c}" for q, c in qualities.most_common(5)),
        "modes": " ".join(f"{m}:{c}" for m, c in modes.most_common()),
    }


class JpegStats:
    """DB から画像を選んで、まとまりごとに JPEG の作りを数える。"""

    def __init__(self, config: Config, max_images: int = 1000, pil_quality: int = 96) -> None:
        self.config = config
        self.max_images = int(max_images)
        self.pil_quality = int(pil_quality)
        self.rng = random.Random(int(config.get("project.seed", 42)))
        self.repo = MetadataRepository(config.get("paths.db_path"))

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "JpegStats":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _sample(self, label: str, names: Sequence[str] | None) -> list[dict[str, Any]]:
        rows = [dict(r) for r in self.repo.list_images(label=label, generator_names=names)]
        if len(rows) > self.max_images:
            rows = self.rng.sample(rows, self.max_images)
        return rows

    def _read(self, rows: Sequence[dict[str, Any]]) -> list[bytes]:
        out = []
        for row in rows:
            try:
                out.append(open_image_bytes_cached(row["filepath"]))
            except Exception as exc:  # 読めない画像は数えない
                logger.warning("読めないためスキップします (%s): %s", row["filepath"], exc)
        return out

    def _inspect_all(self, blobs: Sequence[bytes]) -> list[dict[str, Any]]:
        infos = []
        for blob in blobs:
            try:
                infos.append(inspect_image(blob))
            except Exception as exc:
                logger.warning("画像を調べられないためスキップします: %s", exc)
        return infos

    def run(self) -> list[dict[str, Any]]:
        excluded = set(self.config.get("dataset.excluded_generators", []))
        sources = [s["name"] for s in self.config.get("dataset.sources", []) if s["name"] not in excluded]
        shared = self.config.get("dataset.real_generator_naming", "per_source") == "shared"
        real_groups = [_REAL_GENERATOR_PREFIX] if shared else [f"{_REAL_GENERATOR_PREFIX}@{s}" for s in sources]

        results = []
        for group in real_groups:
            infos = self._inspect_all(self._read(self._sample("real", [group])))
            if infos:
                results.append(summarize(group, infos))
                logger.info("%s: %d枚 4:2:0=%s 品質96=%s", group, len(infos),
                            _fmt(results[-1]["share_420"]), _fmt(results[-1]["share_q96"]))

        ai_blobs = self._read(self._sample("ai", sources))
        results.append(summarize(AI_GROUP, self._inspect_all(ai_blobs)))
        # AI画像に実際にかけている JPEG（jpeg_compress と同じ処理）を、同じ画像で作って調べる
        ours = []
        for blob in ai_blobs:
            try:
                ours.append(inspect_image(jpeg_bytes(load_image_tensor(blob), self.pil_quality)))
            except Exception as exc:
                logger.warning("JPEG にできないためスキップします: %s", exc)
        results.append(summarize(PIL_GROUP.format(quality=self.pil_quality), ours))
        return results


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


COLUMNS = [
    "group", "num_images", "jpeg_share", "share_444", "share_422", "share_420", "share_gray",
    "share_q96", "share_standard", "median_quality", "top_qualities", "modes",
]


def write_report(results: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "jpeg_stats.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({k: "" if row[k] is None else row[k] for k in COLUMNS})
    lines = [
        "# 本物の JPEG と PIL の JPEG の作り（案ク）",
        "",
        "- 割合は JPEG の枚数に対して（JPEG の割合だけは全体に対して）",
        "- 品質は圧縮の表から推定。標準の表＝libjpeg が品質から作る表と完全に一致した割合",
        "- よく出る品質は「品質:枚数」",
        "",
        "| まとまり | 枚数 | JPEG | 4:4:4 | 4:2:2 | 4:2:0 | 白黒 | 品質96 | 標準の表 | 品質の中央値 | よく出る品質 | 色の持ち方 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append("| " + " | ".join(_fmt(row[k]) for k in COLUMNS) + " |")
    (out / "jpeg_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

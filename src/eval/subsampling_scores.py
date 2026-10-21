"""色の間引き方ごとに、条件付きモデル（pθ / qφ）の「予測しにくさ」の点数を比べる（確認コ、学習なし）。

案クで、本物は約8割が 4:4:4（色を間引かない）、AI画像にかけていた PIL の JPEG は全部 4:2:0
（縦横半分に間引く）だった。4:2:0 だと色どうしのつながりが強くなるので、モデルにとって
予測しやすくなるのでは、という見立て（推測）を直接確かめる。

4つのまとまりに一クラス判定と同じ点数 mean(NLL − entropy)（高いほど予測しにくい）をつける。

- 本物 4:4:4 / 本物 4:2:0：本物のうち、元の JPEG の色の間引き方で分けたもの
- AI画像 JPEG 4:2:0 / AI画像 JPEG 4:4:4：同じ AI画像に、品質を同じにして色の間引き方だけ変えた JPEG をかけたもの

ペアごとの AUC は「後ろのまとまりを当てる向き」で、0.5 より下なら後ろのまとまりのほうが予測しやすい。
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from ..data.sources import open_image_bytes_cached
from ..experiment.config import Config
from ..experiment.paths import resolve
from .evaluator import jpeg_transform
from .jpeg_stats import inspect_image
from .metrics import compute_metrics
from .one_class import OneClassEvaluator

logger = logging.getLogger(__name__)

GROUPS = ["本物 4:4:4", "本物 4:2:0", "AI画像 JPEG 4:2:0", "AI画像 JPEG 4:4:4"]
PAIRS = [
    ("本物 4:4:4", "本物 4:2:0"),
    ("本物 4:4:4", "AI画像 JPEG 4:2:0"),
    ("本物 4:4:4", "AI画像 JPEG 4:4:4"),
    ("AI画像 JPEG 4:4:4", "AI画像 JPEG 4:2:0"),
]


def pair_auc(first: np.ndarray, second: np.ndarray) -> float | None:
    """後ろのまとまりを当てる向きの AUC（点数が高いほど後ろとみなす）。"""
    scores = np.concatenate([first, second])
    labels = np.concatenate([np.zeros(len(first)), np.ones(len(second))])
    return compute_metrics(scores, labels, threshold=0.0)["auc"]


class SubsamplingScores:
    def __init__(
        self,
        config: Config,
        checkpoint: str | Path | None = None,
        max_images: int = 1000,
        ai_jpeg_quality: int = 96,
        split: str = "test",
    ) -> None:
        self.config = config
        self.max_images = int(max_images)
        self.quality = int(ai_jpeg_quality)
        self.split = split
        self.scorer = OneClassEvaluator(config, checkpoint)
        self.checkpoint = self.scorer.checkpoint
        self.rng = random.Random(int(config.get("project.seed", 42)))

    def close(self) -> None:
        self.scorer.close()

    def __enter__(self) -> "SubsamplingScores":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def real_rows_by_subsampling(self) -> dict[str, list[dict[str, Any]]]:
        """本物を無作為な順に見ていき、4:4:4 と 4:2:0 をそれぞれ max_images 枚まで集める。"""
        rows = [dict(r) for r in self.scorer.repo.list_images(label="real", split=self.split)]
        self.rng.shuffle(rows)
        found: dict[str, list[dict[str, Any]]] = {"4:4:4": [], "4:2:0": []}
        for row in rows:
            if all(len(v) >= self.max_images for v in found.values()):
                break
            try:
                subsampling = inspect_image(open_image_bytes_cached(row["filepath"]))["subsampling"]
            except Exception as exc:  # 読めない画像は数えない
                logger.warning("読めないためスキップします (%s): %s", row["filepath"], exc)
                continue
            if subsampling in found and len(found[subsampling]) < self.max_images:
                found[subsampling].append(row)
        return found

    def run(self) -> dict[str, Any]:
        real = self.real_rows_by_subsampling()
        ai_rows = [dict(r) for r in self.scorer.repo.list_images(label="ai", split=self.split)]
        if len(ai_rows) > self.max_images:
            ai_rows = self.rng.sample(ai_rows, self.max_images)

        scores: dict[str, np.ndarray] = {}
        for group, rows, transform in (
            ("本物 4:4:4", real["4:4:4"], None),
            ("本物 4:2:0", real["4:2:0"], None),
            ("AI画像 JPEG 4:2:0", ai_rows, jpeg_transform(self.quality, "4:2:0")),
            ("AI画像 JPEG 4:4:4", ai_rows, jpeg_transform(self.quality, "4:4:4")),
        ):
            if not rows:
                logger.warning("%s の画像がありません", group)
                continue
            scores[group] = self.scorer.score_rows(rows, transform=transform)[0]
            logger.info("%s: %d枚 点数の平均 %.4f", group, len(scores[group]), float(scores[group].mean()))

        groups = [
            {"group": g, "num_images": int(len(scores[g])), "mean": float(scores[g].mean()),
             "median": float(np.median(scores[g]))}
            for g in GROUPS if g in scores
        ]
        pairs = [
            {"first": a, "second": b, "auc": pair_auc(scores[a], scores[b])}
            for a, b in PAIRS if a in scores and b in scores
        ]
        return {"groups": groups, "pairs": pairs}


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_report(result: dict[str, Any], out_dir: str | Path, checkpoint: str | Path, quality: int) -> Path:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "subsampling_scores.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "group", "num_images", "mean", "median", "auc"])
        for g in result["groups"]:
            writer.writerow(["group", g["group"], g["num_images"], g["mean"], g["median"], ""])
        for p in result["pairs"]:
            writer.writerow(["pair", f"{p['first']} → {p['second']}", "", "", "", "" if p["auc"] is None else p["auc"]])

    lines = [
        "# 色の間引き方ごとの「予測しにくさ」（確認コ）",
        "",
        f"- モデル: `{checkpoint}`",
        f"- 点数: mean(NLL − entropy)。高いほど予測しにくい。AI画像は品質 {quality} の JPEG にしてから",
        "- AUC は後ろのまとまりを当てる向き。0.5 より下なら、後ろのまとまりのほうが予測しやすい",
        "",
        "| まとまり | 枚数 | 点数の平均 | 点数の中央値 |",
        "|---|---|---|---|",
        *[f"| {g['group']} | {g['num_images']} | {g['mean']:.4f} | {g['median']:.4f} |" for g in result["groups"]],
        "",
        "| 比べるもの | AUC |",
        "|---|---|",
        *[f"| {p['first']} → {p['second']} | {_fmt(p['auc'])} |" for p in result["pairs"]],
    ]
    (out / "subsampling_scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

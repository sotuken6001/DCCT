"""線（閾値）の引き方を調べる（確認セ、学習なし）。

サ1（run52）は一番厳しい条件で AUC .948 なのに正解率 82.8%。ADM は AUC .875 で正解率 63% だった。
線は学習に使った生成器（SDv1.4）の val で決めるので、見たことのない生成器では線がずれて、
正解率だけが下がっているのでは、という見立て（推測）を確かめる。

生成器ごとに次を出す。DB には残さない（同じ run の評価が重なって平均が崩れないように）。

- 正解率（いつもどおり val で決めた線）と、本物・AI画像それぞれの正解率
- 上限の正解率：その生成器の評価画像を見て、線をいちばん良い位置に引いたときの正解率。
  実際の判定には使えないが、「見分ける力はあるのに線がずれているだけか」の目安になる

確認ソ：`real_false_alarm`（%）を渡すと、線を本物の val の点数だけで決める。本物を AI と間違える
割合がその値になる位置（本物の点数の上から何%か）に引く。論文の DCCT† と同じく AI画像を使わないので、
見たことのない生成器でも同じ基準になる。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..experiment.paths import resolve
from .evaluator import Evaluator, ImageTransform
from .metrics import best_accuracy, class_accuracies, compute_metrics

logger = logging.getLogger(__name__)

COLUMNS = ["generator", "accuracy", "real_accuracy", "ai_accuracy", "best_accuracy", "auc", "threshold", "num_samples"]


def real_val_threshold(evaluator: Evaluator, false_alarm: float) -> float:
    """学習に使った生成器の本物の val だけで、本物を AI と間違える割合が `false_alarm`%（0〜100）になる線。"""
    if not 0.0 <= false_alarm <= 100.0:
        raise ValueError(f"false_alarm は 0〜100（%）: {false_alarm!r}")
    rows = [
        row
        for generator in evaluator.config.get("dataset.train_generators", [])
        for row in evaluator._rows_for(generator, "val", limit=None)
        if row["label"] == "real"
    ]
    if not rows:
        raise ValueError("線を決めるための本物の val がありません")
    scores, _ = evaluator.predictor.score_rows(rows)
    threshold = float(np.percentile(scores, 100.0 - false_alarm))
    logger.info("線を本物の val %d枚の上から %.1f%% の位置 %.4f にしました", len(scores), false_alarm, threshold)
    return threshold


def run_threshold_check(
    evaluator: Evaluator,
    ai_transform: ImageTransform | None = None,
    split: str = "test",
    real_false_alarm: float | None = None,
) -> list[dict[str, Any]]:
    """線は同じ条件の val で決める（AI画像だけ JPEG にする評価なら、val の AI画像も同じ JPEG にする）。

    `real_false_alarm` を渡すと、本物の val だけで線を決める（確認ソ）。
    """
    if real_false_alarm is None:
        threshold = evaluator.calibrate_threshold(apply=False, ai_transform=ai_transform)
    else:
        threshold = real_val_threshold(evaluator, float(real_false_alarm))
    results = []
    for generator in evaluator.target_generators():
        rows = evaluator._rows_for(generator, split)
        if not rows:
            logger.warning("評価対象の画像がありません: %s（split=%s）", generator, split)
            continue
        scores, labels = evaluator._score_rows(rows, None, ai_transform)
        metrics = compute_metrics(scores, labels, threshold)
        real_accuracy, ai_accuracy = class_accuracies(scores, labels, threshold)
        row = {
            "generator": generator,
            "accuracy": metrics["accuracy"],
            "real_accuracy": real_accuracy,
            "ai_accuracy": ai_accuracy,
            "best_accuracy": best_accuracy(scores, labels),
            "auc": metrics["auc"],
            "threshold": threshold,
            "num_samples": metrics["num_samples"],
        }
        logger.info(
            "%s: 正解率=%.4f（本物 %s・AI %s） 上限=%s AUC=%s",
            generator, row["accuracy"], _fmt(real_accuracy), _fmt(ai_accuracy), _fmt(row["best_accuracy"]), _fmt(row["auc"]),
        )
        results.append(row)
    return results


def _fmt(value: Any) -> str:
    return "N/A" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def with_mean_row(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return results
    row: dict[str, Any] = {"generator": "平均", "threshold": results[0]["threshold"],
                           "num_samples": sum(r["num_samples"] for r in results)}
    for key in ("accuracy", "real_accuracy", "ai_accuracy", "best_accuracy", "auc"):
        values = [r[key] for r in results if r[key] is not None]
        row[key] = float(np.mean(values)) if values else None
    return [*results, row]


def write_report(
    results: list[dict[str, Any]], out_dir: str | Path, run_id: int, condition: str, threshold_rule: str = ""
) -> Path:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = with_mean_row(results)
    with open(out / "threshold_check.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row[k] is None else row[k] for k in COLUMNS})
    lines = [
        "# 線（閾値）の引き方の確認（確認セ）",
        "",
        f"- run_id: {run_id}",
        f"- 条件: {condition}",
        f"- 線: {threshold_rule or '同じ条件の val で、本物と AI画像の正解率の和がいちばん高くなる位置'}",
        "- 上限: 評価画像を見て線をいちばん良い位置に引いたときの正解率（実際の判定には使えない目安）",
        "",
        "| generator | 正解率 | 本物の正解率 | AI画像の正解率 | 上限 | AUC | 枚数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['generator']} | {_fmt(row['accuracy'])} | {_fmt(row['real_accuracy'])} | {_fmt(row['ai_accuracy'])} "
            f"| {_fmt(row['best_accuracy'])} | {_fmt(row['auc'])} | {row['num_samples']} |"
        )
    (out / "threshold_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

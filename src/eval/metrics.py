"""評価指標と閾値チューニング（FR-5 / 03_詳細設計書 7.3）。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | None]:
    """Accuracy / AUC / AP を計算する。片方のクラスしかない場合 AUC・AP は None。"""
    predictions = (scores > threshold).astype(float)
    accuracy = float((predictions == labels).mean())

    both_classes = len(np.unique(labels)) == 2
    return {
        "accuracy": accuracy,
        "auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "ap": float(average_precision_score(labels, scores)) if both_classes else None,
        "num_samples": int(len(labels)),
    }


def tune_threshold(scores: np.ndarray, labels: np.ndarray, default: float = 0.5) -> float:
    """Youden指数（TPR - FPR）が最大になる閾値τを返す。

    論文にτの具体値の記載がないため、val splitでチューニングする運用とする
    （03_詳細設計書 7.3）。評価用のtest splitには触れない。
    """
    if len(np.unique(labels)) < 2:
        return default

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    best = int(np.argmax(true_positive_rate - false_positive_rate))
    candidate = float(thresholds[best])

    # roc_curve の先頭は +inf になることがあるため、その場合は既定値に戻す
    if not np.isfinite(candidate):
        return default

    # roc_curve の閾値は「score >= τ を陽性」とする規約だが、判定側は Algorithm 2 に従い
    # 「score > τ」で行う。境界のサンプルが逆に判定されないよう、ひとつ下のスコアとの
    # 中点まで下げる。
    lower = scores[scores < candidate]
    if lower.size:
        return float((candidate + lower.max()) / 2.0)
    return float(np.nextafter(candidate, -np.inf))

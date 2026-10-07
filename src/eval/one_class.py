"""一クラス検出（論文 Appendix B の DCCT†）。

pθ（写真で学習した条件付きモデル）だけを使い、AI生成画像を1枚も見ずに判定する。
スコアは論文どおり `D = mean(NLL − entropy)`（予測した色分布に対する負の対数尤度から、
その分布自体のエントロピーを引いたもの）で、高いほど「写真らしくない」= AI生成とみなす。
エントロピーを引くのは、もともと予測しにくい模様の画像（分布が広がる画像）まで
AI生成と判定しないようにするため。

閾値は論文どおり、学習用の写真のスコアの95パーセンタイルに置く。

分類器 gψ を通さないので、未知の生成器に対する pθ 自体の識別力を論文の表5と
直接比べられる。gψ を含む cross_generator 評価の結果が論文に届かないとき、
原因が pθ の側にあるのかを切り分けるのに使う。
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import MultiPatchDataset, real_generator_names, skip_invalid_collate
from ..data.repository import MetadataRepository
from ..experiment import builders
from ..experiment.config import Config
from ..experiment.paths import resolve
from ..model.conditional_unet import MixtureParams
from ..model.losses import NLLLoss
from .evaluator import jpeg_transform
from .metrics import compute_metrics

logger = logging.getLogger(__name__)

_REAL_GENERATOR_PREFIX = "ImageNet(real)"

# 論文 Appendix B 表5（GenImage、DCCT† の Accuracy %）
PAPER_TABLE5_ACCURACY = {
    "Midjourney": 95.16,
    "SDv1.4": 84.11,
    "SDv1.5": 83.99,
    "ADM": 81.65,
    "GLIDE": 93.19,
    "Wukong": 86.89,
    "VQDM": 87.37,
    "BigGAN": 94.48,
}


def value_log_probs(params: MixtureParams, truncation_t: float) -> torch.Tensor:
    """量子化された残差が取りうる各値 {-t,...,t} の対数確率 `[B, V, Cm, H, W]` を返す。"""
    loss = NLLLoss(truncation_t=truncation_t)
    # 全部の値をまとめてブロードキャストすると [B,K,V,Cm,H,W] の中間テンソルができ、
    # チャンネルごとの混合分布（Cm=60）では画像1枚（16パッチ）あたり数GBになる。
    # 値を1つずつ計算して K 方向を畳んでから積むと、最大でも [B,K,Cm,H,W] で済む
    per_value = []
    for value in range(-int(loss.truncation_t), int(loss.truncation_t) + 1):
        target = torch.full((1,) * params.mu.dim(), float(value), device=params.mu.device, dtype=params.mu.dtype)
        component = loss._component_log_prob(target, params)            # [B,K,Cm,H,W]
        per_value.append(torch.logsumexp(params.log_w + component, dim=1))
    return torch.stack(per_value, dim=1)


def nll_minus_entropy(y_prime: torch.Tensor, params: MixtureParams, truncation_t: float) -> torch.Tensor:
    """パッチごとのスコア `D = mean(NLL − entropy)` `[B]` を返す。

    画素ごとの混合分布（Cm=1）なら、y' の全チャンネルが同じ分布とエントロピーを共有する。
    """
    log_table = value_log_probs(params, truncation_t)                   # [B,V,Cm,H,W]
    entropy = -(log_table.exp() * log_table).sum(dim=1)                 # [B,Cm,H,W]

    channels = y_prime.shape[1]
    if log_table.shape[2] == 1:
        log_table = log_table.expand(-1, -1, channels, -1, -1)
    num_values = log_table.shape[1]
    index = (y_prime + truncation_t).round().long().clamp(0, num_values - 1)
    nll = -log_table.gather(1, index.unsqueeze(1)).squeeze(1)           # [B,C,H,W]

    return (nll - entropy).flatten(1).mean(dim=1)


class OneClassEvaluator:
    """pθ だけで各生成器を判定し、論文の表5と比べられる形で結果を出す。"""

    def __init__(
        self,
        config: Config,
        photo_checkpoint: str | Path | None = None,
        device: torch.device | None = None,
        ai_jpeg_quality: int | None = None,
    ) -> None:
        self.config = config
        self.device = device or builders.build_device(config)
        self.preprocessor = builders.build_preprocessor(config, self.device)
        self.sampler = builders.build_patch_sampler(config)
        self.num_patches = int(config.get("inference.num_patches", 16))
        self.truncation_t = float(config.get("preprocess.truncation_t", 7))
        self.max_images = config.get("evaluate.max_images_per_generator", None)
        self.num_calibration_images = int(config.get("evaluate.one_class.num_calibration_images", 2000))
        # 評価時は 画像数 × P パッチ × 15値 × K成分 の表を作るので、学習より小さく区切る
        self.batch_size = int(config.get("evaluate.one_class.batch_size", 4))
        self.threshold: float | None = None
        # GenImage の本物は JPEG、AI画像は PNG。形式の差で見分けていないかを確かめるため、
        # 指定があれば AI画像だけを同じ JPEG にしてから点数をつける（閾値を決める本物の写真には触らない）
        if ai_jpeg_quality is None:
            ai_jpeg_quality = config.get("evaluate.one_class.ai_jpeg_quality", None)
        if ai_jpeg_quality is not None and not 1 <= int(ai_jpeg_quality) <= 100:
            raise ValueError(f"ai_jpeg_quality は 1〜100: {ai_jpeg_quality!r}")
        self.ai_jpeg_quality = None if ai_jpeg_quality is None else int(ai_jpeg_quality)

        self.checkpoint = photo_checkpoint or config.get("stage2.photo_model_checkpoint", None)
        if self.checkpoint is None:
            raise ValueError("pθ のチェックポイントが指定されていません（--photo-checkpoint か stage2.photo_model_checkpoint）")
        self.model = builders.build_conditional_unet(config, self.preprocessor).to(self.device)
        state = torch.load(resolve(self.checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["modules"]["conditional_unet"])
        self.model.freeze()
        logger.info("pθ を読み込みました: %s", self.checkpoint)

        self.repo = MetadataRepository(config.get("paths.db_path"))

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "OneClassEvaluator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # スコア計算
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """`[B, P, 3, s, s]` → 画像ごとの平均スコア `[B]`（高いほどAI生成らしい）。"""
        batch, num_patches = patches.shape[:2]
        flat = patches.reshape(batch * num_patches, *patches.shape[2:]).to(self.device)
        x_prime, y_prime = self.preprocessor.prepare(flat)
        scores = nll_minus_entropy(y_prime, self.model(x_prime), self.truncation_t)
        return scores.reshape(batch, num_patches).mean(dim=1)

    def score_rows(
        self, rows: Sequence[dict[str, Any]], transform: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """画像行のリストにスコアを付ける。戻り値 (scores, labels)。"""
        dataset = MultiPatchDataset(rows, self.sampler, self.num_patches, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(self.config.get("train.num_workers", 0)),
            collate_fn=skip_invalid_collate,
        )
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for batch in loader:
            if batch is None:
                continue
            patches, targets, _ = batch
            scores.append(self.score_patches(patches).cpu().numpy())
            # 共有メモリ上のテンソルを参照したままにするとfdが積み上がるのでコピーする
            labels.append(targets.numpy().copy())
        if not scores:
            raise ValueError(f"{len(rows)}枚すべてが読み込めず、スコアを計算できませんでした")

        all_scores = np.concatenate(scores)
        skipped = len(rows) - len(all_scores)
        if skipped:
            logger.warning("%d枚を評価対象から除外しました（%d枚中）。64px未満か破損した画像です", skipped, len(rows))
        return all_scores, np.concatenate(labels)

    # ------------------------------------------------------------------
    # 閾値と評価
    # ------------------------------------------------------------------

    def calibration_rows(self) -> list[dict[str, Any]]:
        """閾値を決めるための、学習用の写真（学習に使った生成器の実写サブセット）。

        全件は15万枚あるので、seedで固定した無作為抽出で `num_calibration_images` 枚に絞る。
        """
        rows = [
            dict(row)
            for row in self.repo.list_images(
                label="real", split="train", generator_names=real_generator_names(self.config)
            )
        ]
        if len(rows) > self.num_calibration_images:
            rows = random.Random(int(self.config.get("project.seed", 42))).sample(rows, self.num_calibration_images)
        return rows

    def calibrate(self) -> float:
        """学習用の写真のスコアの95パーセンタイルを閾値にする（論文 Appendix B）。"""
        rows = self.calibration_rows()
        if not rows:
            raise ValueError("閾値を決めるための学習用の写真がありません")
        scores, _ = self.score_rows(rows)
        self.threshold = float(np.percentile(scores, 95))
        logger.info("閾値を %.4f にしました（学習用の写真 %d枚の95パーセンタイル）", self.threshold, len(scores))
        return self.threshold

    def _real_generator_for(self, generator: str) -> str:
        if self.config.get("dataset.real_generator_naming", "per_source") == "shared":
            return _REAL_GENERATOR_PREFIX
        return f"{_REAL_GENERATOR_PREFIX}@{generator}"

    def _rows_for(self, generator: str, split: str) -> list[dict[str, Any]]:
        """生成器のAI画像と、それに対応する実写画像を同数ずつ集める。"""
        rows: list[dict[str, Any]] = []
        for label, name in (("ai", generator), ("real", self._real_generator_for(generator))):
            found = self.repo.list_images(label=label, split=split, generator_names=[name], limit=self.max_images)
            rows.extend(dict(row) for row in found)
        return rows

    def run(self, split: str = "test") -> list[dict[str, Any]]:
        """各生成器を判定し、結果の行（論文の値つき）を返す。"""
        if self.threshold is None:
            self.calibrate()

        excluded = set(self.config.get("dataset.excluded_generators", []))
        results: list[dict[str, Any]] = []
        for generator in self.config.get("dataset.eval_generators", []):
            if generator in excluded:
                continue
            rows = self._rows_for(generator, split)
            if not rows:
                logger.warning("評価対象の画像がありません: %s（split=%s）", generator, split)
                continue
            scores, labels = self._score_generator_rows(rows)
            metrics = compute_metrics(scores, labels, self.threshold)
            predictions = scores > self.threshold
            real, ai = labels == 0, labels == 1
            result = {
                "generator": generator,
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "ap": metrics["ap"],
                # どちら側で間違えているかを見るため、クラスごとの正解率も出す
                "real_accuracy": float((~predictions[real]).mean()) if real.any() else None,
                "ai_accuracy": float(predictions[ai].mean()) if ai.any() else None,
                "num_samples": metrics["num_samples"],
                "threshold": self.threshold,
                "paper_accuracy": PAPER_TABLE5_ACCURACY.get(generator),
            }
            logger.info(
                "%s: acc=%.4f auc=%s 実写正解率=%s AI正解率=%s（論文 %s%%）",
                generator, result["accuracy"], _fmt(result["auc"]), _fmt(result["real_accuracy"]),
                _fmt(result["ai_accuracy"]), result["paper_accuracy"],
            )
            results.append(result)
        return results


    def _score_generator_rows(self, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """AI画像と本物を別々に点数づけする。ai_jpeg_quality の指定があれば AI画像だけ JPEG にする。"""
        if self.ai_jpeg_quality is None:
            return self.score_rows(rows)
        ai_rows = [row for row in rows if row["label"] == "ai"]
        real_rows = [row for row in rows if row["label"] != "ai"]
        parts = []
        if ai_rows:
            parts.append(self.score_rows(ai_rows, transform=jpeg_transform(self.ai_jpeg_quality)))
        if real_rows:
            parts.append(self.score_rows(real_rows))
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


_COLUMNS = ["generator", "accuracy", "auc", "ap", "real_accuracy", "ai_accuracy", "num_samples", "threshold", "paper_accuracy"]


def with_mean_row(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成器の平均行を末尾に足す（論文値は、評価した生成器だけで平均し直す）。"""
    if not results:
        return results

    def mean(key: str) -> float | None:
        values = [r[key] for r in results if r[key] is not None]
        return float(np.mean(values)) if values else None

    row = {key: mean(key) for key in _COLUMNS if key not in ("generator", "num_samples", "threshold")}
    row.update(generator="平均", num_samples=sum(r["num_samples"] for r in results), threshold=results[0]["threshold"])
    return [*results, row]


def write_report(
    results: list[dict[str, Any]],
    out_dir: str | Path,
    checkpoint: str | Path,
    ai_jpeg_quality: int | None = None,
) -> Path:
    """CSV と Markdown の表を書き出し、出力先のディレクトリを返す。"""
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = with_mean_row(results)

    with open(out / "one_class.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in _COLUMNS})

    lines = [
        "# 一クラス検出（DCCT†）評価レポート",
        "",
        f"- pθ: `{checkpoint}`",
        f"- 閾値: {rows[0]['threshold']:.4f}（学習用の写真のスコアの95パーセンタイル）" if rows else "",
        "- スコア: パッチごとの mean(NLL − entropy) を16パッチで平均。高いほどAI生成らしい",
        "- 論文値: Appendix B 表5（Accuracy %）。平均は評価した生成器だけで計算し直している",
        *(
            [f"- AI画像だけ JPEG（品質 {ai_jpeg_quality}）に圧縮してから評価した（本物はそのまま）"]
            if ai_jpeg_quality is not None
            else []
        ),
        "",
        "| generator | accuracy | 論文 | auc | 実写の正解率 | AIの正解率 | 枚数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        paper = "" if row["paper_accuracy"] is None else f"{row['paper_accuracy'] / 100:.4f}"
        lines.append(
            f"| {row['generator']} | {_fmt(row['accuracy'])} | {paper} | {_fmt(row['auc'])} | "
            f"{_fmt(row['real_accuracy'])} | {_fmt(row['ai_accuracy'])} | {row['num_samples']} |"
        )
    (out / "one_class.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

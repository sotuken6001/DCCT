"""評価の実行（03_詳細設計書 2節 `Evaluator`）。

- cross-generator評価（Table 1相当）
- ロバスト性評価（Figure 6相当）

結果は `evaluation_results` テーブルに `eval_type` で区別して保存する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import real_generator_names
from ..data.repository import MetadataRepository
from ..experiment.config import Config
from ..preprocess.patch_sampler import jpeg_compress
from .metrics import compute_metrics, tune_threshold
from .predictor import DCCTPredictor, ImageTransform

logger = logging.getLogger(__name__)

_REAL_GENERATOR_PREFIX = "ImageNet(real)"


@dataclass
class GeneratorResult:
    """1生成器分の評価結果。"""

    generator: str
    metrics: dict[str, Any]
    perturbation_type: str = "none"
    perturbation_level: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# 劣化処理はDataLoaderのワーカープロセスへ渡されるため、pickleできる必要がある。
# クロージャ（ローカル関数）はpickleできず `num_workers>0` で
# `Can't get local object` になるので、モジュール直下のクラスとして定義する。


@dataclass(frozen=True)
class DownsampleTransform:
    """倍率 `ratio` に縮小してから元の解像度に戻す（Figure 6のダウンサンプリング）。"""

    ratio: float

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        small = F.interpolate(
            image.unsqueeze(0), scale_factor=self.ratio, mode="bilinear", align_corners=False
        )
        restored = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
        return restored.squeeze(0).clamp(0, 255)


@dataclass(frozen=True)
class JpegTransform:
    """指定QFでJPEG再エンコードする（Figure 6のJPEG圧縮）。"""

    quality: int
    # 色の間引き方。None は PIL の初期設定（4:2:0）
    subsampling: str | None = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return jpeg_compress(image, self.quality, self.subsampling)


def downsample_transform(ratio: float) -> ImageTransform:
    return DownsampleTransform(float(ratio))


def jpeg_transform(quality: int, subsampling: str | None = None) -> ImageTransform:
    return JpegTransform(int(quality), subsampling)


class Evaluator:
    """学習済みモデルを各生成器・各劣化条件で評価し、DBへ保存する。"""

    def __init__(
        self,
        config: Config,
        run_id: int,
        classifier_checkpoint: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.repo = MetadataRepository(config.get("paths.db_path"))
        self.predictor = DCCTPredictor(config, classifier_checkpoint, device)
        self.threshold = float(config.get("inference.threshold", 0.5))
        self.max_images = config.get("evaluate.max_images_per_generator", None)
        # run_ai_jpeg で品質ごとに線を決め直したときの値（{品質: τ}）。決め直さないときは空
        self.ai_jpeg_thresholds: dict[int, float] = {}

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 評価対象の組み立て
    # ------------------------------------------------------------------

    def target_generators(self) -> list[str]:
        """評価対象の生成器（excluded_generators を除く）。"""
        excluded = set(self.config.get("dataset.excluded_generators", []))
        return [g for g in self.config.get("dataset.eval_generators", []) if g not in excluded]

    def _real_generator_for(self, generator: str) -> str:
        """その生成器に対応する実写generator名（ingestの命名規則に合わせる）。"""
        if self.config.get("dataset.real_generator_naming", "per_source") == "shared":
            return _REAL_GENERATOR_PREFIX
        return f"{_REAL_GENERATOR_PREFIX}@{generator}"

    def _rows_for(
        self, generator: str, split: str, limit: int | None | str = "default"
    ) -> list[dict[str, Any]]:
        """生成器のAI画像と、それに対応する実写画像を集める。

        `evaluate.max_images_per_generator` はラベルごとに適用する。まとめてLIMITすると
        片方のクラスだけが残り、AccuracyやAUCが意味を持たなくなるため。

        `limit` に `None` を渡すとその上限を外す（閾値チューニングで使う）。
        """
        if limit == "default":
            limit = self.max_images
        rows: list[dict[str, Any]] = []
        for label, name in (("ai", generator), ("real", self._real_generator_for(generator))):
            found = self.repo.list_images(
                label=label, split=split, generator_names=[name], limit=limit
            )
            rows.extend(dict(row) for row in found)
        return rows

    # ------------------------------------------------------------------
    # 閾値チューニング
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        transform: ImageTransform | None = None,
        apply: bool = True,
        ai_transform: ImageTransform | None = None,
    ) -> float:
        """val split上でYouden指数を最大化する閾値τを決める（test splitには触れない）。

        `transform` を渡すと、その劣化処理を通した val で較正する。ロバスト性評価で
        条件ごとに較正し直すときに使う。`apply=False` なら `self.threshold` は
        書き換えず、求めた値を返すだけにする。`ai_transform` を渡すと、AI画像だけにそれを通す
        （AI画像だけ JPEG にした評価に合わせて、線も同じ条件の val で決めるときに使う）。
        """
        if not self.config.get("evaluate.tune_threshold", True):
            return self.threshold

        # `evaluate.max_images_per_generator` は評価の所要時間を抑えるためのつまみであり、
        # 閾値較正の品質まで変えてしまわないよう、ここでは適用しない。こうしておくと
        # cross_generator と robustness が同じ τ になり、両者を直接比較できる。
        rows: list[dict[str, Any]] = []
        for generator in self.config.get("dataset.train_generators", []):
            rows.extend(self._rows_for(generator, "val", limit=None))
        if not rows:
            logger.warning("val splitに画像がないため、閾値は既定値 %.3f のままにします", self.threshold)
            return self.threshold

        scores, labels = self._score_rows(rows, transform, ai_transform)
        threshold = tune_threshold(scores, labels, default=self.threshold)
        logger.info("閾値τを %.4f にチューニングしました（val %d枚）", threshold, len(labels))
        if apply:
            self.threshold = threshold
        return threshold

    # ------------------------------------------------------------------
    # 各評価モード
    # ------------------------------------------------------------------

    def evaluate_generator(
        self,
        generator: str,
        split: str = "test",
        transform: ImageTransform | None = None,
        threshold: float | None = None,
        ai_transform: ImageTransform | None = None,
    ) -> GeneratorResult | None:
        """`ai_transform` を渡すと、AI画像だけにそれを通す（本物は `transform` だけ）。"""
        rows = self._rows_for(generator, split)
        if not rows:
            logger.warning("評価対象の画像がありません: %s（split=%s）", generator, split)
            return None

        scores, labels = self._score_rows(rows, transform, ai_transform)
        metrics = compute_metrics(scores, labels, self.threshold if threshold is None else threshold)
        logger.info(
            "%s: acc=%.4f auc=%s ap=%s (n=%d)",
            generator,
            metrics["accuracy"],
            f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A",
            f"{metrics['ap']:.4f}" if metrics["ap"] is not None else "N/A",
            metrics["num_samples"],
        )
        return GeneratorResult(generator=generator, metrics=metrics)

    def _score_rows(
        self,
        rows: list[dict[str, Any]],
        transform: ImageTransform | None,
        ai_transform: ImageTransform | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`ai_transform` があれば AI画像だけにそれを通し、本物には `transform` だけを通す。"""
        if ai_transform is None:
            return self.predictor.score_rows(rows, transform=transform)
        parts = []
        ai_rows = [row for row in rows if row["label"] == "ai"]
        real_rows = [row for row in rows if row["label"] != "ai"]
        if ai_rows:
            parts.append(self.predictor.score_rows(ai_rows, transform=ai_transform))
        if real_rows:
            parts.append(self.predictor.score_rows(real_rows, transform=transform))
        return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])

    def _save(self, result: GeneratorResult, eval_type: str, split: str) -> None:
        generator_id = self.repo.generator_id_by_name(
            self.repo.conn.execute("SELECT dataset_id FROM datasets LIMIT 1").fetchone()["dataset_id"],
            result.generator,
        )
        if generator_id is None:
            logger.warning("generators に %s が見つからないため保存をスキップします", result.generator)
            return

        self.repo.save_evaluation_result(
            self.run_id,
            generator_id,
            eval_type,
            split,
            accuracy=result.metrics["accuracy"],
            auc=result.metrics["auc"],
            ap=result.metrics["ap"],
            num_samples=result.metrics["num_samples"],
            perturbation_type=result.perturbation_type,
            perturbation_level=result.perturbation_level,
        )

    def run_cross_generator(self, split: str = "test", eval_type: str = "cross_generator") -> list[GeneratorResult]:
        """Table 1相当: 各生成器に対するAccuracy/AUC/APを算出して保存する。"""
        self.calibrate_threshold()

        results: list[GeneratorResult] = []
        for generator in self.target_generators():
            result = self.evaluate_generator(generator, split)
            if result is None:
                continue
            self._save(result, eval_type, split)
            results.append(result)

        if results:
            mean_accuracy = sum(r.metrics["accuracy"] for r in results) / len(results)
            logger.info("平均Accuracy（%d生成器）: %.4f", len(results), mean_accuracy)
        return results

    def run_ai_jpeg(self, qualities: Sequence[int], split: str = "test") -> list[GeneratorResult]:
        """AI画像だけを JPEG にしてから判定する（本物はそのまま）。

        GenImage の本物は JPEG、AI画像は PNG なので、形式の差で見分けていないかを確かめる。
        既定では線（閾値）を圧縮しない val で決めるので、正解率は参考にならない。AUC を見る。
        `evaluate.ai_jpeg.recalibrate_threshold: true` にすると、品質ごとに「AI画像も同じ品質の
        JPEG にした val」で線を決め直すので、論文と並べられる正解率になる。
        cross_generator の表を書きかえないよう、robustness（perturbation_type='ai_jpeg'）として保存する。
        """
        self.calibrate_threshold()

        recalibrate = bool(self.config.get("evaluate.ai_jpeg.recalibrate_threshold", False))
        # 確認ケ：AI画像の JPEG の色の間引き方を指定する（None は PIL の初期設定 4:2:0）
        subsampling = self.config.get("evaluate.ai_jpeg.subsampling", None)
        if subsampling is not None:
            logger.info("AI画像の JPEG の色の間引き方を %s にします", subsampling)
        self.ai_jpeg_thresholds = {}
        if recalibrate:
            for quality in qualities:
                logger.info("AI画像も JPEG 品質 %d にした val で閾値を決め直します", int(quality))
                self.ai_jpeg_thresholds[int(quality)] = self.calibrate_threshold(
                    apply=False, ai_transform=jpeg_transform(int(quality), subsampling)
                )

        perturbation_type = "ai_jpeg"
        zero = self.predictor.zero_features
        if zero is not None:
            # 片方の出力を 0 にした測り直し（確認イ）は、ふつうの結果と混ざらないよう別の名前で残す
            perturbation_type += f"_zero_{zero}"
        if recalibrate:
            # 線を決め直した結果も、正解率だけが変わるので別の名前で残す
            perturbation_type += "_recal"
        if subsampling is not None:
            # 色の間引き方を変えた結果も別の名前で残す（例: ai_jpeg_recal_s444）
            perturbation_type += "_s" + subsampling.replace(":", "")

        results: list[GeneratorResult] = []
        for generator in self.target_generators():
            for quality in qualities:
                result = self.evaluate_generator(
                    generator,
                    split,
                    ai_transform=jpeg_transform(int(quality), subsampling),
                    threshold=self.ai_jpeg_thresholds.get(int(quality)),
                )
                if result is None:
                    continue
                result.perturbation_type = perturbation_type
                result.perturbation_level = float(quality)
                self._save(result, "robustness", split)
                results.append(result)
        return results

    def run_robustness(self, split: str = "test") -> list[GeneratorResult]:
        """Figure 6相当: JPEG圧縮・ダウンサンプリングに対する精度を測る。"""
        self.calibrate_threshold()

        qualities = self.config.get("evaluate.robustness.jpeg_qualities", [100, 90, 80, 70])
        ratios = self.config.get("evaluate.robustness.downsample_ratios", [0.9, 0.8, 0.7, 0.6])

        conditions: list[tuple[str, float, ImageTransform | None]] = [("none", None, None)]
        conditions += [("jpeg", float(q), jpeg_transform(int(q))) for q in qualities]
        conditions += [("downsample", float(r), downsample_transform(float(r))) for r in ratios]

        # 既定では Algorithm 2 どおり、クリーンなvalで決めた単一のτを全条件に使う。
        # ただしそれだと、劣化でスコア分布がずれた条件では全件が閾値の片側に寄り、
        # 識別力が残っていてもAccuracyが0.5（偶然と同じ）まで落ちる。
        # `recalibrate_threshold: true` にすると、同じ劣化を通したvalで条件ごとに
        # τを取り直すので、劣化そのものの影響だけを見られる。
        recalibrate = bool(self.config.get("evaluate.robustness.recalibrate_threshold", False))
        thresholds: dict[tuple[str, float | None], float] = {}
        if recalibrate:
            for perturbation_type, level, transform in conditions:
                logger.info("閾値を較正します: %s=%s", perturbation_type, level)
                thresholds[(perturbation_type, level)] = self.calibrate_threshold(
                    transform=transform, apply=False
                )

        results: list[GeneratorResult] = []
        for generator in self.target_generators():
            for perturbation_type, level, transform in conditions:
                result = self.evaluate_generator(
                    generator,
                    split,
                    transform=transform,
                    threshold=thresholds.get((perturbation_type, level)),
                )
                if result is None:
                    continue
                result.perturbation_type = perturbation_type
                result.perturbation_level = level
                logger.info("  %s / %s=%s → acc=%.4f", generator, perturbation_type, level, result.metrics["accuracy"])
                self._save(result, "robustness", split)
                results.append(result)
        return results

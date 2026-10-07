"""アブレーション実験の実行（Table 4a〜4d相当 / 03_詳細設計書 4節）。

`configs/ablation/*.yaml` の `ablation.variants` を1件ずつ実行する。各variantは
独立した実験ID（`experiments`）として登録し、`ablation_configs` に条件を残したうえで、
評価結果を `eval_type='ablation'` として保存する。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

from ..experiment.config import Config
from ..experiment.paths import resolve
from ..train.train_stage1 import TrainerStage1
from ..train.train_stage2 import TrainerStage2
from .evaluator import Evaluator

logger = logging.getLogger(__name__)

# variantのキー → configの実際のパス
PARAMETER_PATHS = {
    "use_photo_model": "stage2.use_photo_model",
    "use_ai_model": "stage2.use_ai_model",
    "freeze_conditional_models": "stage2.freeze_conditional_models",
    "use_high_pass": "preprocess.use_high_pass",
    "use_cfa_mask": "preprocess.use_cfa_mask",
    "truncation_t": "preprocess.truncation_t",
}


@dataclass
class AblationResult:
    label: str
    experiment_id: int
    run_id: int
    mean_accuracy: float | None
    parameters: dict[str, Any]


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def build_variant_config(base: Config, variant: dict[str, Any], checkpoint_root: str) -> tuple[Config, dict[str, Any]]:
    """variantの指定をconfigへ反映した派生configを作る。"""
    data = base.as_dict()
    label = variant.get("label", "variant")
    parameters: dict[str, Any] = {}

    for key, value in variant.items():
        if key == "label":
            continue
        path = PARAMETER_PATHS.get(key)
        if path is None:
            raise ValueError(f"未知のアブレーションパラメータです: {key!r}（PARAMETER_PATHS を確認）")
        _set_nested(data, path, value)
        parameters[key] = value

    _set_nested(data, "experiment.name", f"{base.get('experiment.name', 'ablation')}_{label}")
    _set_nested(data, "train.checkpoint_dir", f"{checkpoint_root}/{label}")
    return Config(data), parameters


def run_ablation(
    config: Config,
    *,
    num_epochs: int | None = None,
    max_steps_per_epoch: int | None = None,
    only_labels: list[str] | None = None,
) -> list[AblationResult]:
    """configの `ablation.variants` を順に学習・評価する。"""
    group = config.get("ablation.group")
    variants = config.get("ablation.variants", [])
    retrain_stage1 = bool(config.get("ablation.retrain_stage1", False))
    checkpoint_root = str(resolve(config.get("paths.checkpoint_root", "checkpoints")) / f"ablation_{group}")

    results: list[AblationResult] = []
    for variant in variants:
        label = variant.get("label", "variant")
        if only_labels and label not in only_labels:
            continue

        logger.info("=== Ablation-%s / %s ===", group, label)
        variant_config, parameters = build_variant_config(config, variant, checkpoint_root)

        # Ablation-C のように前処理が変わる条件では、条件付きモデルから学習し直す
        if retrain_stage1:
            for target in ("photo", "ai"):
                stage1_config = Config(variant_config.as_dict())
                _set_nested(
                    stage1_config._data, "train.checkpoint_dir", f"{checkpoint_root}/{label}/stage1_{target}"
                )
                with TrainerStage1(stage1_config, target) as trainer:
                    trainer.fit(num_epochs=num_epochs, max_steps_per_epoch=max_steps_per_epoch)
                _set_nested(
                    variant_config._data,
                    f"stage2.{target}_model_checkpoint",
                    str(trainer.checkpoint_dir / "best.pt"),
                )

        with TrainerStage2(variant_config) as trainer:
            run_id = trainer.fit(num_epochs=num_epochs, max_steps_per_epoch=max_steps_per_epoch)
            experiment_id = trainer.experiment_id
            classifier_checkpoint = trainer.checkpoint_dir / "best.pt"

            # 条件をDBに残す（04_DB設計書 3.9）
            for name, value in parameters.items():
                trainer.repo.save_ablation_config(experiment_id, group, name, str(value))

        with Evaluator(variant_config, run_id, str(classifier_checkpoint)) as evaluator:
            generator_results = evaluator.run_cross_generator(eval_type="ablation")

        mean_accuracy = (
            sum(r.metrics["accuracy"] for r in generator_results) / len(generator_results)
            if generator_results
            else None
        )
        logger.info("Ablation-%s / %s: mAcc=%s", group, label, f"{mean_accuracy:.4f}" if mean_accuracy else "N/A")
        results.append(
            AblationResult(
                label=label,
                experiment_id=experiment_id,
                run_id=run_id,
                mean_accuracy=mean_accuracy,
                parameters=parameters,
            )
        )
    return results

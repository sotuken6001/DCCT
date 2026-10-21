"""Stage II: 二値分類器 gψ の学習（Algorithm 1 後半）。

Frozenにした pθ / qφ から取り出した特徴を連結して分類器に入力し、BCEで学習する。
Ablation-A（使用モデルの選択）と Ablation-D（Frozen / full fine-tuning）を
設定で切り替えられる。
"""

from __future__ import annotations

import logging

import torch

from ..data.dataset import PatchDataset, select_image_rows
from ..experiment import builders
from ..experiment.config import Config
from .base import Trainer

logger = logging.getLogger(__name__)


class TrainerStage2(Trainer):
    """条件付きモデルの特徴から写真/AI生成を判別する分類器を学習する。"""

    stage = "stage2_classifier"
    best_metric_name = "val_accuracy"
    better_is_lower = False

    def setup(self) -> None:
        config = self.config
        self.preprocessor = builders.build_preprocessor(config, self.device)

        self.use_photo_model = bool(config.get("stage2.use_photo_model", True))
        self.use_ai_model = bool(config.get("stage2.use_ai_model", True))
        self.freeze = bool(config.get("stage2.freeze_conditional_models", True))

        self.photo_model = self._load_conditional_model("photo") if self.use_photo_model else None
        self.ai_model = self._load_conditional_model("ai") if self.use_ai_model else None

        reference = self.photo_model or self.ai_model
        in_channels = builders.classifier_input_channels(config, reference)
        self.classifier = builders.build_classifier(config, in_channels).to(self.device)
        self.loss_fn = builders.build_bce_loss(config)

        parameters = list(self.classifier.parameters())
        if not self.freeze:
            # Ablation-D: 条件付きモデルも含めて full fine-tuning する
            for model in (self.photo_model, self.ai_model):
                if model is not None:
                    parameters += list(model.parameters())
        self.optimizer = torch.optim.Adam(
            parameters, lr=float(config.get("train.learning_rate", 1e-4))
        )

        sampler = builders.build_patch_sampler(config)
        limit = config.get("data.max_images", None)
        train_rows = select_image_rows(config, self.repo, label=None, split="train", limit=limit)
        val_rows = select_image_rows(config, self.repo, label=None, split="val", limit=limit)
        logger.info("Stage II: train=%d枚 val=%d枚 入力特徴=%dch", len(train_rows), len(val_rows), in_channels)

        # 実験15：AI画像も JPEG にして本物と形式をそろえる（best.pt を選ぶ val も同じ条件にする）
        ai_transform = builders.build_ai_train_transform(config)
        if ai_transform is not None:
            logger.info(
                "AI画像だけ JPEG（品質 %d〜%d、色の間引き方 %s）にして学習します",
                ai_transform.quality_min, ai_transform.quality_max, ai_transform.subsampling_weights or "PIL の初期設定（4:2:0）",
            )
        # 実験17：本物も PIL の JPEG にかけ直す（元の ImageNet の JPEG との作りの違いを使えなくする）
        real_transform = builders.build_real_train_transform(config)
        if real_transform is not None:
            logger.info("本物も JPEG（品質 %d〜%d）にかけ直して学習します", real_transform.quality_min, real_transform.quality_max)
        self.train_loader = self.make_loader(
            PatchDataset(train_rows, sampler, ai_transform=ai_transform, real_transform=real_transform), shuffle=True
        )
        # 先頭Nバッチだけを検証に使うため、ラベルが偏らないよう決定的にシャッフルする
        self.val_loader = self.make_loader(
            PatchDataset(
                self.prepare_eval_rows(val_rows), sampler, ai_transform=ai_transform, real_transform=real_transform
            ),
            shuffle=False,
        )

    def _load_conditional_model(self, which: str):
        """Stage I のチェックポイントから pθ / qφ を復元する。"""
        model = builders.build_conditional_unet(self.config, self.preprocessor).to(self.device)
        checkpoint = self.config.get(f"stage2.{which}_model_checkpoint", None)

        if checkpoint is None:
            logger.warning(
                "stage2.%s_model_checkpoint が未設定です。ランダム初期化の条件付きモデルを使います"
                "（動作確認用。本番学習ではStage Iのチェックポイントを指定してください）",
                which,
            )
        else:
            from ..experiment.paths import resolve

            checkpoint_path = resolve(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Stage I（{which}）のチェックポイントが見つかりません: {checkpoint_path}\n"
                    f"先に `python -m src.cli train-stage1 --target {which} "
                    f"--config configs/stage1_{'photo' if which == 'photo' else 'ai'}.yaml` を実行するか、"
                    f"configs/stage2_classifier.yaml の stage2.{which}_model_checkpoint を実際の保存先に直してください"
                )
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state["modules"]["conditional_unet"])
            logger.info("%s モデルを読み込みました: %s", which, checkpoint)

        if self.freeze:
            model.freeze()
        return model

    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        modules: dict[str, torch.nn.Module] = {"classifier": self.classifier}
        if not self.freeze:
            if self.photo_model is not None:
                modules["photo_model"] = self.photo_model
            if self.ai_model is not None:
                modules["ai_model"] = self.ai_model
        return modules

    def _features(self, patches: torch.Tensor) -> torch.Tensor:
        x_prime = self.preprocessor.prepare_input(patches)
        features = []
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            if self.photo_model is not None:
                features.append(self.photo_model.extract_feature(x_prime))
            if self.ai_model is not None:
                features.append(self.ai_model.extract_feature(x_prime))
        return torch.cat(features, dim=1)

    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches, targets = batch
        patches = patches.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        logits = self.classifier(self._features(patches))
        return self.loss_fn(logits, targets), logits, targets

    def train_step(self, batch) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss, logits, targets = self._forward(batch)
        loss.backward()
        self.optimizer.step()
        return {"bce_loss": loss.item(), "accuracy": _accuracy(logits, targets)}

    def eval_step(self, batch) -> dict[str, float]:
        loss, logits, targets = self._forward(batch)
        return {"val_bce": loss.item(), "val_accuracy": _accuracy(logits, targets)}


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predicted = (torch.sigmoid(logits) > 0.5).float()
    return (predicted == targets).float().mean().item()

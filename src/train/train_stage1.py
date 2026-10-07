"""Stage I-A / I-B: 条件付き分布モデル pθ / qφ の学習（Algorithm 1 前半）。

pθ は写真データセットのみ、qφ はAI生成データセットのみで**独立に**学習する
（重み共有なし）。収束後は Stage II でFrozenにして使う。
"""

from __future__ import annotations

import logging

import torch

from ..data.dataset import PatchDataset, select_image_rows
from ..experiment import builders
from ..experiment.config import Config
from .base import Trainer

logger = logging.getLogger(__name__)

# --target の値 → (実験stage, 学習に使う画像ラベル)
TARGETS = {
    "photo": ("stage1_photo", "real"),
    "ai": ("stage1_ai", "ai"),
}


class TrainerStage1(Trainer):
    """条件付き分布モデル1本を混合ロジスティックのNLLで学習する。"""

    best_metric_name = "val_nll"

    def __init__(self, config: Config, target: str, device: torch.device | None = None) -> None:
        if target not in TARGETS:
            raise ValueError(f"--target は photo / ai のいずれか: {target!r}")
        self.target = target
        self.stage, self.label = TARGETS[target]
        super().__init__(config, device)

    def setup(self) -> None:
        config = self.config
        self.preprocessor = builders.build_preprocessor(config, self.device)
        self.model = builders.build_conditional_unet(config, self.preprocessor).to(self.device)
        self.loss_fn = builders.build_nll_loss(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(config.get("train.learning_rate", 1e-4))
        )

        sampler = builders.build_patch_sampler(config)
        limit = config.get("data.max_images", None)
        train_rows = select_image_rows(config, self.repo, label=self.label, split="train", limit=limit)
        val_rows = select_image_rows(config, self.repo, label=self.label, split="val", limit=limit)

        logger.info("学習対象 %s: train=%d枚 val=%d枚", self.label, len(train_rows), len(val_rows))
        self.train_loader = self.make_loader(PatchDataset(train_rows, sampler), shuffle=True)
        # 先頭Nバッチだけを検証に使うため、ラベルが偏らないよう決定的にシャッフルする
        self.val_loader = self.make_loader(
            PatchDataset(self.prepare_eval_rows(val_rows), sampler), shuffle=False
        )

    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        return {"conditional_unet": self.model}

    def _forward(self, batch) -> torch.Tensor:
        patches, _ = batch
        patches = patches.to(self.device, non_blocking=True)
        x_prime, y_prime = self.preprocessor.prepare(patches)
        return self.loss_fn(y_prime, self.model(x_prime))

    def train_step(self, batch) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._forward(batch)
        loss.backward()
        self.optimizer.step()
        return {"nll_loss": loss.item()}

    def eval_step(self, batch) -> dict[str, float]:
        return {"val_nll": self._forward(batch).item()}

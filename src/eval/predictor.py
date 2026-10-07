"""推論（Algorithm 2）: 1画像からPパッチを取り、スコアを平均して判定する。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import MultiPatchDataset, skip_invalid_collate
from ..experiment import builders
from ..experiment.config import Config
from ..experiment.paths import resolve

logger = logging.getLogger(__name__)

# 画像全体に適用する劣化処理（ロバスト性評価で使う）
ImageTransform = Callable[[torch.Tensor], torch.Tensor]


class DCCTPredictor:
    """学習済みの pθ / qφ / gψ を読み込み、画像にスコアを付ける。"""

    def __init__(
        self,
        config: Config,
        classifier_checkpoint: str | Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or builders.build_device(config)
        self.preprocessor = builders.build_preprocessor(config, self.device)
        self.sampler = builders.build_patch_sampler(config)
        self.num_patches = int(config.get("inference.num_patches", 16))

        checkpoint_path = classifier_checkpoint or config.get("evaluate.classifier_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError(
                "分類器のチェックポイントが指定されていません"
                "（--run-id か evaluate.classifier_checkpoint を指定してください）"
            )
        state = torch.load(resolve(checkpoint_path), map_location=self.device, weights_only=False)
        modules = state["modules"]

        self.use_photo_model = bool(config.get("stage2.use_photo_model", True))
        self.use_ai_model = bool(config.get("stage2.use_ai_model", True))

        self.photo_model = self._load_model("photo", modules) if self.use_photo_model else None
        self.ai_model = self._load_model("ai", modules) if self.use_ai_model else None

        reference = self.photo_model or self.ai_model
        in_channels = builders.classifier_input_channels(config, reference)
        self.classifier = builders.build_classifier(config, in_channels).to(self.device)
        self.classifier.load_state_dict(modules["classifier"])
        self.classifier.eval()

        logger.info("分類器を読み込みました: %s（入力 %dch）", checkpoint_path, in_channels)

    def _load_model(self, which: str, stage2_modules: dict[str, Any]):
        """条件付きモデルを復元する。

        Ablation-D（full fine-tuning）ではStage IIのチェックポイントに両モデルが
        入っているのでそちらを優先し、Frozen学習ならStage Iのチェックポイントから読む。
        """
        model = builders.build_conditional_unet(self.config, self.preprocessor).to(self.device)
        key = f"{which}_model"
        if key in stage2_modules:
            model.load_state_dict(stage2_modules[key])
            logger.info("%s モデルをStage IIチェックポイントから復元しました", which)
        else:
            path = self.config.get(f"stage2.{which}_model_checkpoint", None)
            if path is None:
                logger.warning(
                    "stage2.%s_model_checkpoint が未設定のため、ランダム初期化のまま評価します", which
                )
            else:
                state = torch.load(resolve(path), map_location=self.device, weights_only=False)
                model.load_state_dict(state["modules"]["conditional_unet"])
        return model.freeze()

    @torch.no_grad()
    def score_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """`[B, P, 3, s, s]` → 画像ごとの平均スコア `[B]`（Algorithm 2）。"""
        batch, num_patches = patches.shape[:2]
        flat = patches.reshape(batch * num_patches, *patches.shape[2:]).to(self.device)

        x_prime = self.preprocessor.prepare_input(flat)
        features = []
        if self.photo_model is not None:
            features.append(self.photo_model.extract_feature(x_prime))
        if self.ai_model is not None:
            features.append(self.ai_model.extract_feature(x_prime))

        scores = self.classifier.predict_proba(torch.cat(features, dim=1))
        return scores.reshape(batch, num_patches).mean(dim=1)

    def score_rows(
        self,
        rows: Sequence[dict[str, Any]],
        transform: ImageTransform | None = None,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """画像行のリストにスコアを付ける。戻り値 (scores, labels)。"""
        dataset = MultiPatchDataset(rows, self.sampler, self.num_patches, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size or int(self.config.get("train.batch_size", 16)),
            shuffle=False,
            num_workers=int(self.config.get("train.num_workers", 0)),
            # 読めない画像は評価対象から外す（別の画像で埋めると二重に数えてしまう）
            collate_fn=skip_invalid_collate,
        )

        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for batch in loader:
            if batch is None:  # バッチ内の全件が除外対象だった
                continue
            patches, targets, _ = batch
            scores.append(self.score_patches(patches).cpu().numpy())
            # ワーカーから返るテンソルは共有メモリ上にあり、fdを1つ消費する。
            # `.numpy()` はそれを参照するだけなので、結果を溜めていくと開いたままの
            # fdが積み上がり `Too many open files` になる。コピーして切り離す。
            labels.append(targets.numpy().copy())

        if not scores:
            raise ValueError(
                f"{len(rows)}枚すべてが読み込めず、スコアを1件も計算できませんでした"
                "（外部SSDがマウントされているか確認してください）"
            )

        all_scores = np.concatenate(scores)
        skipped = len(rows) - len(all_scores)
        if skipped:
            logger.warning(
                "%d枚を評価対象から除外しました（%d枚中）。64px未満か破損した画像です",
                skipped,
                len(rows),
            )
        return all_scores, np.concatenate(labels)

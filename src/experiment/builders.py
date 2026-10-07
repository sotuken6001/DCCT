"""configから前処理・モデルを組み立てるファクトリ。

学習・評価の各エントリポイントが同じ設定解釈を共有するため、組み立てはここに集約する。
"""

from __future__ import annotations

import torch

from ..model.classifier import BinaryClassifier
from ..model.conditional_unet import ConditionalUNet
from ..model.losses import BCELoss, NLLLoss
from ..preprocess.patch_sampler import PatchSampler, RandomJpeg
from ..preprocess.pipeline import DCCTPreprocessor
from .config import Config
from .device import resolve_device


def build_device(config: Config) -> torch.device:
    return resolve_device(
        config.get("device.type", "auto"), config.get("device.enable_mps_fallback", True)
    )


def build_preprocessor(
    config: Config, device: torch.device | None = None, generator: torch.Generator | None = None
) -> DCCTPreprocessor:
    preprocessor = DCCTPreprocessor(
        truncation_t=config.get("preprocess.truncation_t", 7),
        use_cfa_mask=config.get("preprocess.use_cfa_mask", True),
        use_high_pass=config.get("preprocess.use_high_pass", True),
        bayer_pattern=config.get("preprocess.bayer_pattern", "RGGB"),
        target_mode=config.get("model.conditional_unet.target_mode", "single_filter"),
        target_filter=config.get("model.conditional_unet.target_filter", "square5x5"),
        device=device,
        generator=generator,
        filter_mode=config.get("preprocess.filter_mode", "mosaic"),
        residual_scale=config.get("preprocess.residual_scale", 1),
    )
    preprocessor.bank.quantize = config.get("preprocess.quantize_residual", True)
    return preprocessor


def build_patch_sampler(config: Config) -> PatchSampler:
    return PatchSampler(
        patch_size=config.get("preprocess.patch_size", 64),
        jpeg_enabled=config.get("augmentation.jpeg.enabled", True),
        jpeg_probability=config.get("augmentation.jpeg.probability", 0.05),
        jpeg_quality_min=config.get("augmentation.jpeg.quality_min", 70),
        jpeg_quality_max=config.get("augmentation.jpeg.quality_max", 100),
        seed=config.get("project.seed", 42),
    )


def build_ai_train_transform(config: Config) -> RandomJpeg | None:
    """学習で AI画像だけに通す JPEG 再エンコード（`augmentation.ai_jpeg`）。無効なら None。"""
    if not config.get("augmentation.ai_jpeg.enabled", False):
        return None
    return RandomJpeg(
        int(config.get("augmentation.ai_jpeg.quality_min", 70)),
        int(config.get("augmentation.ai_jpeg.quality_max", 95)),
    )


def build_conditional_unet(config: Config, preprocessor: DCCTPreprocessor) -> ConditionalUNet:
    """pθ / qφ を1つ作る（インスタンスごとに独立した重みを持つ）。"""
    return ConditionalUNet(
        in_channels=preprocessor.input_channels,
        target_channels=preprocessor.target_channels,
        num_mixtures=config.get("model.conditional_unet.num_mixtures", 10),
        base_channels=config.get("model.conditional_unet.base_channels", 64),
        feature_source=config.get("model.classifier.feature_source", "mixture_params"),
        mixture_scope=config.get("model.conditional_unet.mixture_scope", "per_pixel"),
        depth=config.get("model.conditional_unet.depth", 2),
    )


def classifier_input_channels(config: Config, model: ConditionalUNet) -> int:
    """Ablation-A の設定に応じた分類器の入力チャンネル数を返す。"""
    use_photo = config.get("stage2.use_photo_model", True)
    use_ai = config.get("stage2.use_ai_model", True)
    num_models = int(bool(use_photo)) + int(bool(use_ai))
    if num_models == 0:
        raise ValueError("stage2.use_photo_model と use_ai_model の少なくとも一方を true にしてください")
    return model.feature_channels * num_models


def build_classifier(config: Config, in_channels: int) -> BinaryClassifier:
    return BinaryClassifier(
        in_channels=in_channels,
        resnet_blocks=config.get("model.classifier.resnet_blocks", 4),
        transformer_layers=config.get("model.classifier.transformer_layers", 2),
        patch_size=config.get("preprocess.patch_size", 64),
    )


def build_nll_loss(config: Config) -> NLLLoss:
    # 残差が量子化されていて混合分布が画素ごとなら、値ごとの確率表を引く実装を使う
    use_lookup = bool(config.get("preprocess.quantize_residual", True)) and (
        config.get("model.conditional_unet.mixture_scope", "per_pixel") == "per_pixel"
    )
    return NLLLoss(
        truncation_t=config.get("preprocess.truncation_t", 7),
        reduction=config.get("train.nll_reduction", "mean"),
        use_lookup=use_lookup,
    )


def build_bce_loss(config: Config) -> BCELoss:
    return BCELoss()

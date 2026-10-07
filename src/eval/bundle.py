"""学習済みモデルを1つのファイルにまとめる・読み込む。

Stage II は pθ / qφ を凍結して学習するので、分類器のチェックポイントには分類器の重みしか
入っていない。推論するには Stage I の pθ / qφ のファイルと設定も別に要る。
`export_bundle` はこれらを1つのファイル（バンドル）にまとめ、`DCCTDetector` はそのファイル
だけで画像を判定できるようにする。学習の最適化状態は入れないので、チェックポイントより小さい。

    detector = DCCTDetector.from_bundle("weights/dcct_A.pt")
    detector.predict("photo.jpg")        # {"prob_ai": 0.93, "label": "ai", "threshold": 0.40}
    detector.embedding("photo.jpg")      # 分類器の最後の層の手前の特徴（ほかのモデルとつなぐ用）
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image

from ..experiment import builders
from ..experiment.config import Config
from ..experiment.paths import PROJECT_ROOT, resolve
from ..preprocess.patch_sampler import load_image_tensor

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = "dcct-bundle-v1"

ImageInput = Union[str, Path, Image.Image, torch.Tensor]


def _conditional_model_state(config: Config, which: str, stage2_modules: dict[str, Any]) -> dict[str, Any]:
    """pθ / qφ の重みを取り出す（full fine-tuning なら Stage II 側、凍結なら Stage I のファイルから）。"""
    key = f"{which}_model"
    if key in stage2_modules:
        return stage2_modules[key]
    path = config.get(f"stage2.{which}_model_checkpoint", None)
    if path is None:
        raise ValueError(f"stage2.{which}_model_checkpoint が設定されていません")
    state = torch.load(resolve(path), map_location="cpu", weights_only=False)
    return state["modules"]["conditional_unet"]


def _relative_to_project(path: str | Path) -> str:
    """プロジェクトの中のパスなら相対パスにする（配るファイルに、学習した PC のユーザー名などを残さない）。"""
    full = resolve(path).resolve()
    try:
        return full.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def export_bundle(
    config: Config,
    classifier_checkpoint: str | Path,
    out_path: str | Path,
    threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """分類器のチェックポイントと、それが使う pθ / qφ・設定を1つのファイルにまとめる。

    `threshold` は判定の線（val で決めた τ）。`predict` の label に使う。
    """
    state = torch.load(resolve(classifier_checkpoint), map_location="cpu", weights_only=False)
    stage2_modules = state["modules"]

    modules: dict[str, Any] = {"classifier": stage2_modules["classifier"]}
    if config.get("stage2.use_photo_model", True):
        modules["photo_model"] = _conditional_model_state(config, "photo", stage2_modules)
    if config.get("stage2.use_ai_model", True):
        modules["ai_model"] = _conditional_model_state(config, "ai", stage2_modules)

    bundle = {
        "format": BUNDLE_FORMAT,
        "config": config.as_dict(),
        "modules": modules,
        "threshold": None if threshold is None else float(threshold),
        "metadata": {
            "classifier_checkpoint": _relative_to_project(classifier_checkpoint),
            "epoch": state.get("epoch"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(metadata or {}),
        },
    }
    out = resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    logger.info("バンドルを書き出しました: %s（%s）", out, ", ".join(modules))
    return out


def _to_tensor(image: ImageInput) -> torch.Tensor:
    """画像のパス・PIL 画像・`[3,H,W]`（0〜255）のテンソルを `[3,H,W]` の float にそろえる。"""
    if isinstance(image, torch.Tensor):
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"テンソルは [3,H,W]（0〜255）で渡してください: {tuple(image.shape)}")
        return image.float()
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float()
    return load_image_tensor(Path(image).read_bytes())


class DCCTDetector:
    """バンドル1つから pθ / qφ / 分類器を復元し、画像を判定する。"""

    def __init__(self, bundle: dict[str, Any], device: torch.device | str | None = None) -> None:
        if bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"DCCT のバンドルではありません（format={bundle.get('format')!r}）")
        self.config = Config(bundle["config"])
        self.device = torch.device(device) if device is not None else builders.build_device(self.config)
        self.threshold = bundle.get("threshold")
        self.metadata = bundle.get("metadata", {})

        self.preprocessor = builders.build_preprocessor(self.config, self.device)
        self.sampler = builders.build_patch_sampler(self.config)
        self.num_patches = int(self.config.get("inference.num_patches", 16))

        modules = bundle["modules"]
        self.photo_model = self._conditional_model(modules.get("photo_model"))
        self.ai_model = self._conditional_model(modules.get("ai_model"))
        reference = self.photo_model or self.ai_model
        if reference is None:
            raise ValueError("バンドルに pθ も qφ も入っていません")
        in_channels = builders.classifier_input_channels(self.config, reference)
        self.classifier = builders.build_classifier(self.config, in_channels).to(self.device)
        self.classifier.load_state_dict(modules["classifier"])
        self.classifier.eval()

    @classmethod
    def from_bundle(cls, path: str | Path, device: torch.device | str | None = None) -> "DCCTDetector":
        bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(bundle, device)

    def _conditional_model(self, state: dict[str, Any] | None):
        if state is None:
            return None
        model = builders.build_conditional_unet(self.config, self.preprocessor).to(self.device)
        model.load_state_dict(state)
        return model.freeze()

    # ------------------------------------------------------------------

    def patches(self, image: ImageInput) -> torch.Tensor:
        """画像から `[P,3,s,s]` のパッチを切り出す。同じ画像なら毎回同じ位置になる。"""
        tensor = _to_tensor(image)
        key = str(image) if isinstance(image, (str, Path)) else "image"
        return self.sampler.sample_test(tensor, self.num_patches, image_key=key)

    @torch.no_grad()
    def patch_features(self, image: ImageInput) -> torch.Tensor:
        """分類器に入る特徴マップ `[P,C,s,s]`（pθ / qφ の混合分布パラメータを並べたもの）。"""
        x_prime = self.preprocessor.prepare_input(self.patches(image).to(self.device))
        features = [model.extract_feature(x_prime) for model in (self.photo_model, self.ai_model) if model is not None]
        return torch.cat(features, dim=1)

    @torch.no_grad()
    def score(self, image: ImageInput) -> float:
        """AI生成らしさ（0〜1）。パッチごとのスコアの平均（論文 Algorithm 2）。"""
        return float(self.classifier.predict_proba(self.patch_features(image)).mean())

    def predict(self, image: ImageInput) -> dict[str, Any]:
        """スコアと、バンドルに入っている線（閾値）で決めたラベルを返す。"""
        prob = self.score(image)
        threshold = 0.5 if self.threshold is None else self.threshold
        return {"prob_ai": prob, "label": "ai" if prob > threshold else "real", "threshold": threshold}

    @torch.no_grad()
    def embedding(self, image: ImageInput) -> torch.Tensor:
        """分類器の最後の層の手前の特徴 `[D]`（パッチの平均）。ほかのモデルの特徴とつなぐのに使う。"""
        return self.classifier.embed(self.patch_features(image)).mean(dim=0)

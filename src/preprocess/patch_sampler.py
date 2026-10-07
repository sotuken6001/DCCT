"""パッチサンプリングと学習時のJPEG拡張（03_詳細設計書 3節 `PatchSampler` / FR-10）。

画素値は0〜255スケールのfloatのまま扱う（truncation閾値 t=7 がこのスケールの値であるため）。
"""

from __future__ import annotations

import hashlib
import io

import torch
from PIL import Image

_MAX_SEED = 2**31 - 1


def _stable_seed(seed: int, image_key: str | int) -> int:
    """`(seed, image_key)` から実行をまたいで同じ値になるシードを作る。

    組み込みの `hash()` は文字列に対してプロセスごとにランダム化される
    （PYTHONHASHSEED）ため、これを使うと同じ画像でも実行のたびにパッチ位置が
    変わり、評価結果が再現しない。ダイジェストから作れば常に同じ値になる。
    """
    digest = hashlib.blake2b(f"{seed}:{image_key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MAX_SEED


def load_image_tensor(data: bytes) -> torch.Tensor:
    """画像バイト列を `[3,H,W]`（0〜255のfloat）へ変換する。"""
    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")
        tensor = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(rgb.height, rgb.width, 3)
    return tensor.permute(2, 0, 1).float()


def jpeg_compress(image: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG再エンコードを通した画像を返す（学習時のベナイン摂動／ロバスト性評価に使う）。"""
    array = image.clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return load_image_tensor(buffer.getvalue())


class RandomJpeg:
    """品質を [quality_min, quality_max] から毎回ランダムに選んで JPEG 再エンコードする。

    GenImage は本物が JPEG、AI画像が PNG なので、AI画像にだけかけて形式の差をなくす（実験15）。
    DataLoader のワーカーに渡すため、関数ではなく pickle できるクラスにしている。
    """

    def __init__(self, quality_min: int, quality_max: int) -> None:
        if not 1 <= quality_min <= quality_max <= 100:
            raise ValueError(f"JPEG の品質は 1 <= min <= max <= 100: {quality_min}, {quality_max}")
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def pick_quality(self) -> int:
        return int(torch.randint(self.quality_min, self.quality_max + 1, (1,)).item())

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return jpeg_compress(image, self.pick_quality())


class PatchSampler:
    """学習時はランダムに1枚、推論時はP枚のパッチを切り出す。

    Args:
        patch_size: パッチの一辺 s（既定64）
        jpeg_enabled / jpeg_probability / jpeg_quality_min / jpeg_quality_max:
            学習時のJPEG拡張（QF〜U(70,100)、適用確率5%）
        seed: 推論時のパッチ位置を画像ごとに決定的にするための基準シード
    """

    def __init__(
        self,
        patch_size: int = 64,
        jpeg_enabled: bool = True,
        jpeg_probability: float = 0.05,
        jpeg_quality_min: int = 70,
        jpeg_quality_max: int = 100,
        seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.jpeg_enabled = jpeg_enabled
        self.jpeg_probability = jpeg_probability
        self.jpeg_quality_min = jpeg_quality_min
        self.jpeg_quality_max = jpeg_quality_max
        self.seed = seed

    def _crop(self, image: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        height, width = image.shape[-2:]
        size = self.patch_size
        if height < size or width < size:
            raise ValueError(
                f"画像がパッチサイズ {size} より小さいです: {height}x{width}"
                "（ingest時に is_croppable=0 として除外されるはずの画像です）"
            )
        top = int(torch.randint(0, height - size + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, width - size + 1, (1,), generator=generator).item())
        return image[..., top : top + size, left : left + size]

    def augment(self, image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """確率 `jpeg_probability` でJPEG圧縮を適用する（FR-10）。"""
        if not self.jpeg_enabled or self.jpeg_probability <= 0:
            return image
        if torch.rand(1, generator=generator).item() >= self.jpeg_probability:
            return image

        span = self.jpeg_quality_max - self.jpeg_quality_min + 1
        quality = self.jpeg_quality_min + int(torch.randint(0, span, (1,), generator=generator).item())
        return jpeg_compress(image, quality)

    def sample_train(
        self, image: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """学習用に1パッチ切り出す（JPEG拡張つき）。戻り値 `[3,s,s]`。"""
        return self._crop(self.augment(image, generator), generator)

    def sample_test(
        self, image: torch.Tensor, num_patches: int = 16, image_key: str | int | None = None
    ) -> torch.Tensor:
        """推論用にP枚のパッチを切り出す。戻り値 `[P,3,s,s]`。

        `image_key`（画像IDやパス）を与えると、その画像に対するパッチ位置が毎回同じになる
        （評価の再現性のため。NFR-1）。
        """
        generator = torch.Generator()
        if image_key is None:
            generator.seed()
        else:
            generator.manual_seed(_stable_seed(self.seed, image_key))
        return torch.stack([self._crop(image, generator) for _ in range(num_patches)])

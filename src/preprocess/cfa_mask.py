"""Bayer CFAマスク（03_詳細設計書 3.1 `CFAMask`）。

カメラのカラーフィルタアレイは画素ごとに1色しか観測しない。本モジュールはRGB画像から
「観測される1ch（x）」と「隠された残り2ch（y）」を分離する。デモザイキングが作る
色チャンネル間の相関を学習対象にするための前処理である。

y のチャンネル順は、常に残ったチャンネルをRGBのインデックス昇順に並べる:

    観測がR → y = (G, B)
    観測がG → y = (R, B)
    観測がB → y = (R, G)

Ablation-B（CFAマスクなし）では `RandomChannelMask` に差し替える。どちらも
`apply(image) -> (x, y)` の同じインタフェースを持つ。
"""

from __future__ import annotations

import torch

# 2×2周期のBayer配列。値はRGBのチャンネルインデックス（R=0, G=1, B=2）。
BAYER_PATTERNS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "RGGB": ((0, 1), (1, 2)),
    "BGGR": ((2, 1), (1, 0)),
    "GRBG": ((1, 0), (2, 1)),
    "GBRG": ((1, 2), (0, 1)),
}

# 観測チャンネルごとの「残り2ch」（インデックス昇順）
_REMAINING = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def _split_by_index(image: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """画素ごとの観測チャンネルインデックスから x と y を取り出す。

    image: [..., 3, H, W] / observed: [..., 1, H, W]（値は0/1/2）
    """
    x = torch.gather(image, -3, observed)

    # 残り2chのインデックスマップを作る
    remaining = torch.empty_like(observed).expand(*observed.shape[:-3], 2, *observed.shape[-2:]).clone()
    for channel, (first, second) in _REMAINING.items():
        hit = observed.squeeze(-3) == channel
        remaining[..., 0, :, :][hit] = first
        remaining[..., 1, :, :][hit] = second

    y = torch.gather(image, -3, remaining)
    return x, y


class CFAMask:
    """Bayer配列に従って観測ch（x）と欠損ch（y）を分ける。"""

    def __init__(self, pattern: str = "RGGB") -> None:
        if pattern not in BAYER_PATTERNS:
            raise ValueError(f"未知のBayerパターンです: {pattern!r}（{list(BAYER_PATTERNS)} のいずれか）")
        self.pattern = pattern
        self._tile = torch.tensor(BAYER_PATTERNS[pattern], dtype=torch.long)

    def observed_index(self, height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
        """[1,H,W] の観測チャンネルインデックスマップを返す。"""
        tile = self._tile.to(device) if device is not None else self._tile
        reps = (height + 1) // 2, (width + 1) // 2
        return tile.repeat(reps)[:height, :width].unsqueeze(0)

    def apply(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RGB画像を (x, y) に分ける。

        image: `[3,H,W]` または `[B,3,H,W]`
        戻り値: x=`[...,1,H,W]`（観測ch）, y=`[...,2,H,W]`（隠されたch）
        """
        if image.shape[-3] != 3:
            raise ValueError(f"RGB画像（3ch）が必要です: shape={tuple(image.shape)}")

        height, width = image.shape[-2:]
        observed = self.observed_index(height, width, image.device)
        if image.dim() == 4:
            observed = observed.unsqueeze(0).expand(image.shape[0], -1, -1, -1)
        return _split_by_index(image, observed)


class RandomChannelMask:
    """Ablation-B用: 画素ごとにランダムな1chを観測とする（CFAの空間構造を壊した対照条件）。"""

    def __init__(self, generator: torch.Generator | None = None) -> None:
        self.generator = generator

    def apply(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-3] != 3:
            raise ValueError(f"RGB画像（3ch）が必要です: shape={tuple(image.shape)}")

        shape = (*image.shape[:-3], 1, *image.shape[-2:])
        observed = torch.randint(
            0, 3, shape, generator=self.generator, device=image.device, dtype=torch.long
        )
        return _split_by_index(image, observed)


def build_mask(use_cfa_mask: bool = True, pattern: str = "RGGB", generator: torch.Generator | None = None):
    """configの `preprocess.use_cfa_mask` に応じてマスクを返す（Ablation-B）。"""
    return CFAMask(pattern) if use_cfa_mask else RandomChannelMask(generator)

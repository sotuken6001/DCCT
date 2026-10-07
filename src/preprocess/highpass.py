"""SRM由来の30種ハイパスフィルタとtruncation（03_詳細設計書 3.2 `HighPassFilterBank`）。

論文Fig.7のプロトタイプカーネルとその回転で30種を構成する方針（03_詳細設計書 7.2）に従い、
Fridrich & Kodovsky (2012) "Rich Models for Steganalysis of Digital Images" の
残差フィルタ群から以下の30種を5×5カーネルとして構成する。

    1次微分 8方向                  8
    2次微分 4方向                  4
    3次微分 8方向                  8
    EDGE3x3 4回転                  4
    SQUARE3x3                      1
    EDGE5x5 4回転                  4
    SQUARE5x5                      1
    ------------------------------ 30

各カーネルはSRMと同じく正規化係数 q（予測子の重みの総和）で割る。画素値は0〜255スケールの
まま扱う前提で、truncation閾値 t=7 はこのスケールに対する値である。

SRMの残差量子化 `trunc(round(K*I/q), T)` に倣い、既定では q で割った後に丸めてから
[-t, t] にクリップする。これにより残差は {-7, ..., 7} の15値をとる離散量となり、
条件付きモデルの尤度を離散化ロジスティック混合（PixelCNN++）で書ける。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 8近傍の方向ベクトル (dy, dx)
_DIRECTIONS_8 = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# 2次微分は180度回転が同じものになるため4方向
_DIRECTIONS_4 = _DIRECTIONS_8[:4]

_EDGE3 = [[-1, 2, -1], [2, -4, 2], [0, 0, 0]]
_SQUARE3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
_EDGE5 = [
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
_SQUARE5 = [
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1],
]


def _embed(kernel: list[list[int]], size: int = 5) -> torch.Tensor:
    """小さいカーネルを size×size の中央に配置する。"""
    k = torch.tensor(kernel, dtype=torch.float32)
    out = torch.zeros(size, size)
    offset = (size - k.shape[0]) // 2
    out[offset : offset + k.shape[0], offset : offset + k.shape[1]] = k
    return out


def _directional(coeffs: list[float], offsets: list[int], direction: tuple[int, int]) -> torch.Tensor:
    """中心からの相対位置 `offsets` に沿って係数を並べた5×5カーネルを作る。"""
    out = torch.zeros(5, 5)
    dy, dx = direction
    for coeff, offset in zip(coeffs, offsets):
        y, x = 2 + dy * offset, 2 + dx * offset
        out[y, x] += coeff
    return out


def _rotations(kernel: list[list[int]]) -> list[torch.Tensor]:
    """90度ずつ回転した4枚を返す。"""
    base = _embed(kernel)
    return [torch.rot90(base, k) for k in range(4)]


def build_srm_kernels() -> tuple[torch.Tensor, list[str]]:
    """30種のハイパスフィルタ（正規化済み）とその名前を返す。

    戻り値: kernels `[30,1,5,5]`, names（長さ30）
    """
    kernels: list[torch.Tensor] = []
    names: list[str] = []

    # 1次微分: 中心 -1、隣接 +1（q=1）
    for i, direction in enumerate(_DIRECTIONS_8):
        kernels.append(_directional([-1.0, 1.0], [0, 1], direction))
        names.append(f"1st_d{i}")

    # 2次微分: [1, -2, 1]（q=2）
    for i, direction in enumerate(_DIRECTIONS_4):
        kernels.append(_directional([1.0, -2.0, 1.0], [-1, 0, 1], direction) / 2.0)
        names.append(f"2nd_d{i}")

    # 3次微分: [1, -3, 3, -1]（q=3）
    for i, direction in enumerate(_DIRECTIONS_8):
        kernels.append(_directional([1.0, -3.0, 3.0, -1.0], [-1, 0, 1, 2], direction) / 3.0)
        names.append(f"3rd_d{i}")

    # EDGE3x3 / SQUARE3x3（q=4）
    for i, kernel in enumerate(_rotations(_EDGE3)):
        kernels.append(kernel / 4.0)
        names.append(f"edge3x3_r{i}")
    kernels.append(_embed(_SQUARE3) / 4.0)
    names.append("square3x3")

    # EDGE5x5 / SQUARE5x5（q=12）
    for i, kernel in enumerate(_rotations(_EDGE5)):
        kernels.append(kernel / 12.0)
        names.append(f"edge5x5_r{i}")
    kernels.append(torch.tensor(_SQUARE5, dtype=torch.float32) / 12.0)
    names.append("square5x5")

    stacked = torch.stack(kernels).unsqueeze(1)  # [30,1,5,5]
    return stacked, names


class HighPassFilterBank:
    """30種のハイパスフィルタを適用し、残差を[-t, t]にクリップする。"""

    def __init__(
        self,
        truncation_t: float = 7.0,
        device: torch.device | None = None,
        enabled: bool = True,
        quantize: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        self.truncation_t = float(truncation_t)
        # 丸める前に残差を何倍するか。2 なら元の値で 0.5 刻みになる（t も倍率をかけた後の値で書く）
        if residual_scale <= 0:
            raise ValueError(f"residual_scale は正の数: {residual_scale!r}")
        self.residual_scale = float(residual_scale)
        # SRMと同じく残差を整数へ量子化する（離散化ロジスティック混合の前提）
        self.quantize = quantize
        # Ablation-B: enabled=False のときは素通し（IDフィルタ相当）にする
        self.enabled = enabled
        self.kernels, self.names = build_srm_kernels()
        if device is not None:
            self.kernels = self.kernels.to(device)
        self.M = self.kernels.shape[0]

    def to(self, device: torch.device) -> "HighPassFilterBank":
        self.kernels = self.kernels.to(device)
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """`[...,C,H,W]` に30種のフィルタを適用し `[...,30*C,H,W]` を返す。

        `enabled=False` の場合は入力をそのまま返す（Ablation-B「ハイパスフィルタなし」）。
        チャンネル順は (c0のフィルタ0..29, c1のフィルタ0..29, ...) となる。
        """
        if not self.enabled:
            return self.truncate(x)

        squeezed = x.dim() == 3
        if squeezed:
            x = x.unsqueeze(0)

        batch, channels = x.shape[0], x.shape[-3]
        kernels = self.kernels.to(x.dtype).to(x.device)
        # チャンネルごとに同じ30種を適用する（グループ化畳み込み）
        weight = kernels.repeat(channels, 1, 1, 1)  # [30*C,1,5,5]
        padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
        out = F.conv2d(padded, weight, groups=channels)

        out = self.truncate(out)
        return out.squeeze(0) if squeezed else out

    def truncate(self, x: torch.Tensor) -> torch.Tensor:
        """残差を residual_scale 倍し、（必要なら丸めてから）[-t, t]にクリップする。"""
        if self.residual_scale != 1.0:
            x = x * self.residual_scale
        if self.quantize:
            x = torch.round(x)
        return torch.clamp(x, -self.truncation_t, self.truncation_t)

    def kernel_by_name(self, name: str) -> torch.Tensor:
        """名前で1枚のカーネル `[1,1,5,5]` を取り出す（2ch目標残差の生成に使う）。"""
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise ValueError(f"未知のカーネル名です: {name!r}") from exc
        return self.kernels[index : index + 1]

    def apply_single(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """1種類のカーネルだけを各チャンネルに適用する（チャンネル数を保つ）。"""
        if not self.enabled:
            return self.truncate(x)

        squeezed = x.dim() == 3
        if squeezed:
            x = x.unsqueeze(0)
        channels = x.shape[-3]
        weight = self.kernel_by_name(name).to(x.dtype).to(x.device).repeat(channels, 1, 1, 1)
        padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
        out = self.truncate(F.conv2d(padded, weight, groups=channels))
        return out.squeeze(0) if squeezed else out

"""前処理パイプライン: パッチ → CFAマスク → ハイパスフィルタ → truncation。

Algorithm 1 / 2 の各行に対応する:

    I~ ← RandomCrop(I, s)
    x  ← CFAMask適用(I~)             → 観測1ch
    y  ← PackMissingChannels(I~)      → 隠された2ch
    x' ← Truncate(Stack([h_m * x]))   → 30ch（条件付きモデルの入力）
    y' ← Truncate(...)                → 2ch（条件付きモデルの予測対象）

y' の作り方は論文 Algorithm 1 line 8 に従い、x' と同じく30種すべてを適用する。

    filter_bank（既定・論文準拠）: 30種すべてを y に適用して 60ch にする
    single_filter:                target_filter の1種類のみ適用し 2ch に保つ（比較用）

ハイパスフィルタのかけ方（`filter_mode`）は論文に書かれていないため、3通りを選べる。

    mosaic（既定）: 色が混ざったモザイク x（と y）にそのままかける。隣の画素との色の違いが
                    大きく出て、t=7 では値の6〜7割が ±t に張りつく（論文の「大部分がゼロの
                    まわりに集まる」と合わない）
    sublattice:     2×2 の位置ごとの4つの格子（半分の大きさ）に分け、同じ色の画素どうしで
                    かけてから元の並びに戻す。モザイクに実際にある画素だけを使う
    per_color:      R・G・B それぞれの画像にかけてから、Bayer の位置で観測色と残り2色を選ぶ。
                    x' を作るのに、本来は隠れている画素の値も使う点に注意
"""

from __future__ import annotations

import torch

from .cfa_mask import _REMAINING, build_mask
from .highpass import HighPassFilterBank

FILTER_MODES = ("mosaic", "sublattice", "per_color")


class DCCTPreprocessor:
    """RGBパッチから (x', y') を作る。"""

    def __init__(
        self,
        truncation_t: float = 7.0,
        use_cfa_mask: bool = True,
        use_high_pass: bool = True,
        bayer_pattern: str = "RGGB",
        target_mode: str = "filter_bank",
        target_filter: str = "square5x5",
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        filter_mode: str = "mosaic",
        residual_scale: float = 1.0,
    ) -> None:
        if target_mode not in ("single_filter", "filter_bank"):
            raise ValueError(f"未知の target_mode です: {target_mode!r}")
        if filter_mode not in FILTER_MODES:
            raise ValueError(f"未知の filter_mode です: {filter_mode!r}（{FILTER_MODES} のいずれか）")
        if filter_mode != "mosaic" and (not use_cfa_mask or target_mode != "filter_bank"):
            # どちらも Bayer の2×2周期と、y にも30種すべてをかけることを前提にしている
            raise ValueError(
                f"filter_mode={filter_mode!r} は use_cfa_mask=true かつ target_mode=filter_bank でのみ使えます"
            )

        self.mask = build_mask(use_cfa_mask, bayer_pattern, generator)
        self.bank = HighPassFilterBank(
            truncation_t, device=device, enabled=use_high_pass, residual_scale=residual_scale
        )
        self.target_mode = target_mode
        self.target_filter = target_filter
        self.filter_mode = filter_mode

    @property
    def input_channels(self) -> int:
        """x' のチャンネル数（条件付きモデルの入力チャンネル数）。"""
        return self.bank.M if self.bank.enabled else 1

    @property
    def target_channels(self) -> int:
        """y' のチャンネル数（条件付きモデルの予測対象チャンネル数）。"""
        if self.target_mode == "single_filter" or not self.bank.enabled:
            return 2
        return 2 * self.bank.M

    def to(self, device: torch.device) -> "DCCTPreprocessor":
        self.bank.to(device)
        return self

    def __call__(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prepare(patch)

    def prepare(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`[...,3,s,s]`（0〜255）→ (x' `[...,30,s,s]`, y' `[...,2,s,s]`)。"""
        if self.filter_mode == "per_color" and self.bank.enabled:
            return self._per_color(patch)

        x, y = self.mask.apply(patch)
        if self.filter_mode == "sublattice":
            return self._per_sublattice(x), self._per_sublattice(y)

        x_prime = self.bank.apply(x)
        if self.target_mode == "filter_bank":
            y_prime = self.bank.apply(y)
        else:
            y_prime = self.bank.apply_single(y, self.target_filter)
        return x_prime, y_prime

    def prepare_input(self, patch: torch.Tensor) -> torch.Tensor:
        """Stage II / 推論で使う x' のみを作る。"""
        if self.filter_mode != "mosaic":
            return self.prepare(patch)[0]
        x, _ = self.mask.apply(patch)
        return self.bank.apply(x)

    def _per_sublattice(self, t: torch.Tensor) -> torch.Tensor:
        """2×2 の位置ごとの格子（どれも1色だけ）に分けてフィルタをかけ、元の並びに戻す。"""
        squeezed = t.dim() == 3
        if squeezed:
            t = t.unsqueeze(0)
        planes = {(dy, dx): self.bank.apply(t[..., dy::2, dx::2]) for dy in (0, 1) for dx in (0, 1)}
        first = planes[(0, 0)]
        out = first.new_empty(*first.shape[:-2], t.shape[-2], t.shape[-1])
        for (dy, dx), plane in planes.items():
            out[..., dy::2, dx::2] = plane
        return out.squeeze(0) if squeezed else out

    def _per_color(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """R・G・B それぞれの残差を作り、Bayer の位置で観測色（x'）と残り2色（y'）を選ぶ。"""
        squeezed = image.dim() == 3
        if squeezed:
            image = image.unsqueeze(0)
        batch, _, height, width = image.shape
        m = self.bank.M
        residual = self.bank.apply(image).reshape(batch, 3, m, height, width)
        observed = self.mask.observed_index(height, width, image.device).squeeze(0)   # [H,W]

        x_prime = residual.new_empty(batch, m, height, width)
        y_prime = residual.new_empty(batch, 2, m, height, width)
        for channel, (first, second) in _REMAINING.items():
            hit = observed == channel
            x_prime[:, :, hit] = residual[:, channel][:, :, hit]
            y_prime[:, 0][:, :, hit] = residual[:, first][:, :, hit]
            y_prime[:, 1][:, :, hit] = residual[:, second][:, :, hit]
        # y' のチャンネル順は mosaic と同じ（1つ目の残り色の30種、2つ目の残り色の30種）
        y_prime = y_prime.reshape(batch, 2 * m, height, width)
        if squeezed:
            return x_prime.squeeze(0), y_prime.squeeze(0)
        return x_prime, y_prime

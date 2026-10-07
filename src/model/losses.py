"""損失関数（03_詳細設計書 3.4 `NLLLoss` / Stage II のBCE）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditional_unet import MixtureParams

# 数値的に潰れた確率を避けるための下限
_EPS = 1e-12


class NLLLoss(nn.Module):
    """離散化ロジスティック混合分布の負の対数尤度（Stage I-A / I-B）。

    y' はSRM残差を丸めて[-t, t]にクリップした離散値（既定では{-7,...,7}の15値）なので、
    PixelCNN++と同じく幅1のビンに対する確率として尤度を計算する。

    Args:
        truncation_t: 残差のクリップ範囲。端のビンは片側の裾全体を確率質量とする
        reduction: `mean`（要素平均・既定）/ `sum`（全画素総和）/ `sum_per_sample`
            03_詳細設計書 3.4 は総和と記載しているが、学習の安定性のため既定は要素平均とし、
            `sum` でも計算できるようにしている（勾配方向は同じ）
    """

    def __init__(
        self, truncation_t: float = 7.0, reduction: str = "mean", use_lookup: bool = False
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "sum_per_sample"):
            raise ValueError(f"未知の reduction です: {reduction!r}")
        self.truncation_t = float(truncation_t)
        self.reduction = reduction
        # 量子化済みの残差 × 画素ごとの混合分布なら、取りうる値の表を引く実装に切り替える
        self.use_lookup = use_lookup

    @property
    def num_values(self) -> int:
        """量子化された残差が取りうる値の数（既定 t=7 なら {-7,...,7} の15値）。"""
        return int(2 * self.truncation_t + 1)

    def log_prob(self, y_prime: torch.Tensor, params: MixtureParams) -> torch.Tensor:
        """画素・チャンネルごとの対数尤度 `[B,C,H,W]` を返す。"""
        if self.use_lookup and params.mu.shape[2] == 1:
            return self._log_prob_via_lookup(y_prime, params)
        return self._log_prob_direct(y_prime, params)

    def _log_prob_via_lookup(self, y_prime: torch.Tensor, params: MixtureParams) -> torch.Tensor:
        """取りうる値ごとの確率表を作ってから引く。

        論文準拠の設定では y' は30種のフィルタ × 2色 = 60チャンネルになるが、値自体は
        量子化されて {-t, ..., t} の 2t+1 通りしかない。混合分布が画素ごとに1つ
        （mixture_scope=per_pixel）なら、全チャンネルが同じ分布を共有するので、
        チャンネル数ぶん計算する代わりに 2t+1 通りの表を1度だけ作って引けばよい。
        t=7・60チャンネルなら計算量とメモリが 15/60 = 1/4 になる（結果は厳密に一致）。
        """
        values = torch.arange(
            -self.truncation_t, self.truncation_t + 1, device=y_prime.device, dtype=params.mu.dtype
        )
        # [1,1,V,1,1] と [B,K,1,H,W] のブロードキャストで [B,K,V,H,W]
        table = self._component_log_prob(values.view(1, 1, -1, 1, 1), params)
        log_table = torch.logsumexp(params.log_w + table, dim=1)     # [B,V,H,W]

        index = (y_prime + self.truncation_t).round().long().clamp(0, self.num_values - 1)
        return log_table.gather(1, index)

    def _log_prob_direct(self, y_prime: torch.Tensor, params: MixtureParams) -> torch.Tensor:
        target = y_prime.unsqueeze(1)                    # [B,1,C,H,W]
        log_prob = self._component_log_prob(target, params)
        return torch.logsumexp(params.log_w + log_prob, dim=1)

    def _component_log_prob(self, target: torch.Tensor, params: MixtureParams) -> torch.Tensor:
        """混合成分ごとの離散化ロジスティック対数確率を返す。"""
        centered = target - params.mu
        inv_s = torch.exp(-params.log_s)

        plus = inv_s * (centered + 0.5)
        minus = inv_s * (centered - 0.5)

        log_cdf_plus = F.logsigmoid(plus)                # 下端ビン: (-inf, v+0.5]
        log_one_minus_cdf_minus = F.logsigmoid(-minus)   # 上端ビン: [v-0.5, inf)
        cdf_delta = torch.sigmoid(plus) - torch.sigmoid(minus)

        # ビン幅に対してスケールが極端に小さい場合の代替（中心での密度で近似）
        mid = inv_s * centered
        log_pdf_mid = mid - params.log_s - 2.0 * F.softplus(mid)

        log_prob = torch.where(
            cdf_delta > 1e-5, torch.log(cdf_delta.clamp(min=_EPS)), log_pdf_mid
        )
        log_prob = torch.where(target < -self.truncation_t + 1e-3, log_cdf_plus, log_prob)
        return torch.where(target > self.truncation_t - 1e-3, log_one_minus_cdf_minus, log_prob)

    def forward(self, y_prime: torch.Tensor, params: MixtureParams) -> torch.Tensor:
        log_prob = self.log_prob(y_prime, params)
        if self.reduction == "mean":
            return -log_prob.mean()
        if self.reduction == "sum":
            return -log_prob.sum()
        return -log_prob.flatten(1).sum(dim=1).mean()


class BCELoss(nn.Module):
    """二値分類器 gψ の損失（Stage II）。ロジットを受け取る数値安定版。"""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits.flatten(), target.flatten().to(logits.dtype), reduction=self.reduction
        )

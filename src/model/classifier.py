"""二値分類器 gψ（03_詳細設計書 3.5 `BinaryClassifier`）。

pθ・qφ から取り出した特徴をチャンネル方向に連結したものを入力とし、
4段のResNetブロック → 2層のTransformer Encoder → 全結合層 でスコアを出す。

【03_詳細設計書 7.4 トークン化方法の確定】
ResNetブロックで 64×64 → 4×4 まで空間解像度を落とし、残った 4×4 = 16 個の
空間位置をそれぞれ1トークン（次元 = 最終段のチャンネル数）として扱う。
学習可能な位置埋め込みを加え、Transformer出力を平均プーリングして全結合層に渡す。
シグモイドは損失側（BCEWithLogits）に含めるため、`forward` はロジットを返す。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """stride=2 で空間解像度を半分にする残差ブロック。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class BinaryClassifier(nn.Module):
    """特徴マップ → 写真(0) / AI生成(1) のロジット。

    Args:
        in_channels: 入力特徴のチャンネル数（案A・pθ+qφなら 3*K*2*2 = 120）
        resnet_blocks: ResNetブロック段数
        transformer_layers: Transformer Encoder の層数
        base_channels: 最初のResNetブロックの出力チャンネル数
        num_heads: Multi-head Attention のヘッド数
        patch_size: 入力特徴マップの一辺（トークン数の計算に使う）
    """

    def __init__(
        self,
        in_channels: int,
        resnet_blocks: int = 4,
        transformer_layers: int = 2,
        base_channels: int = 64,
        num_heads: int = 8,
        patch_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        blocks: list[nn.Module] = []
        channels = in_channels
        for i in range(resnet_blocks):
            out_channels = base_channels * (2 ** min(i, 2))  # 64, 128, 256, 256
            blocks.append(ResidualBlock(channels, out_channels, stride=2))
            channels = out_channels
        self.resnet = nn.Sequential(*blocks)
        self.embed_dim = channels

        # 64×64 を resnet_blocks 回 1/2 にした結果が空間トークン数になる
        grid = max(patch_size // (2 ** resnet_blocks), 1)
        self.num_tokens = grid * grid
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`[B, C_f, H, W]` → ロジット `[B]`。"""
        return self.head(self.embed(features)).squeeze(-1)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        """`[B, C_f, H, W]` → 最後の層（head）の手前の特徴 `[B, D]`。ほかのモデルの特徴とつなぐのに使う。"""
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"入力チャンネル数が一致しません: 期待 {self.in_channels}, 実際 {features.shape[1]}"
            )

        x = self.resnet(features)                       # [B, D, g, g]
        tokens = x.flatten(2).transpose(1, 2)           # [B, g*g, D]
        if tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"トークン数が一致しません: 期待 {self.num_tokens}, 実際 {tokens.shape[1]}。"
                "入力パッチサイズと patch_size 設定を確認してください"
            )

        tokens = tokens + self.position_embedding
        encoded = self.transformer(tokens)
        return self.norm(encoded.mean(dim=1))           # 平均プーリング

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """スコア（0〜1）を返す。推論時のパッチ平均に使う。"""
        return torch.sigmoid(self.forward(features))

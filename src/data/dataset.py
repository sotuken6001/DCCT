"""学習・評価用のDataset（DBのメタデータ → パッチtensor）。

画像本体は外部SSD上に置いたままパス参照で読む（NFR-6）。ZIP内画像も
`open_image_bytes_cached` が透過的に扱う。

前処理のうちCFAマスクとハイパスフィルタは**学習ループ側でバッチ単位・デバイス上**で
実行する。DataLoaderのワーカーはデコードとクロップだけを担当し、畳み込みはMPS/GPUに
任せるほうが速いため。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset, default_collate

from ..experiment.config import Config
from .repository import MetadataRepository
from .sources import open_image_bytes_cached
from ..preprocess.patch_sampler import PatchSampler, load_image_tensor

logger = logging.getLogger(__name__)

# ラベル文字列 → 分類器の教師信号（写真=0 / AI生成=1）
LABEL_TO_TARGET = {"real": 0.0, "ai": 1.0}

# 読み込みに失敗した画像の代わりに別の画像を試す回数の上限
_MAX_FALLBACK = 5

_REAL_GENERATOR_PREFIX = "ImageNet(real)"


def real_generator_names(config: Config) -> list[str]:
    """実写画像側の generator 名を返す（ingest時の命名規則に合わせる）。"""
    naming = config.get("dataset.real_generator_naming", "per_source")
    if naming == "shared":
        return [_REAL_GENERATOR_PREFIX]
    return [f"{_REAL_GENERATOR_PREFIX}@{name}" for name in config.get("dataset.train_generators", [])]


def select_image_rows(
    config: Config,
    repo: MetadataRepository,
    *,
    label: str | None,
    split: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """configの学習対象設定に従って画像行を取り出す。

    - `label='ai'`   → `dataset.train_generators`（論文プロトコルではSDv1.4のみ）
    - `label='real'` → それに対応する実写generator
    - `label=None`   → 両方（Stage II 用）
    """
    train_generators = list(config.get("dataset.train_generators", []))
    real_names = real_generator_names(config)

    if label == "ai":
        generator_names = train_generators
    elif label == "real":
        generator_names = real_names
    else:
        generator_names = train_generators + real_names

    rows = repo.list_images(
        label=label,
        split=split,
        generator_names=generator_names,
        exclude_generator_names=list(config.get("dataset.excluded_generators", [])),
        limit=limit,
    )
    return [dict(row) for row in rows]


class PatchDataset(Dataset):
    """1画像につき1パッチを返す（学習用）。"""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        sampler: PatchSampler,
        ai_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if not rows:
            raise ValueError(
                "対象画像が0件です。ingestが済んでいるか、configの train_generators / "
                "excluded_generators の設定を確認してください"
            )
        self.rows = list(rows)
        self.sampler = sampler
        # AI画像だけに通す処理（実験15：AI画像も JPEG にして本物と形式をそろえる）。パッチを切る前に通す
        self.ai_transform = ai_transform

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, index: int) -> torch.Tensor:
        image = load_image_tensor(open_image_bytes_cached(self.rows[index]["filepath"]))
        if self.ai_transform is not None and self.rows[index]["label"] == "ai":
            image = self.ai_transform(image)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        for attempt in range(_MAX_FALLBACK):
            current = (index + attempt) % len(self.rows)
            try:
                patch = self.sampler.sample_train(self._load(current))
            except Exception as exc:  # 破損・サイズ不足（ingestで拾えなかったもの）
                logger.warning("画像を読めないためスキップします (%s): %s", self.rows[current]["filepath"], exc)
                continue
            target = torch.tensor(LABEL_TO_TARGET[self.rows[current]["label"]])
            return patch, target
        raise RuntimeError(f"連続 {_MAX_FALLBACK} 枚の画像を読めませんでした（index={index}）")


class MultiPatchDataset(Dataset):
    """1画像につきPパッチを返す（推論・評価用、Algorithm 2）。"""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        sampler: PatchSampler,
        num_patches: int = 16,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if not rows:
            raise ValueError("評価対象画像が0件です")
        self.rows = list(rows)
        self.sampler = sampler
        self.num_patches = num_patches
        # ロバスト性評価用の劣化処理（JPEG圧縮・ダウンサンプリング）。パッチ抽出の前に適用する
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """使える画像なら (パッチ, 正解, image_id)、使えなければ None を返す。

        学習用の `PatchDataset` は読めない画像を別の画像で埋めるが、評価では
        それをすると同じ画像を二重に数えてしまうため、代わりに None を返して
        `skip_invalid_collate` に取り除かせ、評価対象から外す。

        ingest はディレクトリ形式のソースでは既定で画像ヘッダを読まないため、
        64px未満の画像が `is_croppable=1` のままDBに残ることがある
        （`ingest --probe` を付ければ登録時に除外できるが、走査が大幅に遅くなる）。
        """
        row = self.rows[index]
        try:
            image = load_image_tensor(open_image_bytes_cached(row["filepath"]))
            if self.transform is not None:
                image = self.transform(image)
            patches = self.sampler.sample_test(image, self.num_patches, image_key=row["filepath"])
        except Exception as exc:  # 破損・サイズ不足（ingestで拾えなかったもの）
            logger.warning("評価対象から除外します (%s): %s", row["filepath"], exc)
            return None
        target = torch.tensor(LABEL_TO_TARGET[row["label"]])
        return patches, target, int(row["image_id"])


def skip_invalid_collate(batch: Sequence[Any]) -> Any | None:
    """`MultiPatchDataset` が返した None を取り除いてからまとめる。

    バッチ内の全件が None の場合はバッチごと None を返す。
    """
    items = [item for item in batch if item is not None]
    if not items:
        return None
    return default_collate(items)

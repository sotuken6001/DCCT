"""実験設定（YAML）の読み込み・検証。

02_基本設計書 8節「設定管理」／01_要件定義書 NFR-5 に対応する。
全てのハイパーパラメータ・パス・アブレーション条件はYAMLに外出しし、
コード変更なしに実験を切り替えられるようにする。

各configは `extends: <相対パス>` で親configを継承できる（辞書は再帰的にマージ、
スカラー・リストは子の値で上書き）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

_MAX_EXTENDS_DEPTH = 10

# src/preprocess/highpass.py が構成するSRMカーネルの枚数（論文のM）
NUM_HIGHPASS_FILTERS = 30


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`base` に `override` を再帰的に重ねた新しい辞書を返す。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw(path: Path, _depth: int = 0) -> dict[str, Any]:
    if _depth > _MAX_EXTENDS_DEPTH:
        raise ValueError(f"config の extends が深すぎます（循環参照の可能性）: {path}")

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config のトップレベルはマッピングである必要があります: {path}")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    parent_path = (path.parent / parent_ref).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"{path} が継承する config が見つかりません: {parent_path}")
    return _deep_merge(_load_raw(parent_path, _depth + 1), data)


class Config:
    """ドット区切りのキーで値を引ける設定オブジェクト。"""

    def __init__(self, data: dict[str, Any], source_path: Path | None = None) -> None:
        self._data = data
        self.source_path = source_path

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"config が見つかりません: {p}")
        cfg = cls(_load_raw(p), source_path=p)
        cfg.validate()
        return cfg

    _MISSING = object()

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """`get("paths.db_path")` のように取得する。既定値なしで欠けていれば KeyError。"""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is Config._MISSING:
                    raise KeyError(f"config に必要なキーがありません: {dotted_key}")
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def snapshot(self) -> str:
        """`experiments.config_snapshot` に保存する文字列表現（JSON）。"""
        return json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True, default=str)

    def validate(self) -> None:
        """必須キーと値域の最低限の検証を行う。"""
        for key in ("paths.db_path", "project.seed"):
            self.get(key)

        device_type = self.get("device.type", "auto")
        if device_type not in ("auto", "cuda", "mps", "cpu"):
            raise ValueError(f"device.type は auto/cuda/mps/cpu のいずれか: {device_type!r}")

        patch_size = self.get("preprocess.patch_size", 64)
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"preprocess.patch_size は正の整数: {patch_size!r}")

        t = self.get("preprocess.truncation_t", 7)
        if not isinstance(t, (int, float)) or t <= 0:
            raise ValueError(f"preprocess.truncation_t は正の数: {t!r}")

        scale = self.get("preprocess.residual_scale", 1)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(f"preprocess.residual_scale は正の数: {scale!r}")

        depth = self.get("model.conditional_unet.depth", 2)
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"model.conditional_unet.depth は1以上の整数: {depth!r}")
        if patch_size % (2**depth) != 0:
            # 縮小のたびに半分にするので、割り切れないと拡大したときに形が合わなくなる
            raise ValueError(
                f"preprocess.patch_size={patch_size} は 2^depth={2**depth} で割り切れる必要があります"
                f"（model.conditional_unet.depth={depth}）"
            )

        feature_source = self.get("model.classifier.feature_source", "mixture_params")
        if feature_source not in ("mixture_params", "bottleneck"):
            raise ValueError(
                f"model.classifier.feature_source は mixture_params/bottleneck のいずれか: {feature_source!r}"
            )

        target_mode = self.get("model.conditional_unet.target_mode", "single_filter")
        if target_mode not in ("single_filter", "filter_bank"):
            raise ValueError(
                f"model.conditional_unet.target_mode は single_filter/filter_bank のいずれか: {target_mode!r}"
            )
        mixture_scope = self.get("model.conditional_unet.mixture_scope", "per_pixel")
        if mixture_scope not in ("per_pixel", "per_channel"):
            raise ValueError(
                f"model.conditional_unet.mixture_scope は per_pixel/per_channel のいずれか: {mixture_scope!r}"
            )
        if (
            target_mode == "filter_bank"
            and mixture_scope == "per_channel"
            and feature_source == "mixture_params"
        ):
            # y'が60chでチャンネルごとにパラメータを持つと 3*K*60 = 1800ch/モデルとなり、
            # 分類器の入力が3600chになって現実的でない
            raise ValueError(
                "target_mode=filter_bank と mixture_scope=per_channel と "
                "feature_source=mixture_params の組み合わせは、分類器入力が3600chになるため"
                "使用できません。mixture_scope=per_pixel（論文準拠）にするか、"
                "feature_source=bottleneck にしてください"
            )

        bayer = self.get("preprocess.bayer_pattern", "RGGB")
        if bayer not in ("RGGB", "BGGR", "GRBG", "GBRG"):
            raise ValueError(f"preprocess.bayer_pattern が不正です: {bayer!r}")

        # ハイパスフィルタの枚数は実装側（SRM 30種）と一致している必要がある
        num_filters = self.get("preprocess.num_highpass_filters", NUM_HIGHPASS_FILTERS)
        if num_filters != NUM_HIGHPASS_FILTERS:
            raise ValueError(
                f"preprocess.num_highpass_filters は {NUM_HIGHPASS_FILTERS} のみ対応しています"
                f"（指定値: {num_filters}）。枚数を変える場合は src/preprocess/highpass.py の"
                "カーネル構成も合わせて変更すること"
            )

        optimizer = str(self.get("train.optimizer", "adam")).lower()
        if optimizer != "adam":
            raise ValueError(
                f"train.optimizer は adam のみ対応しています（指定値: {optimizer!r}）。"
                "論文既定値がAdamのため、他は実装していない"
            )

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"Config(source_path={self.source_path})"

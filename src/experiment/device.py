"""実行デバイスの選択（02_基本設計書 9節 / 03_詳細設計書 7.5）。

`torch.cuda.*` を直接呼ばず、cuda → mps → cpu の順に自動選択する。
macOS実行時はMPS未対応演算をCPUへフォールバックさせる環境変数を設定する。
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def resolve_device(device_type: str = "auto", enable_mps_fallback: bool = True) -> torch.device:
    """configの `device.type` から `torch.device` を決める。"""
    if enable_mps_fallback:
        # MPSで未実装の演算をCPUへ自動フォールバックさせる（macOS以外では無害）
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if device_type == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
        logger.info("デバイスを自動選択しました: %s", resolved)
        return torch.device(resolved)

    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device.type=cuda が指定されましたが CUDA を利用できません")
    if device_type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device.type=mps が指定されましたが MPS を利用できません")
    return torch.device(device_type)


def device_summary(device: torch.device) -> str:
    """ログ・レポート用のデバイス説明文字列。"""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"

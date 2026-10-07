"""実験のトレーサビリティ補助（02_基本設計書 `experiment.tracker` / NFR-7）。"""

from __future__ import annotations

import logging
import os
import random
import subprocess

import numpy as np
import torch

from .paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


def git_commit() -> str | None:
    """実行時のコードバージョン（HEADのコミットハッシュ）を返す。取得できなければNone。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("git commit を取得できませんでした: %s", exc)
        return None

    commit = result.stdout.strip()
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
        if dirty.stdout.strip():
            commit += "-dirty"
    except (subprocess.SubprocessError, OSError):
        pass
    return commit


def set_seed(seed: int) -> None:
    """データ分割・モデル初期化・パッチサンプリングに一貫したseedを適用する（NFR-1）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    logger.info("乱数シードを設定しました: %d", seed)

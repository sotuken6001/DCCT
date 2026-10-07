"""プロジェクト内パスの解決ユーティリティ。

config に書かれた相対パス（例: `db/dcct_metadata.sqlite3`）は、
カレントディレクトリではなくリポジトリルート基準で解決する。
"""

from __future__ import annotations

from pathlib import Path

# src/experiment/paths.py → リポジトリルート
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


def resolve(path: str | Path) -> Path:
    """相対パスをリポジトリルート基準の絶対パスに変換する。絶対パスはそのまま返す。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)

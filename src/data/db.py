"""メタデータDB（SQLite）の接続とマイグレーション適用。

04_DB設計書 4節のDDLは `migrations/0001_init.sql` にそのまま格納してあり、
本モジュールはそれを順に適用するだけのランナーである（同 7節「マイグレーション」対応）。
DBファイル本体を直接手編集しないこと。
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from ..experiment.paths import MIGRATIONS_DIR, resolve

logger = logging.getLogger(__name__)

# 適用済みマイグレーションの記録用テーブル。
# 04_DB設計書のDDLには含まれないが、7節のマイグレーション運用を実現するために
# ランナー側で管理する。
_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# 書き込みロックが取れないときに待つ時間。pθとqφを並列に学習すると、エポックの
# 区切りで両プロセスが同時に指標を書き込むため、既定の5秒では衝突しうる
# （04_DB設計書 7節「同時実行」）。
_BUSY_TIMEOUT_SECONDS = 60.0


def connect(db_path: str | Path) -> sqlite3.Connection:
    """DBへ接続する。外部キー制約を有効化し、行を辞書風に取得できるようにする。"""
    path = resolve(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # 読み書きの並行性と耐障害性のため WAL を使う（04_DB設計書 7節「同時実行」）
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    return conn


def _migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _applied_versions(conn: sqlite3.Connection) -> set[str]:
    conn.executescript(_SCHEMA_MIGRATIONS_DDL)
    return {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}


def backup(db_path: str | Path) -> Path | None:
    """`db/backups/dcct_metadata_YYYYMMDDHHMMSS.sqlite3` にコピーを作る（04_DB設計書 7節）。"""
    path = resolve(db_path)
    if not path.exists():
        return None
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dest = backup_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def init_db(db_path: str | Path, force: bool = False) -> list[str]:
    """未適用のマイグレーションを順に適用し、適用したバージョンのリストを返す。

    `force=True` の場合は既存DBをバックアップしてから削除し、作り直す。
    """
    path = resolve(db_path)

    if force and path.exists():
        saved = backup(path)
        logger.info("既存DBをバックアップしました: %s", saved)
        path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()

    applied: list[str] = []
    conn = connect(path)
    try:
        done = _applied_versions(conn)
        for sql_file in _migration_files():
            version = sql_file.stem
            if version in done:
                continue
            logger.info("マイグレーション適用: %s", sql_file.name)
            with conn:  # 1マイグレーション = 1トランザクション
                conn.executescript(sql_file.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
            applied.append(version)
    finally:
        conn.close()

    return applied


def table_names(db_path: str | Path) -> list[str]:
    """DBに存在するテーブル名（sqlite内部テーブルを除く）を返す。"""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return [row["name"] for row in rows]

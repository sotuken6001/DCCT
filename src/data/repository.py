"""メタデータDBへのアクセス層（03_詳細設計書 3.6 `MetadataRepository` に対応）。

SQLiteの制約上、複数プロセスからの同時書き込みは避け、学習・評価ジョブは
逐次実行する前提とする（04_DB設計書 7節）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import db as db_module

# `images` への一括登録で使う列の並び
_IMAGE_COLUMNS = (
    "dataset_id",
    "generator_id",
    "filepath",
    "label",
    "split",
    "width",
    "height",
    "filesize_bytes",
    "checksum",
    "status",
    "is_croppable",
)


class MetadataRepository:
    """メタデータDBのCRUDをまとめたリポジトリ。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection = db_module.connect(db_path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MetadataRepository":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # マスタ（datasets / generators）
    # ------------------------------------------------------------------

    def get_or_create_dataset(self, name: str, root_path: str, description: str | None = None) -> int:
        row = self.conn.execute("SELECT dataset_id FROM datasets WHERE name = ?", (name,)).fetchone()
        if row is not None:
            # root_path は環境によって変わりうるので最新の値へ更新する
            with self.conn:
                self.conn.execute(
                    "UPDATE datasets SET root_path = ? WHERE dataset_id = ?", (root_path, row["dataset_id"])
                )
            return int(row["dataset_id"])

        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO datasets(name, description, root_path) VALUES (?, ?, ?)",
                (name, description, root_path),
            )
        return int(cur.lastrowid)

    def get_or_create_generator(
        self,
        dataset_id: int,
        name: str,
        category: str,
        is_real: bool,
        notes: str | None = None,
    ) -> int:
        row = self.conn.execute(
            "SELECT generator_id FROM generators WHERE dataset_id = ? AND name = ?", (dataset_id, name)
        ).fetchone()
        if row is not None:
            if notes is not None:
                with self.conn:
                    self.conn.execute(
                        "UPDATE generators SET notes = ? WHERE generator_id = ?", (notes, row["generator_id"])
                    )
            return int(row["generator_id"])

        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO generators(dataset_id, name, category, is_real, notes) VALUES (?, ?, ?, ?, ?)",
                (dataset_id, name, category, int(is_real), notes),
            )
        return int(cur.lastrowid)

    def generator_id_by_name(self, dataset_id: int, name: str) -> int | None:
        row = self.conn.execute(
            "SELECT generator_id FROM generators WHERE dataset_id = ? AND name = ?", (dataset_id, name)
        ).fetchone()
        return int(row["generator_id"]) if row else None

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------

    def register_image(self, **fields: Any) -> int:
        """1枚登録する。既に同じ filepath があれば何もしない（戻り値0）。"""
        return self.register_images([fields])

    def register_images(self, rows: Iterable[dict[str, Any]]) -> int:
        """`images` へ一括登録する。`filepath` が既登録の行は無視する（ingestの再実行安全性）。

        戻り値は実際に挿入された行数。
        """
        values: list[tuple[Any, ...]] = [
            tuple(row.get(column) for column in _IMAGE_COLUMNS) for row in rows
        ]
        if not values:
            return 0

        placeholders = ", ".join("?" for _ in _IMAGE_COLUMNS)
        sql = (
            f"INSERT OR IGNORE INTO images({', '.join(_IMAGE_COLUMNS)}) VALUES ({placeholders})"
        )
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(sql, values)
        return self.conn.total_changes - before

    def list_images(
        self,
        *,
        label: str | None = None,
        split: str | None = None,
        generator_names: Sequence[str] | None = None,
        exclude_generator_names: Sequence[str] | None = None,
        only_usable: bool = True,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        """学習・評価対象の画像を取得する（04_DB設計書 6節のクエリ例に対応）。

        `only_usable=True` のとき、破損画像（status='invalid'）と
        64×64クロップ不可の画像（is_croppable=0）を除外する（03_詳細設計書 6節）。
        """
        sql = [
            "SELECT i.image_id, i.filepath, i.label, i.split, i.width, i.height, g.name AS generator",
            "FROM images i JOIN generators g ON g.generator_id = i.generator_id",
            "WHERE 1=1",
        ]
        params: list[Any] = []
        if label is not None:
            sql.append("AND i.label = ?")
            params.append(label)
        if split is not None:
            sql.append("AND i.split = ?")
            params.append(split)
        if only_usable:
            sql.append("AND i.status = 'valid' AND i.is_croppable = 1")
        if generator_names:
            sql.append(f"AND g.name IN ({', '.join('?' for _ in generator_names)})")
            params.extend(generator_names)
        if exclude_generator_names:
            sql.append(f"AND g.name NOT IN ({', '.join('?' for _ in exclude_generator_names)})")
            params.extend(exclude_generator_names)
        sql.append("ORDER BY i.image_id")
        if limit is not None:
            sql.append("LIMIT ?")
            params.append(limit)

        return self.conn.execute(" ".join(sql), params).fetchall()

    def image_summary(self) -> list[sqlite3.Row]:
        """生成器 × split × label ごとの枚数と、除外対象の枚数を集計する。"""
        return self.conn.execute(
            """
            SELECT g.name AS generator, i.split, i.label,
                   COUNT(*) AS num_images,
                   SUM(CASE WHEN i.status = 'invalid' THEN 1 ELSE 0 END) AS num_invalid,
                   SUM(CASE WHEN i.is_croppable = 0 THEN 1 ELSE 0 END) AS num_too_small
            FROM images i JOIN generators g ON g.generator_id = i.generator_id
            GROUP BY g.name, i.split, i.label
            ORDER BY g.name, i.split, i.label
            """
        ).fetchall()

    # ------------------------------------------------------------------
    # experiments / training_runs / logs / checkpoints / results
    # ------------------------------------------------------------------

    def register_experiment(
        self,
        name: str,
        stage: str,
        config_snapshot: str,
        git_commit: str | None = None,
        seed: int | None = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO experiments(name, stage, config_snapshot, git_commit, seed) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, stage, config_snapshot, git_commit, seed),
            )
        return int(cur.lastrowid)

    def create_training_run(self, experiment_id: int) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO training_runs(experiment_id) VALUES (?)", (experiment_id,)
            )
        return int(cur.lastrowid)

    def finish_training_run(self, run_id: int, status: str = "completed") -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE training_runs SET status = ?, completed_at = datetime('now') WHERE run_id = ?",
                (status, run_id),
            )

    def log_training_metric(
        self, run_id: int, epoch: int, metric_name: str, metric_value: float, step: int | None = None
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO training_logs(run_id, epoch, step, metric_name, metric_value) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, epoch, step, metric_name, float(metric_value)),
            )

    def save_checkpoint_meta(
        self,
        run_id: int,
        epoch: int,
        filepath: str,
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO model_checkpoints(run_id, epoch, filepath, metric_name, metric_value) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, epoch, filepath, metric_name, metric_value),
            )
        return int(cur.lastrowid)

    def save_evaluation_result(
        self,
        run_id: int,
        generator_id: int,
        eval_type: str,
        split: str,
        *,
        accuracy: float | None = None,
        auc: float | None = None,
        ap: float | None = None,
        num_samples: int | None = None,
        perturbation_type: str = "none",
        perturbation_level: float | None = None,
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO evaluation_results("
                "run_id, generator_id, eval_type, split, perturbation_type, perturbation_level, "
                "accuracy, auc, ap, num_samples) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    generator_id,
                    eval_type,
                    split,
                    perturbation_type,
                    perturbation_level,
                    accuracy,
                    auc,
                    ap,
                    num_samples,
                ),
            )
        return int(cur.lastrowid)

    def save_ablation_config(
        self, experiment_id: int, ablation_group: str, parameter_name: str, parameter_value: str
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO ablation_configs(experiment_id, ablation_group, parameter_name, parameter_value) "
                "VALUES (?, ?, ?, ?)",
                (experiment_id, ablation_group, parameter_name, str(parameter_value)),
            )
        return int(cur.lastrowid)

"""04_DB設計書のスキーマが意図どおり作られているかの検証。"""

from __future__ import annotations

import sqlite3

import pytest

from src.data import db as db_module

EXPECTED_TABLES = {
    "datasets",
    "generators",
    "images",
    "experiments",
    "training_runs",
    "training_logs",
    "model_checkpoints",
    "evaluation_results",
    "ablation_configs",
}

EXPECTED_INDEXES = {
    "idx_images_generator",
    "idx_images_split_label",
    "idx_training_logs_run",
    "idx_checkpoints_run",
    "idx_evalresults_run",
    "idx_evalresults_gen_type",
    "idx_ablation_experiment",
}


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "dcct_metadata.sqlite3"
    db_module.init_db(path)
    return path


def test_all_tables_created(db_path):
    assert EXPECTED_TABLES <= set(db_module.table_names(db_path))


def test_all_indexes_created(db_path):
    conn = db_module.connect(db_path)
    names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    conn.close()
    assert EXPECTED_INDEXES <= names


def test_init_db_is_idempotent(db_path):
    assert db_module.init_db(db_path) == []


def test_foreign_keys_are_enforced(db_path):
    conn = db_module.connect(db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO generators(dataset_id, name, category, is_real) VALUES (999, 'X', 'gan', 0)"
        )
    conn.close()


def test_check_constraints(db_path):
    conn = db_module.connect(db_path)
    conn.execute("INSERT INTO datasets(name, root_path) VALUES ('GenImage', '/Volumes/Extreme Pro')")
    conn.execute(
        "INSERT INTO generators(dataset_id, name, category, is_real) VALUES (1, 'SDv1.4', 'diffusion', 0)"
    )

    # category は real/gan/diffusion のみ
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO generators(dataset_id, name, category, is_real) VALUES (1, 'Bad', 'unknown', 0)"
        )

    # images.label は real/ai、split は train/val/test のみ
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO images(dataset_id, generator_id, filepath, label, split) "
            "VALUES (1, 1, '/a.png', 'fake', 'train')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO images(dataset_id, generator_id, filepath, label, split) "
            "VALUES (1, 1, '/b.png', 'ai', 'holdout')"
        )

    # 既定値: status='valid', is_croppable=1
    conn.execute(
        "INSERT INTO images(dataset_id, generator_id, filepath, label, split) "
        "VALUES (1, 1, '/c.png', 'ai', 'train')"
    )
    row = conn.execute("SELECT status, is_croppable FROM images WHERE filepath='/c.png'").fetchone()
    assert row["status"] == "valid"
    assert row["is_croppable"] == 1

    # filepath は UNIQUE（ingest の再実行で重複登録されないこと）
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO images(dataset_id, generator_id, filepath, label, split) "
            "VALUES (1, 1, '/c.png', 'ai', 'train')"
        )
    conn.close()


def test_force_recreates_and_backs_up(tmp_path):
    path = tmp_path / "dcct_metadata.sqlite3"
    db_module.init_db(path)

    conn = db_module.connect(path)
    conn.execute("INSERT INTO datasets(name, root_path) VALUES ('GenImage', '/tmp')")
    conn.commit()
    conn.close()

    db_module.init_db(path, force=True)

    conn = db_module.connect(path)
    assert conn.execute("SELECT COUNT(*) AS n FROM datasets").fetchone()["n"] == 0
    conn.close()
    assert list((tmp_path / "backups").glob("dcct_metadata_*.sqlite3"))

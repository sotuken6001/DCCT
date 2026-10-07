-- 04_DB設計書 4節「DDL（SQLite）」および 5節「インデックス設計」をそのまま適用する。
-- スキーマ変更は本ファイルを編集せず、新しいマイグレーションを追加すること（04_DB設計書 7節）。

PRAGMA foreign_keys = ON;

CREATE TABLE datasets (
    dataset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT,
    root_path    TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE generators (
    generator_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id   INTEGER NOT NULL REFERENCES datasets(dataset_id),
    name         TEXT NOT NULL,
    category     TEXT NOT NULL CHECK (category IN ('real','gan','diffusion')),
    is_real      INTEGER NOT NULL CHECK (is_real IN (0,1)),
    notes        TEXT,
    UNIQUE(dataset_id, name)
);

CREATE TABLE images (
    image_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id     INTEGER NOT NULL REFERENCES datasets(dataset_id),
    generator_id   INTEGER NOT NULL REFERENCES generators(generator_id),
    filepath       TEXT NOT NULL UNIQUE,
    label          TEXT NOT NULL CHECK (label IN ('real','ai')),
    split          TEXT NOT NULL CHECK (split IN ('train','val','test')),
    width          INTEGER,
    height         INTEGER,
    filesize_bytes INTEGER,
    checksum       TEXT,
    status         TEXT NOT NULL DEFAULT 'valid' CHECK (status IN ('valid','invalid')),
    is_croppable   INTEGER NOT NULL DEFAULT 1 CHECK (is_croppable IN (0,1)),
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE experiments (
    experiment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    stage           TEXT NOT NULL CHECK (stage IN ('stage1_photo','stage1_ai','stage2_classifier','ablation','robustness')),
    config_snapshot TEXT NOT NULL,
    git_commit      TEXT,
    seed            INTEGER,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE training_runs (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(experiment_id),
    status         TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','failed')),
    started_at     TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at   TEXT
);

CREATE TABLE training_logs (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES training_runs(run_id),
    epoch        INTEGER NOT NULL,
    step         INTEGER,
    metric_name  TEXT NOT NULL,
    metric_value REAL NOT NULL,
    logged_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE model_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES training_runs(run_id),
    epoch         INTEGER NOT NULL,
    filepath      TEXT NOT NULL,
    metric_name   TEXT,
    metric_value  REAL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE evaluation_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES training_runs(run_id),
    generator_id          INTEGER NOT NULL REFERENCES generators(generator_id),
    eval_type            TEXT NOT NULL CHECK (eval_type IN ('cross_generator','ablation','robustness')),
    split                TEXT NOT NULL,
    perturbation_type    TEXT DEFAULT 'none',
    perturbation_level   REAL,
    accuracy             REAL,
    auc                  REAL,
    ap                   REAL,
    num_samples          INTEGER,
    evaluated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE ablation_configs (
    ablation_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(experiment_id),
    ablation_group  TEXT NOT NULL CHECK (ablation_group IN ('A','B','C','D')),
    parameter_name  TEXT NOT NULL,
    parameter_value TEXT NOT NULL
);

-- 5. インデックス設計
CREATE INDEX idx_images_generator     ON images(generator_id);
CREATE INDEX idx_images_split_label   ON images(split, label);
CREATE INDEX idx_training_logs_run    ON training_logs(run_id, epoch);
CREATE INDEX idx_checkpoints_run      ON model_checkpoints(run_id);
CREATE INDEX idx_evalresults_run      ON evaluation_results(run_id);
CREATE INDEX idx_evalresults_gen_type ON evaluation_results(generator_id, eval_type);
CREATE INDEX idx_ablation_experiment  ON ablation_configs(experiment_id);

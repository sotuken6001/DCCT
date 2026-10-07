"""実験config読み込み（extends継承・検証）のテスト。"""

from __future__ import annotations

import pytest

from src.experiment.config import Config
from src.experiment.paths import PROJECT_ROOT

CONFIG_FILES = sorted((PROJECT_ROOT / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_all_shipped_configs_load(path):
    """configs/ 配下の全YAMLが読み込め、検証を通ること。"""
    cfg = Config.load(path)
    assert cfg.get("paths.db_path")


def test_extends_merges_recursively(tmp_path):
    (tmp_path / "parent.yaml").write_text(
        "paths:\n  db_path: db/a.sqlite3\n  log_root: logs\n"
        "project:\n  seed: 42\n"
        "train:\n  batch_size: 16\n  learning_rate: 1.0e-4\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "extends: parent.yaml\ntrain:\n  batch_size: 8\n", encoding="utf-8"
    )

    cfg = Config.load(tmp_path / "child.yaml")
    assert cfg.get("train.batch_size") == 8          # 子で上書き
    assert cfg.get("train.learning_rate") == 1.0e-4  # 親から継承
    assert cfg.get("paths.log_root") == "logs"


def test_missing_key_raises():
    cfg = Config.load(PROJECT_ROOT / "configs" / "base.yaml")
    with pytest.raises(KeyError):
        cfg.get("no.such.key")
    assert cfg.get("no.such.key", "fallback") == "fallback"


def test_invalid_device_type_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "paths:\n  db_path: db/a.sqlite3\nproject:\n  seed: 1\ndevice:\n  type: tpu\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="device.type"):
        Config.load(tmp_path / "bad.yaml")


def test_midjourney_excluded_by_default():
    """【OI-1】既定では Midjourney が除外対象に入っていること。"""
    cfg = Config.load(PROJECT_ROOT / "configs" / "base.yaml")
    assert "Midjourney" in cfg.get("dataset.excluded_generators")


def test_classifier_feature_source_is_mixture_params():
    """【OI-2】既定の分類器入力特徴は案A（混合分布パラメータ）であること。"""
    cfg = Config.load(PROJECT_ROOT / "configs" / "base.yaml")
    assert cfg.get("model.classifier.feature_source") == "mixture_params"


def test_unsupported_optimizer_rejected(tmp_path):
    """configにあるのに実装が無い、という乖離を検出できること。"""
    (tmp_path / "opt.yaml").write_text(
        "paths:\n  db_path: db/a.sqlite3\nproject:\n  seed: 1\ntrain:\n  optimizer: sgd\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train.optimizer"):
        Config.load(tmp_path / "opt.yaml")


def test_highpass_filter_count_must_match_implementation(tmp_path):
    (tmp_path / "m.yaml").write_text(
        "paths:\n  db_path: db/a.sqlite3\nproject:\n  seed: 1\n"
        "preprocess:\n  num_highpass_filters: 12\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_highpass_filters"):
        Config.load(tmp_path / "m.yaml")


def test_shipped_config_filter_count_matches_built_kernels():
    """configのMと、実際に構成されるカーネル枚数が一致していること。"""
    from src.preprocess.highpass import build_srm_kernels

    kernels, _ = build_srm_kernels()
    config = Config.load(PROJECT_ROOT / "configs" / "base.yaml")
    assert config.get("preprocess.num_highpass_filters") == kernels.shape[0]


def test_stage2_config_points_at_stage1_default_checkpoints():
    """Stage IIのconfigが、Stage Iの既定の保存先をそのまま指していること。"""
    stage2 = Config.load(PROJECT_ROOT / "configs" / "stage2_classifier.yaml")
    photo = Config.load(PROJECT_ROOT / "configs" / "stage1_photo.yaml")
    ai = Config.load(PROJECT_ROOT / "configs" / "stage1_ai.yaml")

    assert stage2.get("stage2.photo_model_checkpoint") == f"{photo.get('train.checkpoint_dir')}/best.pt"
    assert stage2.get("stage2.ai_model_checkpoint") == f"{ai.get('train.checkpoint_dir')}/best.pt"


def test_cli_logs_traceback_on_failure(tmp_path, monkeypatch, caplog):
    """バックグラウンド実行で標準エラーを捨てても、失敗理由がログに残る。"""
    import logging as _logging

    from src import cli

    def _boom(args):
        raise RuntimeError("想定外のエラー")

    monkeypatch.setattr(cli, "_setup_logging", lambda args: None)
    monkeypatch.setattr(cli, "cmd_status", _boom)

    with caplog.at_level(_logging.ERROR):
        code = cli.main(["status", "--config", "configs/base.yaml"])

    assert code == 1
    assert "異常終了しました" in caplog.text
    assert "想定外のエラー" in caplog.text     # トレースバック本文も残る

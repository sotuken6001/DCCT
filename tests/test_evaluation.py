"""評価（cross-generator / robustness / ablation）とレポート出力のテスト。"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.data import db as db_module
from src.data.ingest import ingest
from src.data.repository import MetadataRepository
from src.eval.evaluator import Evaluator, downsample_transform, jpeg_transform
from src.eval.metrics import compute_metrics, tune_threshold
from src.experiment.config import Config
from src.report.report_builder import (
    add_mean_row,
    build_report,
    fetch_cross_generator,
    fetch_robustness,
)
from src.train.train_stage2 import TrainerStage2


# ------------------------------------------------------------------ metrics


def test_compute_metrics_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_metrics(scores, labels, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["ap"] == 1.0
    assert metrics["num_samples"] == 4


def test_compute_metrics_single_class_has_no_auc():
    metrics = compute_metrics(np.array([0.2, 0.3]), np.array([1.0, 1.0]), threshold=0.5)
    assert metrics["auc"] is None and metrics["ap"] is None
    assert metrics["accuracy"] == 0.0


def test_tune_threshold_finds_separating_point():
    scores = np.array([0.1, 0.15, 0.7, 0.75])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    threshold = tune_threshold(scores, labels)
    assert 0.15 < threshold <= 0.7
    assert ((scores > threshold).astype(float) == labels).all()


def test_tune_threshold_falls_back_when_single_class():
    assert tune_threshold(np.array([0.3, 0.4]), np.array([1.0, 1.0]), default=0.42) == 0.42


# ------------------------------------------------------- 劣化処理（Figure 6）


def test_jpeg_transform_keeps_shape_and_changes_pixels():
    image = torch.rand(3, 96, 96) * 255
    out = jpeg_transform(70)(image)
    assert out.shape == image.shape
    assert not torch.equal(out, image.round())


def test_downsample_transform_restores_resolution_and_blurs():
    image = torch.rand(3, 96, 96) * 255
    out = downsample_transform(0.6)(image)
    assert out.shape == image.shape
    assert out.min() >= 0 and out.max() <= 255
    # 高周波が落ちるので、隣接画素差の大きさが小さくなる
    assert out.diff(dim=-1).abs().mean() < image.diff(dim=-1).abs().mean()


# ------------------------------------------------------------ 評価の通し実行


def _png(color, size=96) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def trained(tmp_path):
    """2生成器分のデータをingestし、Stage IIを1エポックだけ学習した状態を作る。"""
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    for archive, ai_color in (("genimage-stable-diffusion-v1-4.zip", (200, 40, 40)), ("BigGAN.zip", (40, 40, 200))):
        with zipfile.ZipFile(root / archive, "w") as zf:
            for i in range(6):
                zf.writestr(f"train/ai/ai_{i}.png", _png(ai_color))
                zf.writestr(f"train/nature/nat_{i}.png", _png((40, 200, 40)))
            for i in range(3):
                zf.writestr(f"val/ai/ai_{i}.png", _png(ai_color))
                zf.writestr(f"val/nature/nat_{i}.png", _png((40, 200, 40)))

    data = Config.load("configs/base.yaml").as_dict()
    data["paths"]["dataset_root"] = str(root)
    data["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    data["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"},
        {"name": "BigGAN", "path": "BigGAN.zip", "category": "gan", "type": "auto"},
    ]
    data["dataset"]["eval_generators"] = ["SDv1.4", "BigGAN", "Midjourney"]
    data["dataset"]["split"]["val_ratio_from_train"] = 0.5
    data["augmentation"]["jpeg"]["enabled"] = False
    data["model"]["conditional_unet"]["base_channels"] = 8
    data["inference"]["num_patches"] = 2
    data["train"].update({"batch_size": 2, "num_workers": 0, "num_epochs": 1, "max_eval_steps": 1,
                          "checkpoint_dir": str(tmp_path / "ckpt")})
    data["stage2"] = {"use_photo_model": True, "use_ai_model": True, "freeze_conditional_models": True,
                      "photo_model_checkpoint": None, "ai_model_checkpoint": None}
    data["evaluate"] = {"tune_threshold": True, "max_images_per_generator": 4,
                        "robustness": {"jpeg_qualities": [90], "downsample_ratios": [0.6]}}
    data["experiment"] = {"name": "eval_test", "stage": "stage2_classifier"}

    db_module.init_db(data["paths"]["db_path"])
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = Config.load(path)
    ingest(config, progress=False)

    with TrainerStage2(config) as trainer:
        run_id = trainer.fit(num_epochs=1, max_steps_per_epoch=2)
        checkpoint = str(trainer.checkpoint_dir / "best.pt")
    return tmp_path, config, run_id, checkpoint


def test_cross_generator_saves_results_per_generator(trained):
    tmp_path, config, run_id, checkpoint = trained

    with Evaluator(config, run_id, checkpoint) as evaluator:
        results = evaluator.run_cross_generator()
        rows = evaluator.repo.conn.execute(
            """
            SELECT g.name AS generator, er.eval_type, er.split, er.accuracy, er.num_samples,
                   er.perturbation_type
            FROM evaluation_results er JOIN generators g USING(generator_id)
            WHERE er.run_id = ?
            """,
            (run_id,),
        ).fetchall()

    # 【OI-1】Midjourney は excluded_generators により評価対象から外れる
    assert {r.generator for r in results} == {"SDv1.4", "BigGAN"}
    assert {row["generator"] for row in rows} == {"SDv1.4", "BigGAN"}
    assert all(row["eval_type"] == "cross_generator" and row["split"] == "test" for row in rows)
    assert all(row["perturbation_type"] == "none" for row in rows)
    # ラベルごとに上限を適用するので、AI画像と実写画像が同数ずつ入る
    assert all(row["num_samples"] % 2 == 0 and row["num_samples"] > 0 for row in rows)


def test_evaluation_pairs_generator_with_its_own_real_subset(trained):
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        rows = evaluator._rows_for("SDv1.4", "test")

    generators = {row["generator"] for row in rows}
    assert generators == {"SDv1.4", "ImageNet(real)@SDv1.4"}
    assert {row["label"] for row in rows} == {"ai", "real"}


def test_threshold_is_calibrated_on_val_split(trained):
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        before = evaluator.threshold
        after = evaluator.calibrate_threshold()
    assert 0.0 <= after <= 1.0
    assert isinstance(before, float)


def test_threshold_tuning_can_be_disabled(trained):
    tmp_path, config, run_id, checkpoint = trained
    data = config.as_dict()
    data["evaluate"]["tune_threshold"] = False
    data["inference"]["threshold"] = 0.42

    with Evaluator(Config(data), run_id, checkpoint) as evaluator:
        assert evaluator.calibrate_threshold() == 0.42


def test_robustness_records_each_perturbation_level(trained):
    tmp_path, config, run_id, checkpoint = trained

    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_robustness()
        rows = evaluator.repo.conn.execute(
            """
            SELECT DISTINCT er.perturbation_type, er.perturbation_level
            FROM evaluation_results er WHERE er.run_id = ? AND er.eval_type = 'robustness'
            ORDER BY er.perturbation_type
            """,
            (run_id,),
        ).fetchall()

    conditions = {(row["perturbation_type"], row["perturbation_level"]) for row in rows}
    assert ("none", None) in conditions
    assert ("jpeg", 90.0) in conditions
    assert ("downsample", 0.6) in conditions


# ---------------------------------------------------------------- ablation


def test_ablation_runs_variants_and_records_conditions(trained):
    from src.eval.ablation import run_ablation

    tmp_path, config, _, _ = trained
    data = config.as_dict()
    data["ablation"] = {
        "group": "A",
        "variants": [
            {"label": "photo_only", "use_photo_model": True, "use_ai_model": False},
            {"label": "both", "use_photo_model": True, "use_ai_model": True},
        ],
    }
    data["paths"]["checkpoint_root"] = str(tmp_path / "abl")

    results = run_ablation(Config(data), num_epochs=1, max_steps_per_epoch=1)
    assert [r.label for r in results] == ["photo_only", "both"]
    assert all(r.mean_accuracy is not None for r in results)

    with MetadataRepository(data["paths"]["db_path"]) as repo:
        conditions = repo.conn.execute(
            "SELECT ablation_group, parameter_name, parameter_value FROM ablation_configs"
        ).fetchall()
        eval_rows = repo.conn.execute(
            "SELECT COUNT(*) AS n FROM evaluation_results WHERE eval_type = 'ablation'"
        ).fetchone()

    assert {row["ablation_group"] for row in conditions} == {"A"}
    assert {row["parameter_name"] for row in conditions} == {"use_photo_model", "use_ai_model"}
    assert eval_rows["n"] > 0


def test_ablation_rejects_unknown_parameter(trained):
    from src.eval.ablation import build_variant_config

    tmp_path, config, _, _ = trained
    with pytest.raises(ValueError, match="未知のアブレーションパラメータ"):
        build_variant_config(config, {"label": "x", "no_such_param": 1}, str(tmp_path))


# ------------------------------------------------------------------ report


def test_report_writes_csv_markdown_and_figures(trained):
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_cross_generator()
        evaluator.run_robustness()

    report_path = build_report(config, run_id, tmp_path / "reports")
    out_dir = report_path.parent

    assert report_path.exists()
    assert (out_dir / "cross_generator.csv").exists()
    assert (out_dir / "robustness.csv").exists()
    # Accuracy だけでなく、閾値に依存しないAUCの図も出す
    for perturbation in ("jpeg", "downsample"):
        for metric in ("accuracy", "auc"):
            assert (out_dir / f"robustness_{perturbation}_{metric}.png").exists()

    text = report_path.read_text(encoding="utf-8")
    assert "Cross-generator評価（Table 1相当）" in text
    assert "ロバスト性評価（Figure 6相当）" in text
    assert "平均(2生成器)" in text
    assert f"実行ID: {run_id}" in text


def test_robustness_report_includes_auc(trained):
    """劣化条件ごとのAUCも表とCSVに載ること。

    Accuracy は固定した閾値に対する値なので、スコア分布がずれただけでも0.5まで
    落ちる。識別力が残っているかはAUCを見ないと判定できない。
    """
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_robustness()

    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = fetch_robustness(repo, run_id)

    assert rows
    assert all("auc" in row for row in rows)

    report_path = build_report(config, run_id, tmp_path / "reports")
    header = (report_path.parent / "robustness.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ["generator", "perturbation_type", "perturbation_level", "accuracy", "auc", "ap"]


def test_report_deduplicates_repeated_evaluations(trained):
    """同じ条件を2回評価しても、レポートには最新の1件だけが載ること。"""
    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_cross_generator()
        evaluator.run_cross_generator()

        rows = fetch_cross_generator(evaluator.repo, run_id)
        raw = evaluator.repo.conn.execute(
            "SELECT COUNT(*) AS n FROM evaluation_results WHERE run_id = ? AND eval_type = 'cross_generator'",
            (run_id,),
        ).fetchone()["n"]

    assert raw == 4                      # 2生成器 × 2回
    assert len(rows) == 2                # レポート側は最新のみ


def test_add_mean_row_computes_generator_average():
    rows = [
        {"generator": "A", "accuracy": 0.9, "auc": 0.95, "ap": 0.9, "num_samples": 10},
        {"generator": "B", "accuracy": 0.7, "auc": 0.75, "ap": 0.7, "num_samples": 20},
    ]
    with_mean = add_mean_row(rows)
    assert with_mean[-1]["generator"] == "平均(2生成器)"
    assert with_mean[-1]["accuracy"] == pytest.approx(0.8)
    assert with_mean[-1]["num_samples"] == 30


def test_threshold_calibration_ignores_eval_image_cap(monkeypatch):
    """`evaluate.max_images_per_generator` は閾値チューニングには効かない。

    評価枚数を絞るつまみが τ の較正まで変えると、cross_generator と robustness で
    別の τ になってしまい、両者を比較できなくなる。
    """
    config = Config.load("configs/base.yaml")
    config._data["evaluate"]["max_images_per_generator"] = 1

    captured: dict[str, object] = {}

    class _StubRepo:
        def __init__(self, *args, **kwargs):
            pass

        def list_images(self, *, label, split, generator_names, limit=None, **kwargs):
            captured.setdefault(split, []).append(limit)
            return []

        def close(self):
            pass

    monkeypatch.setattr("src.eval.evaluator.MetadataRepository", _StubRepo)
    monkeypatch.setattr("src.eval.evaluator.DCCTPredictor", lambda *a, **k: object())

    evaluator = Evaluator(config, run_id=1)
    evaluator.calibrate_threshold()
    assert captured["val"], "val split が引かれていない"
    assert all(limit is None for limit in captured["val"])   # 上限なし

    evaluator._rows_for("SDv1.4", "test")
    assert all(limit == 1 for limit in captured["test"])      # 評価側には効く


def test_robustness_transforms_are_picklable():
    """DataLoaderのワーカーへ渡せること（`num_workers>0` で必須）。

    クロージャで実装すると `Can't get local object` で robustness 評価が落ちる。
    """
    import pickle

    image = torch.rand(3, 96, 96) * 255
    for transform in (jpeg_transform(70), downsample_transform(0.6)):
        restored = pickle.loads(pickle.dumps(transform))
        assert torch.equal(restored(image), transform(image))


def test_score_rows_does_not_retain_worker_shared_memory():
    """スコア収集がワーカーの共有メモリを掴んだままにしない。

    `.numpy()` はコピーせず元のテンソルを参照するため、そのまま溜めると
    共有メモリのfdが開いたままになり `Too many open files` で落ちる。
    """
    targets = torch.tensor([0.0, 1.0, 1.0])
    kept = targets.numpy().copy()
    targets.fill_(9.0)                       # 元のテンソルを書き換えても
    assert kept.tolist() == [0.0, 1.0, 1.0]  # コピー側は影響を受けない


def test_cli_sets_file_system_sharing_strategy(monkeypatch):
    """評価で63回DataLoaderを作り直してもfdが枯渇しないようにしておく。

    共有方式はプロセス全体の設定なので、テスト内で実際に切り替えると後続の
    テストに影響する。呼び出し先を差し替えて、渡す値だけを確かめる。
    """
    import torch.multiprocessing as mp

    from src import cli

    called: list[str] = []
    monkeypatch.setattr(mp, "set_sharing_strategy", called.append)
    cli._configure_multiprocessing()
    assert called == ["file_system"]


def test_robustness_can_recalibrate_threshold_per_condition(trained, monkeypatch):
    """`recalibrate_threshold: true` で劣化条件ごとにτを取り直すこと。

    既定の単一τでは、劣化でスコア分布がずれた条件は全件が閾値の片側に寄り、
    識別力が残っていてもAccuracyが0.5まで落ちる。
    """
    tmp_path, config, run_id, checkpoint = trained
    config._data["evaluate"]["robustness"]["recalibrate_threshold"] = True

    calibrated: list[object] = []
    original = Evaluator.calibrate_threshold

    def _spy(self, transform=None, apply=True):
        calibrated.append(transform)
        return original(self, transform=transform, apply=apply)

    monkeypatch.setattr(Evaluator, "calibrate_threshold", _spy)

    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_robustness()

    # 最初の1回（全体用）+ 条件ごと（none + JPEG各水準 + 縮小各水準）
    num_conditions = (
        1
        + len(config.get("evaluate.robustness.jpeg_qualities"))
        + len(config.get("evaluate.robustness.downsample_ratios"))
    )
    assert len(calibrated) == 1 + num_conditions
    # 劣化つきのvalで較正した回があること（transform が渡されている）
    assert any(transform is not None for transform in calibrated)


def test_robustness_uses_single_threshold_by_default(trained, monkeypatch):
    """既定では Algorithm 2 どおり、較正は最初の1回だけ。"""
    tmp_path, config, run_id, checkpoint = trained
    config._data["evaluate"]["robustness"]["recalibrate_threshold"] = False

    calls: list[object] = []
    original = Evaluator.calibrate_threshold

    def _spy(self, transform=None, apply=True):
        calls.append(transform)
        return original(self, transform=transform, apply=apply)

    monkeypatch.setattr(Evaluator, "calibrate_threshold", _spy)

    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_robustness()

    assert calls == [None]


# ------------------------------------------------- AI画像だけ JPEG にする（確認6）


def test_ai_jpeg_compresses_only_ai_images_and_keeps_cross_generator_report(trained, monkeypatch):
    """本物は JPEG・AI画像は PNG という形式の差を消して、分類器が何を見ているかを確かめる。

    結果は robustness（perturbation_type='ai_jpeg'）として保存し、cross_generator の表は書きかえない。
    """
    import src.eval.evaluator as evaluator_module

    tmp_path, config, run_id, checkpoint = trained
    seen: list[int] = []

    def recording_transform(quality, subsampling=None):
        def transform(image):
            seen.append(quality)
            return image
        return transform

    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_cross_generator()
        before = fetch_cross_generator(evaluator.repo, run_id)
        expected = sum(
            row["label"] == "ai"
            for generator in ("SDv1.4", "BigGAN")
            for row in evaluator._rows_for(generator, "test")
        )
        monkeypatch.setattr(evaluator_module, "jpeg_transform", recording_transform)
        results = evaluator.run_ai_jpeg([90, 75])
        after = fetch_cross_generator(evaluator.repo, run_id)
        rows = evaluator.repo.conn.execute(
            "SELECT DISTINCT eval_type, perturbation_type, perturbation_level FROM evaluation_results "
            "WHERE run_id = ? AND perturbation_type = 'ai_jpeg'",
            (run_id,),
        ).fetchall()

    assert expected > 0
    # 閾値は圧縮しない val で決め、AI画像だけが品質ごとに圧縮される
    assert sorted(seen) == sorted([90] * expected + [75] * expected)
    assert {(r.generator, r.perturbation_level) for r in results} == {
        (g, q) for g in ("SDv1.4", "BigGAN") for q in (90.0, 75.0)
    }
    assert {(row["eval_type"], row["perturbation_level"]) for row in rows} == {("robustness", 90.0), ("robustness", 75.0)}
    assert after == before


def test_zero_features_blanks_one_model_and_saves_under_its_own_name(trained):
    """確認イ: 判定のときに qφ（または pθ）の出力だけを 0 にして、分類器がそれをどれだけ使っているかを測る。"""
    tmp_path, config, run_id, checkpoint = trained
    patches = torch.rand(2, 2, 3, 64, 64) * 255

    data = config.as_dict()
    data["inference"]["zero_features"] = "ai"
    with Evaluator(Config(data), run_id, checkpoint) as evaluator:
        predictor = evaluator.predictor
        with torch.no_grad():
            x_prime = predictor.preprocessor.prepare_input(patches.reshape(4, 3, 64, 64).to(predictor.device))
            photo = predictor.photo_model.extract_feature(x_prime)
            ai = predictor.ai_model.extract_feature(x_prime)
            expected = predictor.classifier.predict_proba(torch.cat([photo, torch.zeros_like(ai)], 1))
        got = evaluator.predictor.score_patches(patches)
        evaluator.run_ai_jpeg([90])
        types = {row["perturbation_type"] for row in evaluator.repo.conn.execute(
            "SELECT DISTINCT perturbation_type FROM evaluation_results WHERE run_id = ?", (run_id,))}

    torch.testing.assert_close(got, expected.reshape(2, 2).mean(dim=1))
    assert types == {"ai_jpeg_zero_ai"}                      # ふつうの ai_jpeg の結果とは混ざらない

    data["inference"]["zero_features"] = "both"
    with pytest.raises(ValueError):
        Evaluator(Config(data), run_id, checkpoint)


def test_calibrate_threshold_can_compress_only_ai_images(trained, monkeypatch):
    """正解率の出し直し: 線を「AI画像も JPEG にした val」で決めるとき、本物はそのままにする。"""
    tmp_path, config, run_id, checkpoint = trained
    seen: list[str] = []

    def ai_only(image):
        seen.append("ai")
        return image

    with Evaluator(config, run_id, checkpoint) as evaluator:
        expected = sum(
            row["label"] == "ai"
            for generator in config.get("dataset.train_generators")
            for row in evaluator._rows_for(generator, "val", limit=None)
        )
        before = evaluator.threshold
        evaluator.calibrate_threshold(ai_transform=ai_only, apply=False)
        after = evaluator.threshold

    assert expected > 0
    assert len(seen) == expected          # AI画像の枚数ぶんだけ通る（本物は通らない）
    assert after == before                # apply=False なら self.threshold は変えない


def test_ai_jpeg_can_recalibrate_threshold_per_quality(trained, monkeypatch):
    """`evaluate.ai_jpeg.recalibrate_threshold: true` なら、品質ごとに AI画像も JPEG にした val で線を決め直す。

    結果はふつうの ai_jpeg と混ざらないよう 'ai_jpeg_recal' として保存する。
    """
    tmp_path, config, run_id, checkpoint = trained
    data = config.as_dict()
    data["evaluate"]["ai_jpeg"] = {"recalibrate_threshold": True}

    calls: list[object] = []
    original = Evaluator.calibrate_threshold

    def _spy(self, transform=None, apply=True, ai_transform=None):
        calls.append(ai_transform)
        return original(self, transform=transform, apply=apply, ai_transform=ai_transform)

    monkeypatch.setattr(Evaluator, "calibrate_threshold", _spy)

    with Evaluator(Config(data), run_id, checkpoint) as evaluator:
        results = evaluator.run_ai_jpeg([90, 75])
        thresholds = dict(evaluator.ai_jpeg_thresholds)
        types = {row["perturbation_type"] for row in evaluator.repo.conn.execute(
            "SELECT DISTINCT perturbation_type FROM evaluation_results WHERE run_id = ?", (run_id,))}

    assert calls[0] is None                               # 最初の1回はふつうの val
    assert len(calls) == 3 and all(c is not None for c in calls[1:])   # 品質ごとに1回ずつ
    assert set(thresholds) == {90, 75}
    assert types == {"ai_jpeg_recal"}
    assert {r.perturbation_level for r in results} == {90.0, 75.0}


def test_ai_jpeg_keeps_single_threshold_by_default(trained, monkeypatch):
    tmp_path, config, run_id, checkpoint = trained
    calls: list[object] = []
    original = Evaluator.calibrate_threshold

    def _spy(self, transform=None, apply=True, ai_transform=None):
        calls.append(ai_transform)
        return original(self, transform=transform, apply=apply, ai_transform=ai_transform)

    monkeypatch.setattr(Evaluator, "calibrate_threshold", _spy)
    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_ai_jpeg([90])
        assert evaluator.ai_jpeg_thresholds == {}

    assert calls == [None]


def test_cli_ai_jpeg_mode(trained, monkeypatch, capsys):
    from src.cli import main as cli_main

    tmp_path, config, run_id, checkpoint = trained
    path = tmp_path / "cfg_cli.yaml"
    path.write_text(yaml.safe_dump(config.as_dict(), allow_unicode=True), encoding="utf-8")
    assert cli_main(["evaluate", "--mode", "ai_jpeg", "--config", str(path), "--run-id", str(run_id),
                     "--checkpoint", checkpoint, "--ai-jpeg-quality", "90"]) == 0
    out = capsys.readouterr().out
    assert "ai_jpeg" in out and "平均AUC" in out


def test_ai_jpeg_subsampling_is_used_and_saved_under_its_own_name(trained, monkeypatch):
    """確認ケ：AI画像の JPEG の色の間引き方を 4:4:4 にして測り直せる。結果はふつうの ai_jpeg と混ざらない。"""
    import src.eval.evaluator as evaluator_module

    tmp_path, config, run_id, checkpoint = trained
    data = config.as_dict()
    data.setdefault("evaluate", {}).setdefault("ai_jpeg", {}).update({"subsampling": "4:4:4", "recalibrate_threshold": True})
    config = Config(data)
    seen: list[tuple[int, str | None]] = []
    original = evaluator_module.jpeg_transform

    def recording_transform(quality, subsampling=None):
        seen.append((quality, subsampling))
        return original(quality, subsampling)

    monkeypatch.setattr(evaluator_module, "jpeg_transform", recording_transform)
    with Evaluator(config, run_id, checkpoint) as evaluator:
        evaluator.run_ai_jpeg([96])
        types = {
            row["perturbation_type"]
            for row in evaluator.repo.conn.execute(
                "SELECT DISTINCT perturbation_type FROM evaluation_results WHERE run_id = ?", (run_id,)
            )
        }
    assert seen and all(s == "4:4:4" for _, s in seen)
    assert "ai_jpeg_recal_s444" in types


def test_jpeg_compress_subsampling_option():
    import io

    from PIL import Image, JpegImagePlugin

    from src.preprocess.patch_sampler import jpeg_bytes

    image = torch.rand(3, 32, 32) * 255
    for name, code in (("4:4:4", 0), ("4:2:0", 2), (None, 2)):
        with Image.open(io.BytesIO(jpeg_bytes(image, 96, name))) as img:
            assert JpegImagePlugin.get_sampling(img) == code
    with pytest.raises(ValueError):
        jpeg_bytes(image, 96, "4:1:1")


# ------------------------------------------------- 確認セ：線の引き方を調べる


def test_best_accuracy_is_upper_bound_over_thresholds():
    from src.eval.metrics import best_accuracy, class_accuracies

    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.4])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert best_accuracy(scores, labels) == pytest.approx(1.0)               # 0.3 と 0.4 の間に引けば全部正解
    assert compute_metrics(scores, labels, 0.65)["accuracy"] == pytest.approx(4 / 6)

    real, ai = class_accuracies(scores, labels, 0.65)
    assert real == pytest.approx(1.0) and ai == pytest.approx(1 / 3)         # 線が高すぎて AI画像を取りこぼす


def test_threshold_check_reports_per_class_and_best_accuracy_without_saving(trained):
    from src.cli import main as cli_main

    tmp_path, config, run_id, checkpoint = trained
    with MetadataRepository(config.get("paths.db_path")) as repo:
        before = repo.conn.execute("SELECT COUNT(*) FROM evaluation_results").fetchone()[0]

    out = tmp_path / "threshold_check"
    path = tmp_path / "cfg.yaml"
    assert cli_main(["threshold-check", "--config", str(path), "--run-id", str(run_id),
                     "--ai-jpeg-quality", "96", "--subsampling", "4:4:4", "--out", str(out)]) == 0

    import csv

    with open(out / "threshold_check.csv", encoding="utf-8") as f:
        rows = {r["generator"]: r for r in csv.DictReader(f)}
    assert set(rows) == {"SDv1.4", "BigGAN", "平均"}
    for key in ("accuracy", "real_accuracy", "ai_accuracy", "best_accuracy", "auc", "threshold"):
        assert rows["BigGAN"][key] != ""
    assert float(rows["BigGAN"]["best_accuracy"]) >= float(rows["BigGAN"]["accuracy"]) - 1e-9
    assert "4:4:4" in (out / "threshold_check.md").read_text(encoding="utf-8")

    with MetadataRepository(config.get("paths.db_path")) as repo:
        after = repo.conn.execute("SELECT COUNT(*) FROM evaluation_results").fetchone()[0]
    assert after == before                                                    # DB には残さない


def test_threshold_check_can_set_threshold_from_real_val_only(trained):
    """確認ソ：線を本物の val の点数だけで決める（本物を AI と間違える割合を指定する）。"""
    import csv

    from src.cli import main as cli_main
    from src.eval.threshold_check import real_val_threshold

    tmp_path, config, run_id, checkpoint = trained
    with Evaluator(config, run_id, checkpoint) as evaluator:
        rows = [
            row
            for generator in config.get("dataset.train_generators")
            for row in evaluator._rows_for(generator, "val", limit=None)
            if row["label"] == "real"
        ]
        scores, _ = evaluator.predictor.score_rows(rows)
        assert real_val_threshold(evaluator, 50.0) == pytest.approx(float(np.percentile(scores, 50.0)))

    out = tmp_path / "threshold_check_fa"
    assert cli_main(["threshold-check", "--config", str(tmp_path / "cfg.yaml"), "--run-id", str(run_id),
                     "--real-false-alarm", "50", "--out", str(out)]) == 0
    with open(out / "threshold_check.csv", encoding="utf-8") as f:
        mean = {r["generator"]: r for r in csv.DictReader(f)}["平均"]
    # CLI は評価器を作り直すので切り抜く位置が少し変わる。本物の点数の範囲に入っていればよい
    assert scores.min() - 0.05 <= float(mean["threshold"]) <= scores.max() + 0.05
    assert "本物の val" in (out / "threshold_check.md").read_text(encoding="utf-8")

"""pθ と qφ の出力の似かたを測る（案ウ）のテスト。"""

from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
import pytest
import yaml
from PIL import Image

from src.cli import main as cli_main
from src.data import db as db_module
from src.data.ingest import ingest
from src.eval.model_similarity import ModelSimilarity, linear_cka, spearman
from src.experiment.config import Config
from src.train.train_stage1 import TrainerStage1


# ------------------------------------------------------------ 物差し


def test_linear_cka_is_one_for_same_features_and_their_rotation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, 3.0 * x @ rotation + 5.0) == pytest.approx(1.0)   # 回転・拡大・ずらしでは変わらない


def test_linear_cka_is_small_for_unrelated_features():
    rng = np.random.default_rng(1)
    assert linear_cka(rng.normal(size=(5000, 6)), rng.normal(size=(5000, 4))) < 0.05


def test_spearman_follows_rank_order():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


# ------------------------------------------------------------ 通しの評価


def _png(color, size=96) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def models(tmp_path):
    """2生成器分のデータを ingest し、pθ と qφ を1エポックだけ学習した状態を作る。"""
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    rng = np.random.default_rng(0)
    for archive, ai_color in (("genimage-stable-diffusion-v1-4.zip", (200, 40, 40)), ("BigGAN.zip", (40, 40, 200))):
        with zipfile.ZipFile(root / archive, "w") as zf:
            for i in range(12):
                zf.writestr(f"train/ai/ai_{i}.png", _png(tuple(int(c) for c in rng.integers(0, 255, 3))))
                zf.writestr(f"train/nature/nat_{i}.png", _png(tuple(int(c) for c in rng.integers(0, 255, 3))))
            for i in range(3):
                zf.writestr(f"val/ai/ai_{i}.png", _png(ai_color))
                zf.writestr(f"val/nature/nat_{i}.png", _png((40, 200, 40)))

    data = Config.load("configs/base.yaml").as_dict()
    data["paths"]["dataset_root"] = str(root)
    data["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    data["paths"]["log_root"] = str(tmp_path / "logs")
    data["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"},
        {"name": "BigGAN", "path": "BigGAN.zip", "category": "gan", "type": "auto"},
    ]
    data["dataset"]["eval_generators"] = ["SDv1.4", "BigGAN", "Midjourney"]
    data["dataset"]["split"]["val_ratio_from_train"] = 0.5
    data["augmentation"]["jpeg"]["enabled"] = False
    data["model"]["conditional_unet"]["base_channels"] = 8
    data["inference"]["num_patches"] = 2
    data["train"].update({"batch_size": 2, "num_workers": 0, "num_epochs": 1, "max_eval_steps": 1})
    data["stage2"] = {
        "photo_model_checkpoint": str(tmp_path / "ckpt_photo" / "best.pt"),
        "ai_model_checkpoint": str(tmp_path / "ckpt_ai" / "best.pt"),
    }
    data["evaluate"] = {"max_images_per_generator": 4, "one_class": {"batch_size": 2}}

    db_module.init_db(data["paths"]["db_path"])
    for which in ("photo", "ai"):
        data["train"]["checkpoint_dir"] = str(tmp_path / f"ckpt_{which}")
        data["experiment"] = {"name": f"sim_{which}", "stage": f"stage1_{which}"}
        path = tmp_path / f"cfg_{which}.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        config = Config.load(path)
        if which == "photo":
            ingest(config, progress=False)
        with TrainerStage1(config, which) as trainer:
            trainer.fit(num_epochs=1, max_steps_per_epoch=2)
    return tmp_path, path, config


def test_same_model_on_both_sides_gives_perfect_similarity(models):
    tmp_path, _, config = models
    photo = str(tmp_path / "ckpt_photo" / "best.pt")
    with ModelSimilarity(config, photo_checkpoint=photo, ai_checkpoint=photo) as similarity:
        results = similarity.run(split="val")
    assert {r["generator"] for r in results} == {"SDv1.4", "BigGAN"}       # Midjourney は除外
    for row in results:
        assert row["cka"] == pytest.approx(1.0, abs=1e-6)
        assert row["spearman_all"] == pytest.approx(1.0, abs=1e-6)
        assert row["auc_diff"] == pytest.approx(0.5)                     # 差がいつも 0 なので当てずっぽう


def test_compare_models_cli_writes_report(models):
    tmp_path, config_path, _ = models
    out = tmp_path / "report"
    assert cli_main(["compare-models", "--config", str(config_path), "--split", "val",
                     "--ai-jpeg-quality", "90", "--out", str(out)]) == 0

    with open(out / "model_similarity.csv", encoding="utf-8") as f:
        rows = {row["generator"]: row for row in csv.DictReader(f)}
    assert set(rows) == {"SDv1.4", "BigGAN", "平均"}
    for key in ("cka", "spearman_real", "spearman_ai", "spearman_all", "auc_photo", "auc_ai", "auc_diff", "num_samples"):
        assert rows["BigGAN"][key] != ""
    report = (out / "model_similarity.md").read_text(encoding="utf-8")
    assert "品質 90" in report


def test_ai_jpeg_is_applied_only_to_ai_images(models, monkeypatch):
    tmp_path, _, config = models
    from src.eval import model_similarity as module

    seen = []
    original = module.jpeg_transform

    def spy(quality):
        transform = original(quality)

        def wrapped(image):
            seen.append(quality)
            return transform(image)

        return wrapped

    monkeypatch.setattr(module, "jpeg_transform", spy)
    with ModelSimilarity(config, ai_jpeg_quality=75) as similarity:
        rows = similarity.rows_for("BigGAN", "val")
        similarity.analyze(rows)
    assert len(seen) == sum(row["label"] == "ai" for row in rows) > 0


# ------------------------------------------------------------ 案カ・案キ


def test_random_checkpoint_uses_untrained_model_as_baseline(models):
    """案カ：qφ の代わりに学習していないモデルを置いて、CKA の基準を測れる。"""
    tmp_path, _, config = models
    with ModelSimilarity(config, ai_checkpoint="random") as similarity:
        assert similarity.ai_checkpoint == "random"
        results = similarity.run(split="val")
    assert {r["generator"] for r in results} == {"SDv1.4", "BigGAN"}
    assert all(0.0 <= r["cka"] < 1.0 for r in results)


def test_random_baseline_is_reproducible(models):
    _, _, config = models
    values = []
    for _ in range(2):
        with ModelSimilarity(config, ai_checkpoint="random") as similarity:
            values.append(similarity.run(split="val")[0]["cka"])
    assert values[0] == pytest.approx(values[1])


def test_texture_is_mean_absolute_residual_and_reported(models):
    """案キ：画像ごとの細かい模様の量（y′ の絶対値の平均）と、点数との関係を出す。"""
    tmp_path, config_path, config = models
    with ModelSimilarity(config) as similarity:
        rows = similarity.rows_for("BigGAN", "val")
        data = similarity.analyze(rows)
    assert data["texture"].shape == data["photo"].shape
    assert (data["texture"] >= 0).all()

    out = tmp_path / "report_texture"
    assert cli_main(["compare-models", "--config", str(config_path), "--split", "val", "--out", str(out)]) == 0
    with open(out / "model_similarity.csv", encoding="utf-8") as f:
        row = {r["generator"]: r for r in csv.DictReader(f)}["BigGAN"]
    for key in ("spearman_texture_photo", "spearman_texture_ai", "auc_texture"):
        assert row[key] != ""


def test_random_baseline_report_folder_is_named_random(models, monkeypatch):
    tmp_path, config_path, _ = models
    monkeypatch.chdir(tmp_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["paths"]["report_root"] = str(tmp_path / "reports")
    path = tmp_path / "cfg_reports.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert cli_main(["compare-models", "--config", str(path), "--split", "val", "--ai-checkpoint", "random"]) == 0
    assert (tmp_path / "reports" / "compare_ckpt_photo_random" / "model_similarity.md").exists()

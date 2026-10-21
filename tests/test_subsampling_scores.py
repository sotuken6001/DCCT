"""色の間引き方ごとに予測しにくさの点数を比べる（確認コ）のテスト。"""

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
from src.eval.subsampling_scores import SubsamplingScores, pair_auc
from src.experiment.config import Config
from src.train.train_stage1 import TrainerStage1


def _encode(seed, fmt="JPEG", **kwargs) -> bytes:
    array = np.random.default_rng(seed).integers(0, 255, (96, 96, 3)).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_pair_auc_direction():
    assert pair_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(1.0)   # 後ろのほうが点数が高い
    assert pair_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)   # 後ろのほうが予測しやすい


@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    with zipfile.ZipFile(root / "genimage-stable-diffusion-v1-4.zip", "w") as zf:
        for i in range(10):
            zf.writestr(f"train/ai/ai_{i}.png", _encode(i, fmt="PNG"))
            zf.writestr(f"train/nature/nat_{i}.JPEG", _encode(100 + i, quality=96, subsampling=0))
        for i in range(6):
            zf.writestr(f"val/ai/ai_{i}.png", _encode(200 + i, fmt="PNG"))
            sub = 0 if i % 2 == 0 else 2                                   # 本物に 4:4:4 と 4:2:0 を混ぜる
            zf.writestr(f"val/nature/nat_{i}.JPEG", _encode(300 + i, quality=96, subsampling=sub))

    data = Config.load("configs/base.yaml").as_dict()
    data["paths"]["dataset_root"] = str(root)
    data["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    data["paths"]["log_root"] = str(tmp_path / "logs")
    data["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"}
    ]
    data["dataset"]["split"]["val_ratio_from_train"] = 0.5
    data["augmentation"]["jpeg"]["enabled"] = False
    data["model"]["conditional_unet"]["base_channels"] = 8
    data["inference"]["num_patches"] = 2
    data["train"].update({"batch_size": 2, "num_workers": 0, "num_epochs": 1, "max_eval_steps": 1,
                          "checkpoint_dir": str(tmp_path / "ckpt_photo")})
    data["stage2"] = {"photo_model_checkpoint": str(tmp_path / "ckpt_photo" / "best.pt")}
    data["evaluate"] = {"one_class": {"batch_size": 2, "num_calibration_images": 4}}
    data["experiment"] = {"name": "subsampling_test", "stage": "stage1_photo"}

    db_module.init_db(data["paths"]["db_path"])
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = Config.load(path)
    ingest(config, progress=False)
    with TrainerStage1(config, "photo") as trainer:
        trainer.fit(num_epochs=1, max_steps_per_epoch=2)
    return tmp_path, path, config


def test_real_rows_are_split_by_their_jpeg_subsampling(project):
    _, _, config = project
    with SubsamplingScores(config, max_images=2) as scores:
        found = scores.real_rows_by_subsampling()
    assert len(found["4:4:4"]) == 2 and len(found["4:2:0"]) == 2
    assert all(row["label"] == "real" for rows in found.values() for row in rows)


def test_subsampling_scores_cli_writes_groups_and_pairs(project):
    tmp_path, config_path, _ = project
    out = tmp_path / "report"
    assert cli_main(["subsampling-scores", "--config", str(config_path), "--max-images", "3", "--out", str(out)]) == 0
    with open(out / "subsampling_scores.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = {r["group"]: r for r in rows if r["kind"] == "group"}
    assert set(groups) == {"本物 4:4:4", "本物 4:2:0", "AI画像 JPEG 4:2:0", "AI画像 JPEG 4:4:4"}
    assert int(groups["本物 4:4:4"]["num_images"]) == 3
    pairs = [r for r in rows if r["kind"] == "pair"]
    assert len(pairs) == 4 and all(r["auc"] != "" for r in pairs)
    assert (out / "subsampling_scores.md").exists()

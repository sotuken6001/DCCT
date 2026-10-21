"""本物の JPEG の作り（色の間引き方・圧縮の表）を数える（案ク）のテスト。"""

from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.cli import main as cli_main
from src.data import db as db_module
from src.data.ingest import ingest
from src.eval.jpeg_stats import inspect_image, summarize
from src.experiment.config import Config
from src.preprocess.patch_sampler import jpeg_bytes, jpeg_compress, load_image_tensor


def _array(seed=0, size=64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3)).astype("uint8")


def _encode(array, fmt="JPEG", **kwargs) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt, **kwargs)
    return buf.getvalue()


# ------------------------------------------------------------ 1枚を調べる


def test_inspect_reads_subsampling_and_quality_of_standard_tables():
    info = inspect_image(_encode(_array(), quality=96))
    assert info["format"] == "JPEG" and info["mode"] == "RGB"
    assert info["subsampling"] == "4:2:0"                         # PIL の初期設定
    assert info["quality"] == 96 and info["standard_tables"] is True

    info = inspect_image(_encode(_array(), quality=80, subsampling=0))
    assert info["subsampling"] == "4:4:4" and info["quality"] == 80


def test_inspect_flags_custom_tables_and_gives_nearest_quality():
    tables = {0: [3] * 64, 1: [5] * 64}                           # libjpeg の標準の表ではない
    info = inspect_image(_encode(_array(), qtables=tables))
    assert info["standard_tables"] is False
    assert 1 <= info["quality"] <= 100


def test_inspect_non_jpeg_and_grayscale():
    assert inspect_image(_encode(_array(), fmt="PNG"))["format"] == "PNG"
    assert inspect_image(_encode(_array(), fmt="PNG"))["subsampling"] is None

    gray = io.BytesIO()
    Image.fromarray(_array()[..., 0]).save(gray, format="JPEG", quality=90)
    info = inspect_image(gray.getvalue())
    assert info["mode"] == "L" and info["subsampling"] == "なし（白黒）" and info["quality"] == 90


def test_jpeg_bytes_is_what_jpeg_compress_decodes():
    """学習・評価で AI画像にかけている JPEG と、調べる JPEG が同じものであること。"""
    image = torch.from_numpy(_array()).permute(2, 0, 1).float()
    data = jpeg_bytes(image, 96)
    torch.testing.assert_close(jpeg_compress(image, 96), load_image_tensor(data))
    assert inspect_image(data)["subsampling"] == "4:2:0"


def test_summarize_counts_shares():
    infos = [
        inspect_image(_encode(_array(i), quality=q, subsampling=s))
        for i, (q, s) in enumerate([(96, 2), (96, 2), (90, 0), (75, 2)])
    ]
    summary = summarize("x", infos)
    assert summary["num_images"] == 4
    assert summary["jpeg_share"] == pytest.approx(1.0)
    assert summary["share_420"] == pytest.approx(0.75)
    assert summary["share_444"] == pytest.approx(0.25)
    assert summary["share_q96"] == pytest.approx(0.5)
    assert summary["top_qualities"].startswith("96:2")


# ------------------------------------------------------------ 通し


@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    with zipfile.ZipFile(root / "genimage-stable-diffusion-v1-4.zip", "w") as zf:
        for i in range(8):
            zf.writestr(f"train/ai/ai_{i}.png", _encode(_array(i, 96), fmt="PNG"))
            zf.writestr(f"train/nature/nat_{i}.JPEG", _encode(_array(100 + i, 96), quality=90, subsampling=0))

    data = Config.load("configs/base.yaml").as_dict()
    data["paths"]["dataset_root"] = str(root)
    data["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    data["paths"]["log_root"] = str(tmp_path / "logs")
    data["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"}
    ]
    db_module.init_db(data["paths"]["db_path"])
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    ingest(Config.load(path), progress=False)
    return tmp_path, path


def test_jpeg_stats_cli_compares_real_images_with_our_pil_jpeg(project):
    tmp_path, config_path = project
    out = tmp_path / "report"
    assert cli_main(["jpeg-stats", "--config", str(config_path), "--max-images", "5", "--out", str(out)]) == 0

    with open(out / "jpeg_stats.csv", encoding="utf-8") as f:
        rows = {row["group"]: row for row in csv.DictReader(f)}
    real = rows["ImageNet(real)@SDv1.4"]
    assert int(real["num_images"]) == 5
    assert float(real["share_444"]) == pytest.approx(1.0)
    assert real["top_qualities"].startswith("90:5")
    ours = rows["PIL JPEG96（AI画像にかけているもの）"]
    assert float(ours["share_420"]) == pytest.approx(1.0) and float(ours["share_q96"]) == pytest.approx(1.0)
    ai = rows["AI画像（元のまま）"]
    assert float(ai["jpeg_share"]) == pytest.approx(0.0)
    assert (out / "jpeg_stats.md").exists()

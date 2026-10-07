"""一クラス検出（論文 Appendix B の DCCT†）のテスト。"""

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
from src.eval.one_class import (
    PAPER_TABLE5_ACCURACY,
    OneClassEvaluator,
    nll_minus_entropy,
    value_log_probs,
)
from src.experiment.config import Config
from src.model.conditional_unet import MixtureParams
from src.train.train_stage1 import TrainerStage1


def _params(mu: float, log_s: float, shape=(1, 1, 1, 2, 2)) -> MixtureParams:
    """K=1・画素ごとの混合分布（成分1つ）を作る。"""
    return MixtureParams(
        log_w=torch.zeros(shape),
        mu=torch.full(shape, mu),
        log_s=torch.full(shape, log_s),
    )


# ------------------------------------------------------------ 分布の計算


def test_value_log_probs_are_normalized_over_quantized_values():
    table = value_log_probs(_params(mu=0.0, log_s=0.0), truncation_t=7)
    assert table.shape == (1, 15, 1, 2, 2)                    # {-7,...,7} の15値
    torch.testing.assert_close(table.exp().sum(dim=1), torch.ones(1, 1, 2, 2))


def test_score_is_nll_minus_entropy_of_predicted_distribution():
    params = _params(mu=0.0, log_s=0.0)
    table = value_log_probs(params, truncation_t=7)
    probs = table.exp()
    entropy = -(probs * table).sum(dim=1)                     # [1,1,2,2]

    y_prime = torch.full((1, 4, 2, 2), 3.0)                   # 4ch すべてが +3
    expected = -table[:, 7 + 3] - entropy                     # 値 +3 のビン
    score = nll_minus_entropy(y_prime, params, truncation_t=7)
    assert score.shape == (1,)
    torch.testing.assert_close(score, expected.mean().reshape(1))


def test_score_grows_when_residual_is_far_from_prediction():
    params = _params(mu=0.0, log_s=-1.0)
    near = nll_minus_entropy(torch.zeros(1, 2, 2, 2), params, truncation_t=7)
    far = nll_minus_entropy(torch.full((1, 2, 2, 2), 6.0), params, truncation_t=7)
    assert far.item() > near.item()


def test_score_supports_per_channel_mixtures():
    params = _params(mu=0.0, log_s=0.0, shape=(2, 3, 4, 2, 2))   # Cm = C = 4
    score = nll_minus_entropy(torch.zeros(2, 4, 2, 2), params, truncation_t=7)
    assert score.shape == (2,)


def test_value_log_probs_match_training_nll_for_per_channel_mixtures():
    """評価で作る確率表は、学習時の NLLLoss と同じ対数確率を返すこと。"""
    from src.model.losses import NLLLoss

    torch.manual_seed(0)
    shape = (2, 3, 4, 5, 5)                                   # B, K, C, H, W
    params = MixtureParams(
        log_w=torch.log_softmax(torch.randn(shape), dim=1),
        mu=torch.randn(shape) * 3,
        log_s=torch.randn(shape) * 0.5,
    )
    y_prime = torch.randint(-7, 8, (2, 4, 5, 5)).float()

    table = value_log_probs(params, truncation_t=7)           # [B,V,C,H,W]
    from_table = table.gather(1, (y_prime + 7).long().unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(from_table, NLLLoss(truncation_t=7).log_prob(y_prime, params))


def test_paper_table5_covers_all_genimage_generators():
    assert set(PAPER_TABLE5_ACCURACY) == {
        "Midjourney", "SDv1.4", "SDv1.5", "ADM", "GLIDE", "Wukong", "VQDM", "BigGAN"
    }
    assert PAPER_TABLE5_ACCURACY["BigGAN"] == pytest.approx(94.48)


# ------------------------------------------------------------ 通しの評価


def _png(color, size=96) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(params=["per_pixel", "per_channel"])
def photo_model(tmp_path, request):
    """2生成器分のデータをingestし、pθ を1エポックだけ学習した状態を作る。

    混合分布を画素ごとに1つ持つ形（per_pixel）と、チャンネルごとに持つ形（per_channel）の両方で通す。
    """
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    for archive, ai_color in (("genimage-stable-diffusion-v1-4.zip", (200, 40, 40)), ("BigGAN.zip", (40, 40, 200))):
        with zipfile.ZipFile(root / archive, "w") as zf:
            # 学習用とvalへの振り分けはパスで決まるので、少なすぎると学習用が0枚になりうる
            for i in range(12):
                zf.writestr(f"train/ai/ai_{i}.png", _png(ai_color))
                zf.writestr(f"train/nature/nat_{i}.png", _png((40, 200, 40)))
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
    data["model"]["conditional_unet"]["mixture_scope"] = request.param
    if request.param == "per_channel":
        # 分類器を使わないので、分類器入力が巨大になる組み合わせのチェックを避ける
        data["model"]["classifier"]["feature_source"] = "bottleneck"
    data["inference"]["num_patches"] = 2
    data["train"].update({"batch_size": 2, "num_workers": 0, "num_epochs": 1, "max_eval_steps": 1,
                          "checkpoint_dir": str(tmp_path / "ckpt_photo")})
    data["stage2"] = {"photo_model_checkpoint": str(tmp_path / "ckpt_photo" / "best.pt")}
    data["evaluate"] = {"max_images_per_generator": 4, "one_class": {"num_calibration_images": 4}}
    data["experiment"] = {"name": "one_class_test", "stage": "stage1_photo"}

    db_module.init_db(data["paths"]["db_path"])
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = Config.load(path)
    ingest(config, progress=False)

    with TrainerStage1(config, "photo") as trainer:
        trainer.fit(num_epochs=1, max_steps_per_epoch=2)
    return tmp_path, path, config


def test_threshold_is_95th_percentile_of_real_training_scores(photo_model, monkeypatch):
    tmp_path, _, config = photo_model
    with OneClassEvaluator(config) as evaluator:
        rows = evaluator.calibration_rows()
        assert rows and all(row["label"] == "real" and row["split"] == "train" for row in rows)
        assert {row["generator"] for row in rows} == {"ImageNet(real)@SDv1.4"}   # 学習に使う生成器の実写のみ
        assert len(rows) <= 4

        scores, _ = evaluator.score_rows(rows)
        assert evaluator.calibrate() == pytest.approx(float(np.percentile(scores, 95)))


def test_one_class_evaluation_writes_csv_with_paper_values(photo_model):
    tmp_path, config_path, _ = photo_model
    out = tmp_path / "report"
    assert cli_main(["evaluate-one-class", "--config", str(config_path), "--out", str(out)]) == 0

    with open(out / "one_class.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_generator = {row["generator"]: row for row in rows}
    # 【OI-1】Midjourney は excluded_generators により評価対象から外れる
    assert set(by_generator) == {"SDv1.4", "BigGAN", "平均"}
    big = by_generator["BigGAN"]
    assert float(big["paper_accuracy"]) == pytest.approx(94.48)
    for key in ("accuracy", "auc", "real_accuracy", "ai_accuracy", "threshold"):
        assert big[key] != ""
    assert (out / "one_class.md").exists()


# ------------------------------------------------------------ AI画像だけ JPEG にする（確認5）


def test_ai_jpeg_quality_compresses_only_ai_images(photo_model, monkeypatch):
    """本物は JPEG、AI画像は PNG という形式の差を消すため、AI画像だけを JPEG にしてから点数をつける。"""
    import src.eval.one_class as one_class

    seen: list[int] = []

    def recording_transform(quality):
        def transform(image):
            seen.append(quality)
            return image
        return transform

    monkeypatch.setattr(one_class, "jpeg_transform", recording_transform)
    _, _, config = photo_model
    with OneClassEvaluator(config, ai_jpeg_quality=75) as evaluator:
        evaluator.calibrate()
        assert seen == []                                    # 閾値は本物の写真だけで決めるので、圧縮しない
        expected = sum(
            row["label"] == "ai"
            for generator in ("SDv1.4", "BigGAN")
            for row in evaluator._rows_for(generator, "test")
        )
        evaluator.run()
    assert expected > 0
    assert seen == [75] * expected                           # AI画像だけが、指定した品質で圧縮される


def test_ai_jpeg_quality_must_be_between_1_and_100(photo_model):
    _, _, config = photo_model
    for quality in (0, 101):
        with pytest.raises(ValueError):
            OneClassEvaluator(config, ai_jpeg_quality=quality)


def test_ai_jpeg_quality_is_written_in_report(photo_model):
    tmp_path, config_path, _ = photo_model
    out = tmp_path / "report_jpeg"
    assert cli_main(["evaluate-one-class", "--config", str(config_path), "--out", str(out),
                     "--ai-jpeg-quality", "90"]) == 0
    assert "AI画像だけ JPEG（品質 90）" in (out / "one_class.md").read_text(encoding="utf-8")

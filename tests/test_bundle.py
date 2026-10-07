"""学習済みモデルを1つのファイルにまとめて配る仕組み（export-bundle / predict）のテスト。"""

from __future__ import annotations

import io

import pytest
import torch
import yaml
from PIL import Image

from src.cli import main as cli_main
from src.eval.bundle import DCCTDetector, export_bundle
from src.eval.predictor import DCCTPredictor
from src.experiment import builders
from src.experiment.config import Config


def _config(tmp_path, use_ai_model: bool) -> Config:
    data = Config.load("configs/base.yaml").as_dict()
    data["device"]["type"] = "cpu"
    data["model"]["conditional_unet"]["base_channels"] = 8
    data["inference"]["num_patches"] = 2
    data["stage2"] = {
        "use_photo_model": True,
        "use_ai_model": use_ai_model,
        "photo_model_checkpoint": str(tmp_path / "photo.pt"),
        "ai_model_checkpoint": str(tmp_path / "ai.pt"),
    }
    return Config(data)


@pytest.fixture(params=[True, False], ids=["photo_and_ai", "photo_only"])
def trained(tmp_path, request):
    """Stage I の pθ / qφ と Stage II の分類器のチェックポイントを、学習のときと同じ形で作る。"""
    torch.manual_seed(0)
    config = _config(tmp_path, use_ai_model=request.param)
    preprocessor = builders.build_preprocessor(config)
    models = {}
    for which in ("photo", "ai"):
        model = builders.build_conditional_unet(config, preprocessor)
        torch.save({"modules": {"conditional_unet": model.state_dict()}}, tmp_path / f"{which}.pt")
        models[which] = model
    in_channels = builders.classifier_input_channels(config, models["photo"])
    classifier = builders.build_classifier(config, in_channels)
    torch.save({"modules": {"classifier": classifier.state_dict()}, "epoch": 3, "optimizer": {}}, tmp_path / "best.pt")
    return tmp_path, config


def _image(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, 96, 96, generator=generator) * 255


def test_bundle_gives_same_score_as_the_original_checkpoints(trained):
    tmp_path, config = trained
    out = export_bundle(config, tmp_path / "best.pt", tmp_path / "bundle.pt", threshold=0.4)

    detector = DCCTDetector.from_bundle(out, device="cpu")
    predictor = DCCTPredictor(config, tmp_path / "best.pt", device=torch.device("cpu"))

    image = _image()
    patches = detector.patches(image)
    expected = predictor.score_patches(patches.unsqueeze(0)).item()
    assert detector.score(image) == pytest.approx(expected, abs=1e-6)
    assert detector.threshold == pytest.approx(0.4)


def test_bundle_only_holds_the_models_it_needs(trained):
    tmp_path, config = trained
    out = export_bundle(config, tmp_path / "best.pt", tmp_path / "bundle.pt")
    bundle = torch.load(out, weights_only=False)

    expected = {"classifier", "photo_model"} | ({"ai_model"} if config.get("stage2.use_ai_model") else set())
    assert set(bundle["modules"]) == expected
    assert "optimizer" not in bundle                       # 学習の状態は入れない
    assert bundle["metadata"]["epoch"] == 3


def test_detector_accepts_path_pil_and_tensor(trained, tmp_path):
    root, config = trained
    detector = DCCTDetector.from_bundle(export_bundle(config, root / "best.pt", root / "bundle.pt"), device="cpu")

    array = _image().round().to(torch.uint8).permute(1, 2, 0).numpy()
    pil = Image.fromarray(array)
    path = tmp_path / "photo.png"
    pil.save(path)

    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    assert detector.score(tensor) == pytest.approx(detector.score(pil), abs=1e-6)
    assert 0.0 <= detector.score(path) <= 1.0
    assert detector.score(path) == detector.score(path)    # 同じ画像なら同じパッチ位置・同じ点数

    result = detector.predict(path)
    assert result["label"] in ("ai", "real") and result["threshold"] == 0.5   # 線を入れていなければ 0.5


def test_embedding_is_the_classifier_feature_before_the_last_layer(trained):
    tmp_path, config = trained
    detector = DCCTDetector.from_bundle(export_bundle(config, tmp_path / "best.pt", tmp_path / "bundle.pt"), device="cpu")

    image = _image()
    embedding = detector.embedding(image)
    assert embedding.shape == (detector.classifier.embed_dim,)

    features = detector.patch_features(image)
    logits = detector.classifier.head(detector.classifier.embed(features)).squeeze(-1)
    torch.testing.assert_close(logits, detector.classifier(features))


def test_rejects_files_that_are_not_bundles(tmp_path):
    torch.save({"modules": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        DCCTDetector.from_bundle(tmp_path / "x.pt", device="cpu")


def test_cli_export_and_predict(trained, capsys):
    tmp_path, config = trained
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config.as_dict(), allow_unicode=True), encoding="utf-8")
    image_path = tmp_path / "photo.png"
    Image.fromarray(_image().round().to(torch.uint8).permute(1, 2, 0).numpy()).save(image_path)

    assert cli_main(["export-bundle", "--config", str(config_path), "--checkpoint", str(tmp_path / "best.pt"),
                     "--threshold", "0.4", "--out", str(tmp_path / "dcct.pt")]) == 0
    assert cli_main(["predict", "--bundle", str(tmp_path / "dcct.pt"), "--device", "cpu", str(image_path)]) == 0
    out = capsys.readouterr().out
    assert "photo.png" in out and "0.4000" in out


def test_bundle_does_not_keep_absolute_paths(trained):
    """配るファイルに、学習した PC のフォルダ名（ユーザー名を含む）を残さない。"""
    tmp_path, config = trained
    bundle = torch.load(export_bundle(config, tmp_path / "best.pt", tmp_path / "bundle.pt"), weights_only=False)
    assert bundle["metadata"]["classifier_checkpoint"] == "best.pt"

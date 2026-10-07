"""条件付きモデル・損失・分類器のテスト。"""

from __future__ import annotations

import math

import pytest
import torch

from src.experiment import builders
from src.experiment.config import Config
from src.experiment.paths import PROJECT_ROOT
from src.model.classifier import BinaryClassifier
from src.model.conditional_unet import ConditionalUNet
from src.model.losses import BCELoss, NLLLoss
from src.preprocess.pipeline import DCCTPreprocessor


# --------------------------------------------------------- ConditionalUNet


def test_forward_shapes():
    """論文 Algorithm 1 line 10: 混合分布パラメータは画素ごとに1組（per_pixel）。"""
    net = ConditionalUNet(in_channels=30, target_channels=60, num_mixtures=10)
    params = net(torch.randn(2, 30, 64, 64))
    for tensor in (params.log_w, params.mu, params.log_s):
        assert tensor.shape == (2, 10, 1, 64, 64)


def test_per_channel_scope_gives_params_per_channel():
    """比較用の per_channel ではチャンネルごとにパラメータを持つこと。"""
    net = ConditionalUNet(in_channels=30, target_channels=2, num_mixtures=10,
                          mixture_scope="per_channel")
    params = net(torch.randn(2, 30, 64, 64))
    assert params.log_w.shape == (2, 10, 2, 64, 64)
    assert net.feature_channels == 60


def test_mixture_weights_sum_to_one():
    net = ConditionalUNet()
    params = net(torch.randn(2, 30, 32, 32))
    assert torch.allclose(params.w.sum(dim=1), torch.ones(2, 2, 32, 32), atol=1e-5)


def test_feature_map_is_60ch_when_concatenated():
    """【OI-2 確定】論文 Algorithm 1 line 9/32: f_θ(x') = 混合分布パラメータ。

    line 10 がパラメータを画素位置のみで添字付けしているため 3K = 30ch/モデル、
    pθ・qφ連結で60ch（03_詳細設計書 1.3節の記述とも一致）。
    """
    net = ConditionalUNet(num_mixtures=10, target_channels=60, feature_source="mixture_params")
    feature = net.extract_feature(torch.randn(2, 30, 64, 64))
    assert feature.shape == (2, 30, 64, 64)
    assert net.feature_channels == 30
    assert torch.cat([feature, feature], dim=1).shape[1] == 60


def test_bottleneck_feature_source():
    net = ConditionalUNet(base_channels=64, feature_source="bottleneck")
    feature = net.extract_feature(torch.randn(2, 30, 64, 64))
    assert feature.shape == (2, 64, 64, 64)
    assert net.feature_channels == 64


def test_unknown_feature_source_rejected():
    with pytest.raises(ValueError, match="feature_source"):
        ConditionalUNet(feature_source="something_else")


# 縮小2回（depth=2）の U-Net が持っていた部品の名前。学習済みの best.pt はこの名前で保存されている
_DEPTH2_MODULES = {"enc1", "enc2", "enc3", "up2", "dec2", "up1", "dec1", "head"}


def test_default_depth_keeps_module_names_of_existing_checkpoints():
    """depth を指定しなければ従来と同じ構造で、これまでの best.pt がそのまま読めること。"""
    net = ConditionalUNet()
    assert net.depth == 2
    assert {key.split(".")[0] for key in net.state_dict()} == _DEPTH2_MODULES

    explicit = ConditionalUNet(depth=2)
    explicit.load_state_dict(net.state_dict())             # 名前も形も一致する
    x = torch.randn(1, 30, 64, 64)
    torch.testing.assert_close(explicit(x).mu, net(x).mu)


def test_deeper_unet_keeps_output_shapes_and_has_more_parameters():
    shallow = ConditionalUNet(depth=2)
    deep = ConditionalUNet(depth=4)
    params = deep(torch.randn(2, 30, 64, 64))
    for tensor in (params.log_w, params.mu, params.log_s):
        assert tensor.shape == (2, 10, 1, 64, 64)
    assert {"enc5", "up4", "dec4"} <= {key.split(".")[0] for key in deep.state_dict()}

    def count(net):
        return sum(p.numel() for p in net.parameters())

    assert count(deep) > 10 * count(shallow)
    # 縮小4回・幅64は標準の U-Net と同じく約3,100万パラメータ
    assert 25_000_000 < count(deep) < 40_000_000


def test_wider_unet_via_base_channels():
    wide = ConditionalUNet(base_channels=128)
    assert wide(torch.randn(1, 30, 64, 64)).mu.shape == (1, 10, 1, 64, 64)
    assert wide.extract_feature(torch.randn(1, 30, 64, 64)).shape == (1, 30, 64, 64)


def test_invalid_depth_rejected():
    with pytest.raises(ValueError, match="depth"):
        ConditionalUNet(depth=0)


def test_patch_size_must_be_divisible_by_unet_downsampling(tmp_path):
    """パッチの大きさが 2^depth で割り切れないと、縮小と拡大で形が合わなくなるので弾くこと。"""
    import yaml

    data = Config.load(PROJECT_ROOT / "configs" / "base.yaml").as_dict()
    data["model"]["conditional_unet"]["depth"] = 7           # 64 は 2^7=128 で割り切れない
    path = tmp_path / "too_deep.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="depth"):
        Config.load(path)

    data["model"]["conditional_unet"]["depth"] = 4
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = Config.load(path)
    model = builders.build_conditional_unet(config, builders.build_preprocessor(config))
    assert model.depth == 4


def test_p_and_q_models_do_not_share_weights():
    """pθ と qφ は独立した重みを持つこと（03_詳細設計書 1.2）。"""
    torch.manual_seed(0)
    p_model = ConditionalUNet()
    q_model = ConditionalUNet()
    p_params = dict(p_model.named_parameters())
    assert any(
        not torch.equal(p_params[name], value) for name, value in q_model.named_parameters()
    )


def test_freeze_stops_gradients():
    net = ConditionalUNet().freeze()
    assert all(not p.requires_grad for p in net.parameters())

    feature = net.extract_feature(torch.randn(1, 30, 64, 64))
    assert not feature.requires_grad


def test_gradients_flow_in_stage1():
    net = ConditionalUNet()
    loss = NLLLoss()(torch.randint(-7, 8, (2, 2, 64, 64)).float(), net(torch.randn(2, 30, 64, 64)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


# ------------------------------------------------------------------ NLLLoss


def _params(mu: float, log_s: float, shape=(1, 1, 1, 1, 1)) -> object:
    from src.model.conditional_unet import MixtureParams

    return MixtureParams(
        log_w=torch.zeros(shape),                      # K=1 なので log_softmax 済みと同じ
        mu=torch.full(shape, mu),
        log_s=torch.full(shape, log_s),
    )


def test_discretized_likelihood_sums_to_one():
    """{-7,...,7} 全体にわたる確率の総和が1になること（正しく離散化されている）。"""
    loss = NLLLoss(truncation_t=7.0)
    params = _params(mu=0.5, log_s=0.3)
    total = sum(
        loss.log_prob(torch.full((1, 1, 1, 1), float(v)), params).exp().item()
        for v in range(-7, 8)
    )
    assert total == pytest.approx(1.0, abs=1e-5)


def test_likelihood_is_higher_near_the_mean():
    loss = NLLLoss(truncation_t=7.0)
    params = _params(mu=2.0, log_s=0.0)
    near = loss.log_prob(torch.full((1, 1, 1, 1), 2.0), params).item()
    far = loss.log_prob(torch.full((1, 1, 1, 1), -6.0), params).item()
    assert near > far


def test_wide_scale_approaches_uniform_on_interior_bins():
    """スケールを大きくすると内側のビンの確率がほぼ一様になること。"""
    loss = NLLLoss(truncation_t=7.0)
    params = _params(mu=0.0, log_s=math.log(20.0))
    probs = [
        loss.log_prob(torch.full((1, 1, 1, 1), float(v)), params).exp().item()
        for v in range(-6, 7)      # 端のビンは裾を含むので内側のみ比較する
    ]
    assert max(probs) / min(probs) < 1.2


def test_confident_prediction_beats_vague_one():
    """正解に近い平均・小さいスケールのほうがNLLが小さいこと。"""
    loss = NLLLoss(truncation_t=7.0)
    target = torch.full((1, 1, 1, 1), 2.0)
    sharp = -loss.log_prob(target, _params(mu=2.0, log_s=math.log(0.5))).item()
    vague = -loss.log_prob(target, _params(mu=2.0, log_s=math.log(20.0))).item()
    wrong = -loss.log_prob(target, _params(mu=-5.0, log_s=math.log(0.5))).item()
    assert sharp < vague < wrong


def test_nll_is_finite_at_extremes():
    """スケールが極端でも有限値にとどまること（学習の数値安定性）。"""
    loss = NLLLoss(truncation_t=7.0)
    for log_s in (-7.0, -3.0, 0.0, 3.0):
        value = loss.log_prob(torch.full((1, 1, 1, 1), 7.0), _params(-7.0, log_s))
        assert torch.isfinite(value).all()


def test_reduction_modes():
    net = ConditionalUNet()
    params = net(torch.randn(2, 30, 16, 16))
    target = torch.randint(-7, 8, (2, 2, 16, 16)).float()

    mean = NLLLoss(reduction="mean")(target, params)
    total = NLLLoss(reduction="sum")(target, params)
    per_sample = NLLLoss(reduction="sum_per_sample")(target, params)

    num_elements = 2 * 2 * 16 * 16
    assert total.item() == pytest.approx(mean.item() * num_elements, rel=1e-4)
    assert per_sample.item() == pytest.approx(total.item() / 2, rel=1e-4)


def test_unknown_reduction_rejected():
    with pytest.raises(ValueError, match="reduction"):
        NLLLoss(reduction="median")


# ---------------------------------------------------------- BinaryClassifier


def test_classifier_output_shape_and_tokens():
    clf = BinaryClassifier(in_channels=120)
    assert clf(torch.randn(4, 120, 64, 64)).shape == (4,)
    assert clf.num_tokens == 16          # 64 -> 4x4
    assert clf.embed_dim == 256


def test_classifier_rejects_wrong_channels():
    clf = BinaryClassifier(in_channels=120)
    with pytest.raises(ValueError, match="入力チャンネル数"):
        clf(torch.randn(2, 60, 64, 64))


def test_classifier_accepts_single_model_features():
    """Ablation-A: pθのみ/qφのみのとき60ch入力になること。"""
    assert BinaryClassifier(in_channels=60)(torch.randn(2, 60, 64, 64)).shape == (2,)


def test_predict_proba_in_unit_range():
    proba = BinaryClassifier(in_channels=120).predict_proba(torch.randn(8, 120, 64, 64))
    assert proba.shape == (8,) and (proba >= 0).all() and (proba <= 1).all()


def test_classifier_can_overfit_two_samples():
    """学習が進むこと（勾配が正しく流れていること）の最小確認。"""
    torch.manual_seed(0)
    clf = BinaryClassifier(in_channels=8, base_channels=8)
    features = torch.randn(2, 8, 64, 64)
    labels = torch.tensor([0.0, 1.0])
    loss_fn = BCELoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    first = loss_fn(clf(features), labels).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_fn(clf(features), labels)
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.5


# ------------------------------------------------------ config経由の組み立て


def test_builders_wire_config_to_paper_shapes():
    """論文準拠の既定: x'=30ch, y'=60ch, 特徴30ch/モデル, 分類器入力60ch。"""
    config = Config.load(PROJECT_ROOT / "configs" / "stage2_classifier.yaml")
    preprocessor = builders.build_preprocessor(config)
    model = builders.build_conditional_unet(config, preprocessor)
    channels = builders.classifier_input_channels(config, model)

    assert preprocessor.input_channels == 30      # 30フィルタ × 観測1ch
    assert preprocessor.target_channels == 60     # 30フィルタ × 隠された2ch（Alg.1 line 8）
    assert channels == 60                         # 3K=30ch × 2モデル
    assert builders.build_classifier(config, channels).in_channels == 60


def test_builders_respect_ablation_a():
    """Ablation-A: 片方のモデルだけ使う設定では入力が60chになること。"""
    data = Config.load(PROJECT_ROOT / "configs" / "stage2_classifier.yaml").as_dict()
    data["stage2"]["use_ai_model"] = False
    config = Config(data)

    model = builders.build_conditional_unet(config, builders.build_preprocessor(config))
    assert builders.classifier_input_channels(config, model) == 30

    data["stage2"]["use_photo_model"] = False
    with pytest.raises(ValueError, match="少なくとも一方"):
        builders.classifier_input_channels(Config(data), model)


def test_filter_bank_with_per_channel_scope_is_rejected(tmp_path):
    """y'60ch × チャンネルごとのパラメータは分類器入力3600chになるため弾くこと。"""
    import yaml

    data = Config.load(PROJECT_ROOT / "configs" / "base.yaml").as_dict()
    data["model"]["conditional_unet"]["mixture_scope"] = "per_channel"
    path = tmp_path / "conflict.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="3600ch"):
        Config.load(path)


# ------------------------------------------------ 前処理〜分類まで通した確認


def test_end_to_end_shapes_from_patch_to_logit():
    """Algorithm 1 / 2 の一連の流れが論文どおりの形状で通ること。"""
    preprocessor = DCCTPreprocessor()
    p_model = ConditionalUNet(preprocessor.input_channels, preprocessor.target_channels).freeze()
    q_model = ConditionalUNet(preprocessor.input_channels, preprocessor.target_channels).freeze()
    classifier = BinaryClassifier(in_channels=60)

    patches = torch.rand(3, 3, 64, 64) * 255
    x_prime, y_prime = preprocessor.prepare(patches)
    # Algorithm 1 line 8: x' も y' も30種のフィルタをSTACKしたもの
    assert x_prime.shape == (3, 30, 64, 64) and y_prime.shape == (3, 60, 64, 64)

    nll = NLLLoss()(y_prime, ConditionalUNet()(x_prime))
    assert torch.isfinite(nll)

    feature = torch.cat([p_model.extract_feature(x_prime), q_model.extract_feature(x_prime)], dim=1)
    assert feature.shape == (3, 60, 64, 64)
    assert classifier(feature).shape == (3,)


def test_inference_averages_patch_scores():
    """Algorithm 2: 1画像16パッチのスコア平均が閾値判定に使えること。"""
    preprocessor = DCCTPreprocessor()
    p_model = ConditionalUNet().freeze()
    q_model = ConditionalUNet().freeze()
    classifier = BinaryClassifier(in_channels=60)

    patches = torch.rand(16, 3, 64, 64) * 255
    x_prime = preprocessor.prepare_input(patches)
    feature = torch.cat([p_model.extract_feature(x_prime), q_model.extract_feature(x_prime)], dim=1)
    scores = classifier.predict_proba(feature)

    assert scores.shape == (16,)
    mean_score = scores.mean()
    assert 0.0 <= mean_score.item() <= 1.0
    assert int(mean_score.item() > 0.5) in (0, 1)

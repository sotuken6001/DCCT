"""前処理（CFAマスク / ハイパスフィルタ / パッチサンプリング）のテスト。"""

from __future__ import annotations

import io
import os

import pytest
import torch
from PIL import Image

from src.preprocess.cfa_mask import BAYER_PATTERNS, CFAMask, RandomChannelMask, build_mask
from src.preprocess.highpass import HighPassFilterBank, build_srm_kernels
from src.preprocess.patch_sampler import PatchSampler, jpeg_compress, load_image_tensor
from src.preprocess.pipeline import DCCTPreprocessor


# ---------------------------------------------------------------- CFAマスク


def test_cfa_splits_into_one_and_two_channels():
    image = torch.rand(3, 64, 64) * 255
    x, y = CFAMask().apply(image)
    assert x.shape == (1, 64, 64)
    assert y.shape == (2, 64, 64)


def test_cfa_is_lossless_partition():
    """x と y を合わせると元のRGB3chがちょうど復元できること（重複も欠落もない）。"""
    image = torch.rand(3, 16, 16) * 255
    mask = CFAMask()
    x, y = mask.apply(image)
    observed = mask.observed_index(16, 16)

    recovered = torch.empty_like(image)
    for row in range(16):
        for col in range(16):
            channel = int(observed[0, row, col])
            remaining = [c for c in range(3) if c != channel]
            recovered[channel, row, col] = x[0, row, col]
            recovered[remaining[0], row, col] = y[0, row, col]
            recovered[remaining[1], row, col] = y[1, row, col]
    assert torch.equal(recovered, image)


def test_bayer_tile_repeats_every_two_pixels():
    observed = CFAMask("RGGB").observed_index(8, 8)
    assert torch.equal(observed[0, :2, :2], torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(observed[0, :2, :2], observed[0, 4:6, 4:6])


@pytest.mark.parametrize("pattern", sorted(BAYER_PATTERNS))
def test_all_bayer_patterns_work(pattern):
    x, y = CFAMask(pattern).apply(torch.rand(3, 8, 8))
    assert x.shape == (1, 8, 8) and y.shape == (2, 8, 8)


def test_batched_input():
    x, y = CFAMask().apply(torch.rand(4, 3, 64, 64))
    assert x.shape == (4, 1, 64, 64) and y.shape == (4, 2, 64, 64)


def test_random_mask_has_no_bayer_structure():
    """Ablation-B: ランダムマスクは2×2周期を持たないこと。"""
    generator = torch.Generator().manual_seed(0)
    image = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)
    x_random, _ = RandomChannelMask(generator).apply(image)
    x_bayer, _ = CFAMask().apply(image)
    assert not torch.equal(x_random, x_bayer)


def test_build_mask_switches_by_flag():
    assert isinstance(build_mask(True), CFAMask)
    assert isinstance(build_mask(False), RandomChannelMask)


def test_rejects_non_rgb():
    with pytest.raises(ValueError, match="3ch"):
        CFAMask().apply(torch.rand(1, 8, 8))


# ------------------------------------------------------- ハイパスフィルタ


def test_thirty_distinct_kernels():
    kernels, names = build_srm_kernels()
    assert kernels.shape == (30, 1, 5, 5)
    assert len(set(names)) == 30
    # 同じ係数のカーネルが重複していないこと
    flat = {tuple(k.flatten().tolist()) for k in kernels}
    assert len(flat) == 30


def test_kernels_are_dc_free():
    """すべてハイパスであること（係数和=0 → 平坦画像への応答が0）。"""
    kernels, _ = build_srm_kernels()
    assert torch.allclose(kernels.sum(dim=(1, 2, 3)), torch.zeros(30), atol=1e-6)

    bank = HighPassFilterBank(truncation_t=7.0)
    assert bank.apply(torch.full((1, 32, 32), 128.0)).abs().max().item() == 0.0


def test_channel_expansion_and_truncation():
    bank = HighPassFilterBank(truncation_t=7.0)
    out = bank.apply(torch.rand(2, 64, 64) * 255)
    assert out.shape == (60, 64, 64)          # 30種 × 2ch
    assert out.min() >= -7.0 and out.max() <= 7.0


def test_truncation_threshold_is_configurable():
    """Ablation-C: t を変えるとクリップ範囲が変わること。"""
    noise = torch.rand(1, 64, 64) * 255
    for t in (3, 7, 15):
        out = HighPassFilterBank(truncation_t=t).apply(noise)
        assert out.abs().max() <= t


def test_residual_is_quantized_to_integers():
    out = HighPassFilterBank(truncation_t=7.0, quantize=True).apply(torch.rand(1, 32, 32) * 255)
    assert torch.equal(out, out.round())
    assert set(out.unique().tolist()) <= {float(v) for v in range(-7, 8)}

    unquantized = HighPassFilterBank(truncation_t=7.0, quantize=False).apply(torch.rand(1, 32, 32) * 255)
    assert not torch.equal(unquantized, unquantized.round())


@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_residual_scale_makes_finer_steps(scale):
    image = torch.rand(1, 32, 32) * 255
    raw = HighPassFilterBank(truncation_t=1000.0, quantize=False).apply(image)
    scaled = HighPassFilterBank(truncation_t=2 * scale, residual_scale=scale).apply(image)
    # 元の値で言うと、範囲 ±2 を 1/scale 刻みで丸めたものになる
    assert torch.equal(scaled, torch.clamp(torch.round(raw * scale), -2 * scale, 2 * scale))
    assert len(scaled.unique()) > len(HighPassFilterBank(truncation_t=2.0).apply(image).unique())


def test_residual_scale_one_is_unchanged():
    image = torch.rand(1, 32, 32) * 255
    assert torch.equal(
        HighPassFilterBank(truncation_t=2.0, residual_scale=1.0).apply(image),
        HighPassFilterBank(truncation_t=2.0).apply(image),
    )


def test_residual_scale_must_be_positive():
    with pytest.raises(ValueError):
        HighPassFilterBank(residual_scale=0)


def test_disabled_bank_passes_through():
    """Ablation-B: ハイパスフィルタなしのときはチャンネル数が増えないこと。"""
    bank = HighPassFilterBank(truncation_t=7.0, enabled=False)
    out = bank.apply(torch.rand(2, 64, 64) * 255)
    assert out.shape == (2, 64, 64)


def test_apply_single_keeps_channel_count():
    bank = HighPassFilterBank(truncation_t=7.0)
    assert bank.apply_single(torch.rand(2, 32, 32) * 255, "square5x5").shape == (2, 32, 32)
    with pytest.raises(ValueError, match="未知のカーネル名"):
        bank.apply_single(torch.rand(2, 32, 32), "no_such_kernel")


# --------------------------------------------------- パッチサンプリング


def _image_bytes(width=200, height=150):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (30, 90, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_tensor_is_0_255_scale():
    image = load_image_tensor(_image_bytes())
    assert image.shape == (3, 150, 200)
    assert torch.equal(image[:, 0, 0], torch.tensor([30.0, 90.0, 200.0]))


def test_sample_train_returns_single_patch():
    sampler = PatchSampler(patch_size=64, jpeg_enabled=False)
    patch = sampler.sample_train(torch.rand(3, 150, 200) * 255)
    assert patch.shape == (3, 64, 64)


def test_sample_test_returns_p_patches_deterministically():
    """同じ画像キーなら毎回同じ位置のパッチが得られること（NFR-1）。"""
    sampler = PatchSampler(patch_size=64, jpeg_enabled=False, seed=42)
    image = torch.rand(3, 150, 200) * 255
    first = sampler.sample_test(image, num_patches=16, image_key="img-1")
    second = sampler.sample_test(image, num_patches=16, image_key="img-1")
    other = sampler.sample_test(image, num_patches=16, image_key="img-2")

    assert first.shape == (16, 3, 64, 64)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_too_small_image_is_rejected():
    with pytest.raises(ValueError, match="パッチサイズ"):
        PatchSampler(patch_size=64).sample_train(torch.rand(3, 32, 32))


def test_jpeg_compression_changes_pixels():
    image = torch.rand(3, 64, 64) * 255
    compressed = jpeg_compress(image, quality=70)
    assert compressed.shape == image.shape
    assert not torch.equal(compressed, image.round())


def test_jpeg_augmentation_probability():
    """適用確率どおりに（おおむね）JPEGがかかること。"""
    sampler = PatchSampler(jpeg_enabled=True, jpeg_probability=1.0)
    image = torch.rand(3, 64, 64) * 255
    assert not torch.equal(sampler.augment(image, torch.Generator().manual_seed(0)), image)

    never = PatchSampler(jpeg_enabled=True, jpeg_probability=0.0)
    assert torch.equal(never.augment(image), image)


# ------------------------------------------------------------ パイプライン


def test_pipeline_produces_30ch_input_and_60ch_target():
    """論文 Algorithm 1 line 8: x' も y' も30種のフィルタをSTACKしたもの。"""
    pre = DCCTPreprocessor(truncation_t=7.0)
    x_prime, y_prime = pre.prepare(torch.rand(4, 3, 64, 64) * 255)
    assert x_prime.shape == (4, 30, 64, 64)      # 30フィルタ × 観測1ch
    assert y_prime.shape == (4, 60, 64, 64)      # 30フィルタ × 隠された2ch
    assert pre.input_channels == 30 and pre.target_channels == 60
    assert x_prime.abs().max() <= 7 and y_prime.abs().max() <= 7


def test_pipeline_single_filter_mode():
    """比較用モードでは y' が2chに保たれること。"""
    pre = DCCTPreprocessor(target_mode="single_filter")
    _, y_prime = pre.prepare(torch.rand(2, 3, 64, 64) * 255)
    assert y_prime.shape == (2, 2, 64, 64)
    assert pre.target_channels == 2


def test_pipeline_ablation_b_combinations():
    """Ablation-B（Table 4b）: ハイパスフィルタ有無 × CFAマスク有無。"""
    cases = (
        # (CFAマスク, ハイパス, x'のch, y'のch)
        (True, False, 1, 2),      # フィルタなし: 素通しなのでch数は増えない
        (False, True, 30, 60),    # CFAマスクなし: ランダムRGB選択。ch数は既定と同じ
        (False, False, 1, 2),
    )
    for use_cfa, use_hp, expected_in, expected_target in cases:
        pre = DCCTPreprocessor(use_cfa_mask=use_cfa, use_high_pass=use_hp)
        x_prime, y_prime = pre.prepare(torch.rand(2, 3, 64, 64) * 255)
        assert pre.input_channels == expected_in
        assert x_prime.shape[1] == expected_in
        assert pre.target_channels == expected_target
        assert y_prime.shape[1] == expected_target


@pytest.mark.parametrize("filter_mode", ["mosaic", "sublattice", "per_color"])
def test_filter_modes_keep_channel_layout(filter_mode):
    """フィルタのかけ方を変えても、x' 30ch・y' 60ch の形と [-t,t] の範囲は変わらないこと。"""
    pre = DCCTPreprocessor(truncation_t=7.0, filter_mode=filter_mode)
    image = torch.rand(3, 3, 64, 64) * 255
    x_prime, y_prime = pre.prepare(image)
    assert x_prime.shape == (3, 30, 64, 64) and y_prime.shape == (3, 60, 64, 64)
    assert x_prime.abs().max() <= 7 and y_prime.abs().max() <= 7
    assert torch.equal(pre.prepare_input(image), x_prime)


def test_mosaic_is_the_default_filter_mode():
    """filter_mode を指定しなければ、従来どおりモザイクにそのままフィルタをかける。"""
    image = torch.rand(2, 3, 64, 64) * 255
    default = DCCTPreprocessor()
    x, y = default.mask.apply(image)
    x_prime, y_prime = default.prepare(image)
    assert default.filter_mode == "mosaic"
    assert torch.equal(x_prime, default.bank.apply(x)) and torch.equal(y_prime, default.bank.apply(y))


def test_sublattice_filters_only_same_color_neighbors():
    """同じ色どうし: 2×2 の各位置の格子ごとにフィルタをかけるので、別の格子の値は影響しない。"""
    pre = DCCTPreprocessor(filter_mode="sublattice")
    image = torch.rand(1, 3, 64, 64) * 255
    x_prime, _ = pre.prepare(image)

    changed = image.clone()
    changed[:, :, 1::2, :] += 50          # 奇数行（別の格子）だけ変える
    x_changed, _ = pre.prepare(changed)
    assert torch.equal(x_prime[:, :, 0::2, :], x_changed[:, :, 0::2, :])

    x, _ = pre.mask.apply(image)
    assert torch.equal(x_prime[:, :, 0::2, 0::2], pre.bank.apply(x[:, :, 0::2, 0::2]))


def test_per_color_selects_residual_of_observed_color_at_each_bayer_site():
    """色ごと→選ぶ: R・G・B それぞれの残差から、Bayer の位置で観測色と残り2色を選ぶ。"""
    pre = DCCTPreprocessor(filter_mode="per_color", bayer_pattern="RGGB")
    image = torch.rand(2, 3, 64, 64) * 255
    x_prime, y_prime = pre.prepare(image)
    full = pre.bank.apply(image).reshape(2, 3, 30, 64, 64)       # 色ごとの30種

    # RGGB: (0,0)=R → y=(G,B)、(0,1)=G → y=(R,B)、(1,1)=B → y=(R,G)
    assert torch.equal(x_prime[:, :, 0, 0], full[:, 0, :, 0, 0])
    assert torch.equal(y_prime[:, :30, 0, 0], full[:, 1, :, 0, 0])
    assert torch.equal(y_prime[:, 30:, 0, 0], full[:, 2, :, 0, 0])
    assert torch.equal(x_prime[:, :, 0, 1], full[:, 1, :, 0, 1])
    assert torch.equal(y_prime[:, :30, 0, 1], full[:, 0, :, 0, 1])
    assert torch.equal(x_prime[:, :, 1, 1], full[:, 2, :, 1, 1])
    assert torch.equal(y_prime[:, 30:, 1, 1], full[:, 1, :, 1, 1])


def test_filter_modes_other_than_mosaic_need_bayer_mask_and_filter_bank():
    with pytest.raises(ValueError, match="filter_mode"):
        DCCTPreprocessor(filter_mode="sublattice", use_cfa_mask=False)
    with pytest.raises(ValueError, match="filter_mode"):
        DCCTPreprocessor(filter_mode="per_color", target_mode="single_filter")
    with pytest.raises(ValueError, match="filter_mode"):
        DCCTPreprocessor(filter_mode="unknown")


def test_test_patch_positions_are_stable_across_processes():
    """同じ画像なら別プロセスで実行しても同じパッチが得られる（NFR-1）。

    組み込みの `hash()` は文字列に対してプロセスごとにランダム化されるため、
    シードに使うと実行のたびに評価結果が変わってしまう。
    """
    import subprocess
    import sys

    script = (
        "import torch;"
        "from src.preprocess.patch_sampler import PatchSampler;"
        "torch.manual_seed(0);"
        "img = torch.arange(3 * 128 * 128, dtype=torch.float32).reshape(3, 128, 128);"
        "p = PatchSampler(patch_size=64, jpeg_enabled=False, seed=42)"
        ".sample_test(img, num_patches=4, image_key='some/file.png');"
        "print(int(p.sum().item()))"
    )

    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": os.environ["PATH"]},
        )
        outputs.add(result.stdout.strip())

    assert len(outputs) == 1, f"PYTHONHASHSEED でパッチ位置が変わっている: {outputs}"

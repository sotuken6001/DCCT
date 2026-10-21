"""DataLoaderと学習ループ（Stage I-A/I-B, Stage II）のテスト。"""

from __future__ import annotations

import io
import sqlite3
import zipfile

import pytest
import torch
import yaml
from PIL import Image

from src.data import db as db_module
from src.data.dataset import (
    MultiPatchDataset,
    PatchDataset,
    select_image_rows,
    skip_invalid_collate,
)
from src.data.ingest import ingest
from src.data.repository import MetadataRepository
from src.experiment import builders
from src.experiment.config import Config
from src.train.train_stage1 import TrainerStage1
from src.train.train_stage2 import TrainerStage2


def _png(width=96, height=96, color=(120, 80, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def project(tmp_path):
    """GenImage風のZIPを1つ作り、ingest済みのDBとconfigを用意する。"""
    root = tmp_path / "Extreme Pro"
    root.mkdir()
    with zipfile.ZipFile(root / "genimage-stable-diffusion-v1-4.zip", "w") as zf:
        for i in range(12):
            zf.writestr(f"train/ai/ai_{i:03d}.png", _png(color=(200, 50, 50)))
            zf.writestr(f"train/nature/nat_{i:03d}.png", _png(color=(50, 200, 50)))
        for i in range(4):
            zf.writestr(f"val/ai/ai_{i:03d}.png", _png(color=(200, 50, 50)))
            zf.writestr(f"val/nature/nat_{i:03d}.png", _png(color=(50, 200, 50)))

    data = Config.load("configs/base.yaml").as_dict()
    data["paths"]["dataset_root"] = str(root)
    data["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    data["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"}
    ]
    # 小さなフィクスチャでも val split が空にならないようにする
    data["dataset"]["split"]["val_ratio_from_train"] = 0.5
    data["augmentation"]["jpeg"]["enabled"] = False
    data["model"]["conditional_unet"]["base_channels"] = 8
    data["train"].update(
        {"batch_size": 2, "num_workers": 0, "num_epochs": 1, "max_eval_steps": 1, "save_every_epoch": 1}
    )

    db_module.init_db(data["paths"]["db_path"])
    path = tmp_path / "test_base.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    ingest(Config.load(path), progress=False)
    return tmp_path, data


def _write_config(tmp_path, data, name, **overrides):
    merged = yaml.safe_load(yaml.safe_dump(data))
    for key, value in overrides.items():
        merged[key] = {**merged.get(key, {}), **value} if isinstance(value, dict) else value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(merged, allow_unicode=True), encoding="utf-8")
    return Config.load(path)


# ------------------------------------------------------------------ Dataset


def test_select_rows_picks_train_generator_and_its_real_subset(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    with MetadataRepository(config.get("paths.db_path")) as repo:
        ai_rows = select_image_rows(config, repo, label="ai", split="train")
        real_rows = select_image_rows(config, repo, label="real", split="train")
        both = select_image_rows(config, repo, label=None, split="train")

    assert {r["generator"] for r in ai_rows} == {"SDv1.4"}
    assert {r["generator"] for r in real_rows} == {"ImageNet(real)@SDv1.4"}
    assert len(both) == len(ai_rows) + len(real_rows)


def test_excluded_generators_are_not_selected(project):
    """【OI-1】excluded_generators に入れた生成器は学習対象から外れること。"""
    tmp_path, data = project
    config = _write_config(tmp_path, data, "excl.yaml", dataset={**data["dataset"], "excluded_generators": ["SDv1.4"]})
    with MetadataRepository(config.get("paths.db_path")) as repo:
        assert select_image_rows(config, repo, label="ai", split="train") == []


def test_patch_dataset_returns_patch_and_label(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    sampler = builders.build_patch_sampler(config)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label=None, split="train")

    dataset = PatchDataset(rows, sampler)
    patch, target = dataset[0]
    assert patch.shape == (3, 64, 64)
    assert patch.max() > 1.0                      # 0〜255スケールのまま
    assert target.item() in (0.0, 1.0)

    labels = {dataset[i][1].item() for i in range(len(dataset))}
    assert labels == {0.0, 1.0}                   # 写真とAI画像の両方が含まれる


def test_patch_dataset_passes_each_label_through_its_own_transform(project):
    """実験17: AI画像は ai_transform、本物は real_transform だけを通す（本物も PIL の JPEG にかけ直す）。"""
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    sampler = builders.build_patch_sampler(config)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label=None, split="train")

    seen: list[str] = []

    def mark(name):
        def transform(image):
            seen.append(name)
            return image
        return transform

    dataset = PatchDataset(rows, sampler, ai_transform=mark("ai"), real_transform=mark("real"))
    for i in range(len(dataset)):
        dataset[i]

    assert seen.count("ai") == sum(row["label"] == "ai" for row in rows)
    assert seen.count("real") == sum(row["label"] != "ai" for row in rows)


def test_real_train_transform_follows_config():
    data = Config.load("configs/base.yaml").as_dict()
    assert builders.build_real_train_transform(Config(data)) is None          # 既定は何もしない
    data["augmentation"]["real_jpeg"] = {"enabled": True, "quality_min": 70, "quality_max": 95}
    transform = builders.build_real_train_transform(Config(data))
    assert (transform.quality_min, transform.quality_max) == (70, 95)


def test_patch_dataset_rejects_empty_rows():
    with pytest.raises(ValueError, match="0件"):
        PatchDataset([], builders.build_patch_sampler(Config.load("configs/base.yaml")))


def test_multipatch_dataset_returns_p_patches(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label=None, split="test")

    dataset = MultiPatchDataset(rows, builders.build_patch_sampler(config), num_patches=16)
    patches, target, image_id = dataset[0]
    assert patches.shape == (16, 3, 64, 64)
    assert target.item() in (0.0, 1.0)
    assert isinstance(image_id, int)
    # 同じ画像なら毎回同じパッチ位置（NFR-1）
    assert torch.equal(dataset[0][0], dataset[0][0])


def test_multipatch_dataset_skips_too_small_images(project, tmp_path_factory):
    """64px未満の画像は例外ではなく None を返し、評価対象から外れる。

    ディレクトリ形式のソースは ingest 時に画像ヘッダを読まないため、
    `is_croppable=1` のままDBに残った小さい画像が評価に流れ込むことがある。
    """
    from PIL import Image

    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label=None, split="test")

    tiny_path = tmp_path_factory.mktemp("tiny") / "tiny.png"
    Image.new("RGB", (56, 74)).save(tiny_path)
    tiny_row = dict(rows[0])
    tiny_row["filepath"] = str(tiny_path)

    dataset = MultiPatchDataset([tiny_row, *rows], builders.build_patch_sampler(config))
    assert dataset[0] is None            # 小さすぎる画像
    assert dataset[1] is not None        # 後続はそのまま使える

    # collate は None を取り除き、全件Noneならバッチごと None を返す
    batch = skip_invalid_collate([dataset[0], dataset[1]])
    assert batch is not None
    assert batch[0].shape[0] == 1        # 有効な1件だけが残る
    assert skip_invalid_collate([None, None]) is None


def test_dataset_reads_images_from_inside_zip(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label="ai", split="train")
    assert all(".zip!" in row["filepath"] for row in rows)
    assert PatchDataset(rows, builders.build_patch_sampler(config))[0][0].shape == (3, 64, 64)


# ------------------------------------------------------------- Stage I 学習


def test_stage1_trains_and_records_to_db(project):
    tmp_path, data = project
    config = _write_config(
        tmp_path, data, "s1.yaml",
        experiment={"name": "t_stage1", "stage": "stage1_photo"},
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s1"), "num_epochs": 2},
    )

    with TrainerStage1(config, "photo") as trainer:
        run_id = trainer.fit(max_steps_per_epoch=2)
        repo = trainer.repo
        run = repo.conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
        logs = repo.conn.execute(
            "SELECT DISTINCT metric_name FROM training_logs WHERE run_id = ?", (run_id,)
        ).fetchall()
        checkpoints = repo.conn.execute(
            "SELECT * FROM model_checkpoints WHERE run_id = ?", (run_id,)
        ).fetchall()
        experiment = repo.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (trainer.experiment_id,)
        ).fetchone()

    assert run["status"] == "completed" and run["completed_at"]
    assert {"nll_loss", "val_nll", "lr"} <= {row["metric_name"] for row in logs}
    assert len(checkpoints) >= 2
    assert experiment["stage"] == "stage1_photo"
    assert experiment["seed"] == 42
    assert experiment["config_snapshot"]                      # NFR-7 トレーサビリティ
    assert (tmp_path / "ckpt_s1" / "best.pt").exists()


def test_stage1_photo_and_ai_use_different_data(project):
    tmp_path, data = project
    photo = _write_config(tmp_path, data, "s1p.yaml",
                          train={**data["train"], "checkpoint_dir": str(tmp_path / "c1")})
    ai = _write_config(tmp_path, data, "s1a.yaml",
                       train={**data["train"], "checkpoint_dir": str(tmp_path / "c2")})

    with TrainerStage1(photo, "photo") as p_trainer, TrainerStage1(ai, "ai") as a_trainer:
        assert p_trainer.stage == "stage1_photo" and a_trainer.stage == "stage1_ai"
        photo_paths = {row["filepath"] for row in p_trainer.train_loader.dataset.rows}
        ai_paths = {row["filepath"] for row in a_trainer.train_loader.dataset.rows}
    assert photo_paths and ai_paths and not (photo_paths & ai_paths)


def test_stage1_resume_continues_from_checkpoint(project):
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s1r.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_r")})

    with TrainerStage1(config, "photo") as trainer:
        trainer.fit(num_epochs=1, max_steps_per_epoch=1)
        checkpoint = tmp_path / "ckpt_r" / "best.pt"

    with TrainerStage1(config, "photo") as resumed:
        resumed.load_checkpoint(checkpoint, resume=True)
        assert resumed.start_epoch == 1
        resumed.fit(num_epochs=2, max_steps_per_epoch=1)
        assert resumed.start_epoch == 1                        # 2エポック目のみ実行された


def test_resume_learning_rate_overrides_restored_optimizer(project):
    """再開時は optimizer の状態ごと lr も戻るので、`train.resume_learning_rate` で上書きできること。

    これが無いと、config の learning_rate を下げても前回の lr のまま学習が続いてしまう。
    """
    tmp_path, data = project
    first = _write_config(tmp_path, data, "s1lr.yaml",
                          train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_lr")})
    with TrainerStage1(first, "photo") as trainer:
        trainer.fit(num_epochs=1, max_steps_per_epoch=1)
    checkpoint = tmp_path / "ckpt_lr" / "best.pt"

    lowered = _write_config(tmp_path, data, "s1lr2.yaml",
                            train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_lr2"),
                                   "resume_learning_rate": 1.0e-5})
    with TrainerStage1(lowered, "photo") as resumed:
        resumed.load_checkpoint(checkpoint, resume=True)
        assert [g["lr"] for g in resumed.optimizer.param_groups] == [1.0e-5]
        run_id = resumed.fit(num_epochs=2, max_steps_per_epoch=1)
        logged = resumed.repo.conn.execute(
            "SELECT metric_value FROM training_logs WHERE run_id = ? AND metric_name = 'lr'", (run_id,)
        ).fetchall()
    assert logged and all(row["metric_value"] == pytest.approx(1.0e-5) for row in logged)

    # 未設定なら従来どおり、保存されていた lr がそのまま復元される
    with TrainerStage1(first, "photo") as kept:
        kept.load_checkpoint(checkpoint, resume=True)
        assert [g["lr"] for g in kept.optimizer.param_groups] == [1.0e-4]


def test_failed_run_is_marked_failed(project, monkeypatch):
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s1f.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_f")})

    with TrainerStage1(config, "photo") as trainer:
        monkeypatch.setattr(trainer, "train_step", lambda batch: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            trainer.fit(num_epochs=1)
        status = trainer.repo.conn.execute(
            "SELECT status FROM training_runs WHERE run_id = ?", (trainer.run_id,)
        ).fetchone()["status"]
    assert status == "failed"


# ------------------------------------------------------------ Stage II 学習


def _stage1_checkpoints(tmp_path, data):
    """Stage II のテスト用に pθ / qφ のチェックポイントを作る。"""
    paths = {}
    for target, directory in (("photo", "ck_p"), ("ai", "ck_a")):
        config = _write_config(tmp_path, data, f"s1_{target}.yaml",
                               train={**data["train"], "checkpoint_dir": str(tmp_path / directory)})
        with TrainerStage1(config, target) as trainer:
            trainer.fit(num_epochs=1, max_steps_per_epoch=1)
        paths[target] = str(tmp_path / directory / "best.pt")
    return paths


def test_stage2_trains_classifier_and_keeps_models_frozen(project):
    tmp_path, data = project
    checkpoints = _stage1_checkpoints(tmp_path, data)
    config = _write_config(
        tmp_path, data, "s2.yaml",
        experiment={"name": "t_stage2", "stage": "stage2_classifier"},
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s2")},
        stage2={
            "photo_model_checkpoint": checkpoints["photo"],
            "ai_model_checkpoint": checkpoints["ai"],
            "use_photo_model": True,
            "use_ai_model": True,
            "freeze_conditional_models": True,
        },
    )

    with TrainerStage2(config) as trainer:
        assert trainer.classifier.in_channels == 60           # 論文準拠: 3K=30ch × 2モデル
        before_photo = trainer.photo_model.head.weight.clone()
        before_classifier = trainer.classifier.head.weight.clone()

        run_id = trainer.fit(num_epochs=1, max_steps_per_epoch=2)

        # Frozen: 条件付きモデルは更新されず、分類器だけが更新される
        assert torch.equal(trainer.photo_model.head.weight, before_photo)
        assert not torch.equal(trainer.classifier.head.weight, before_classifier)
        assert set(trainer.trainable_modules()) == {"classifier"}

        metrics = {
            row["metric_name"]
            for row in trainer.repo.conn.execute(
                "SELECT DISTINCT metric_name FROM training_logs WHERE run_id = ?", (run_id,)
            )
        }
    assert {"bce_loss", "accuracy", "val_bce", "val_accuracy"} <= metrics


def test_stage2_ablation_a_single_model(project):
    """Ablation-A: qφのみを使う設定では入力が60chになること。"""
    tmp_path, data = project
    checkpoints = _stage1_checkpoints(tmp_path, data)
    config = _write_config(
        tmp_path, data, "s2_a.yaml",
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s2a")},
        stage2={
            "ai_model_checkpoint": checkpoints["ai"],
            "use_photo_model": False,
            "use_ai_model": True,
            "freeze_conditional_models": True,
        },
    )

    with TrainerStage2(config) as trainer:
        assert trainer.photo_model is None
        assert trainer.classifier.in_channels == 30
        trainer.fit(num_epochs=1, max_steps_per_epoch=1)


def test_stage2_ablation_d_full_finetuning(project):
    """Ablation-D: freeze=False なら条件付きモデルも更新され、保存対象に含まれること。"""
    tmp_path, data = project
    checkpoints = _stage1_checkpoints(tmp_path, data)
    config = _write_config(
        tmp_path, data, "s2_d.yaml",
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s2d")},
        stage2={
            "photo_model_checkpoint": checkpoints["photo"],
            "ai_model_checkpoint": checkpoints["ai"],
            "use_photo_model": True,
            "use_ai_model": True,
            "freeze_conditional_models": False,
        },
    )

    with TrainerStage2(config) as trainer:
        before = trainer.photo_model.head.weight.clone()
        assert set(trainer.trainable_modules()) == {"classifier", "photo_model", "ai_model"}
        trainer.fit(num_epochs=1, max_steps_per_epoch=2)
        assert not torch.equal(trainer.photo_model.head.weight, before)

    saved = torch.load(tmp_path / "ckpt_s2d" / "best.pt", weights_only=False)
    assert {"classifier", "photo_model", "ai_model"} <= set(saved["modules"])


def test_db_logging_failure_does_not_kill_training(project, monkeypatch):
    """指標のDB記録に失敗しても、学習とチェックポイント保存は続くこと。

    pθとqφを並列に流すとSQLiteの書き込みが競合しうる。数時間の学習が
    記録の失敗で落ちないことを保証する。
    """
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s1_dbfail.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_dbfail")})

    with TrainerStage1(config, "photo") as trainer:
        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(trainer.repo, "log_training_metric", boom)
        monkeypatch.setattr(trainer.repo, "save_checkpoint_meta", boom)

        run_id = trainer.fit(num_epochs=1, max_steps_per_epoch=1)
        status = trainer.repo.conn.execute(
            "SELECT status FROM training_runs WHERE run_id = ?", (run_id,)
        ).fetchone()["status"]

    # 学習は完走し、チェックポイントのファイルも残っている
    assert status == "completed"
    assert (tmp_path / "ckpt_dbfail" / "best.pt").exists()


def test_connection_sets_busy_timeout(tmp_path):
    """並列実行時にロック待ちで即座に失敗しないこと。"""
    path = tmp_path / "busy.sqlite3"
    db_module.init_db(path)
    conn = db_module.connect(path)
    timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    conn.close()
    assert timeout >= 30_000


def test_old_checkpoints_are_pruned(project):
    """古いエポックのチェックポイントが整理され、best.pt は残ること。

    全エポック分を残すとディスクを圧迫し、実機では学習中にディスクが埋まって
    torch.save が失敗した。
    """
    tmp_path, data = project
    config = _write_config(
        tmp_path, data, "s1_prune.yaml",
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_prune"),
               "keep_last_checkpoints": 2},
    )

    with TrainerStage1(config, "photo") as trainer:
        trainer.fit(num_epochs=5, max_steps_per_epoch=1)

    directory = tmp_path / "ckpt_prune"
    epoch_files = sorted(p.name for p in directory.glob("run*_epoch*.pt"))
    assert len(epoch_files) == 2                  # 直近2エポックのみ
    assert epoch_files == sorted(epoch_files)[-2:]
    assert (directory / "best.pt").exists()       # best.pt は常に残る
    assert not list(directory.glob("*.tmp"))      # 一時ファイルが残らない


def test_checkpoint_save_failure_does_not_kill_training(project, monkeypatch):
    """保存に失敗しても（ディスク不足など）学習は継続すること。"""
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s1_savefail.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_savefail")})

    with TrainerStage1(config, "photo") as trainer:
        def no_space(*args, **kwargs):
            raise RuntimeError("[enforce fail at inline_container.cc:672] unexpected pos")

        monkeypatch.setattr(torch, "save", no_space)
        run_id = trainer.fit(num_epochs=2, max_steps_per_epoch=1)
        status = trainer.repo.conn.execute(
            "SELECT status FROM training_runs WHERE run_id = ?", (run_id,)
        ).fetchone()["status"]

    assert status == "completed"
    # 書きかけの一時ファイルは残さない
    assert not list((tmp_path / "ckpt_savefail").glob("*.tmp"))


def test_eval_rows_are_shuffled_so_both_labels_appear(project):
    """検証の先頭Nバッチが片方のラベルに偏らないこと。

    ingestは train/ai → train/nature の順に走査するため、DBのimage_id順のままだと
    val splitの先頭はすべて ai になる。max_eval_steps で先頭だけを使うと
    val_accuracy が「AI画像の再現率」になってしまい、実写の誤検出率が測れない。
    """
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s2_evalrows.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_evalrows")})

    with TrainerStage2(config) as trainer:
        # DBの並び順を模した、片方のラベルに固まった行
        ordered = [{"image_id": i, "label": "ai"} for i in range(50)]
        ordered += [{"image_id": 50 + i, "label": "real"} for i in range(50)]

        shuffled = trainer.prepare_eval_rows(ordered)

        # 先頭20件（= 先頭数バッチ相当）に両方のラベルが現れる
        assert set(row["label"] for row in shuffled[:20]) == {"ai", "real"}
        # 元の並びは ai に偏っていた（修正前の状態）
        assert set(row["label"] for row in ordered[:20]) == {"ai"}
        # 決定的であること（同じseedなら毎回同じ並び）
        assert [r["image_id"] for r in trainer.prepare_eval_rows(ordered)] == [
            r["image_id"] for r in shuffled
        ]


# ------------------------------------------- AI画像だけ JPEG にして学習する（実験15）


def test_random_jpeg_picks_quality_within_range():
    from src.preprocess.patch_sampler import RandomJpeg

    transform = RandomJpeg(70, 95)
    qualities = {transform.pick_quality() for _ in range(300)}
    assert qualities <= set(range(70, 96))
    assert min(qualities) < 75 and max(qualities) > 90      # 範囲の両端まで使う

    image = torch.rand(3, 64, 64) * 255
    out = transform(image)
    assert out.shape == image.shape and not torch.equal(out, image)


def test_patch_dataset_applies_ai_transform_only_to_ai_rows(project):
    tmp_path, data = project
    config = Config.load(tmp_path / "test_base.yaml")
    sampler = builders.build_patch_sampler(config)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = select_image_rows(config, repo, label=None, split="train")

    calls: list[int] = []

    def transform(image):
        calls.append(1)
        return image

    dataset = PatchDataset(rows, sampler, ai_transform=transform)
    for i in range(len(dataset)):
        dataset[i]
    assert len(calls) == sum(row["label"] == "ai" for row in rows) > 0


def test_ai_train_transform_is_built_from_config():
    from src.preprocess.patch_sampler import RandomJpeg

    data = Config.load("configs/base.yaml").as_dict()
    assert builders.build_ai_train_transform(Config(data)) is None          # 既定では何もしない

    data["augmentation"]["ai_jpeg"] = {"enabled": True, "quality_min": 70, "quality_max": 95}
    transform = builders.build_ai_train_transform(Config(data))
    assert isinstance(transform, RandomJpeg)
    assert (transform.quality_min, transform.quality_max) == (70, 95)

    data["augmentation"]["ai_jpeg"] = {"enabled": True, "quality_min": 95, "quality_max": 70}
    with pytest.raises(ValueError):
        builders.build_ai_train_transform(Config(data))


def test_stage2_uses_ai_jpeg_for_train_and_val(project, monkeypatch):
    tmp_path, data = project
    checkpoints = _stage1_checkpoints(tmp_path, data)
    config = _write_config(
        tmp_path, data, "s2_aijpeg.yaml",
        experiment={"name": "t_stage2_aijpeg", "stage": "stage2_classifier"},
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s2_aijpeg")},
        augmentation={**data["augmentation"], "ai_jpeg": {"enabled": True, "quality_min": 70, "quality_max": 95}},
        stage2={
            "photo_model_checkpoint": checkpoints["photo"],
            "ai_model_checkpoint": checkpoints["ai"],
            "use_photo_model": True,
            "use_ai_model": True,
            "freeze_conditional_models": True,
        },
    )
    with TrainerStage2(config) as trainer:
        for loader in (trainer.train_loader, trainer.val_loader):
            assert loader.dataset.ai_transform is not None
        trainer.fit(num_epochs=1, max_steps_per_epoch=1)


def test_stage1_ai_uses_ai_jpeg_for_train_and_val(project):
    """qφ の学習でも AI画像だけ JPEG にできる（実験19：形式をそろえて qφ を学習し直す）。"""
    tmp_path, data = project
    config = _write_config(
        tmp_path, data, "s1a_aijpeg.yaml",
        train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s1a_aijpeg")},
        augmentation={**data["augmentation"], "ai_jpeg": {"enabled": True, "quality_min": 70, "quality_max": 95}},
    )
    with TrainerStage1(config, "ai") as trainer:
        for loader in (trainer.train_loader, trainer.val_loader):
            assert loader.dataset.ai_transform is not None
        trainer.fit(num_epochs=1, max_steps_per_epoch=1)


def test_stage1_without_ai_jpeg_keeps_images_as_is(project):
    tmp_path, data = project
    config = _write_config(tmp_path, data, "s1a_plain.yaml",
                           train={**data["train"], "checkpoint_dir": str(tmp_path / "ckpt_s1a_plain")})
    with TrainerStage1(config, "ai") as trainer:
        assert trainer.train_loader.dataset.ai_transform is None
        assert trainer.val_loader.dataset.ai_transform is None


def test_random_jpeg_mixes_subsampling_by_weights():
    """実験サ：AI画像の JPEG の色の間引き方を、本物と同じ割合で混ぜる。"""
    from src.preprocess.patch_sampler import RandomJpeg

    torch.manual_seed(0)
    transform = RandomJpeg(96, 96, subsampling_weights={"4:4:4": 0.8, "4:2:0": 0.2})
    picks = [transform.pick_subsampling() for _ in range(2000)]
    assert set(picks) == {"4:4:4", "4:2:0"}
    assert 0.75 < picks.count("4:4:4") / len(picks) < 0.85

    assert RandomJpeg(96, 96).pick_subsampling() is None                   # 指定がなければ PIL の初期設定
    with pytest.raises(ValueError):
        RandomJpeg(96, 96, subsampling_weights={"4:1:1": 1.0})
    with pytest.raises(ValueError):
        RandomJpeg(96, 96, subsampling_weights={"4:4:4": 0.0})


def test_random_jpeg_applies_picked_subsampling(monkeypatch):
    import src.preprocess.patch_sampler as module

    seen = []
    monkeypatch.setattr(module, "jpeg_compress", lambda image, q, s=None: seen.append((q, s)) or image)
    transform = module.RandomJpeg(96, 96, subsampling_weights={"4:4:4": 1.0})
    transform(torch.zeros(3, 8, 8))
    assert seen == [(96, "4:4:4")]


def test_ai_train_transform_reads_subsampling_weights():
    data = Config.load("configs/base.yaml").as_dict()
    data["augmentation"]["ai_jpeg"] = {
        "enabled": True, "quality_min": 96, "quality_max": 96, "subsampling": {"4:4:4": 0.8, "4:2:0": 0.2},
    }
    transform = builders.build_ai_train_transform(Config(data))
    assert transform.subsampling_weights == {"4:4:4": 0.8, "4:2:0": 0.2}

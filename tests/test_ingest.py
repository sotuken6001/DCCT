"""データ取り込みバッチ（ingest）のテスト。

外部SSDの実データは使わず、GenImageと同じ構成の小さなZIP／ディレクトリを
一時領域に合成して検証する。
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.data import db as db_module
from src.data.ingest import ingest
from src.data.repository import MetadataRepository
from src.data.sources import ZIP_SEPARATOR, open_image_bytes
from src.experiment.config import Config


def _png_bytes(width: int = 128, height: int = 128) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 80, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def dataset_root(tmp_path):
    """GenImage風のZIP（SDv1.4相当）とディレクトリ（BigGAN相当）を作る。"""
    root = tmp_path / "Extreme Pro"
    root.mkdir()

    # --- ZIPソース: train/{ai,nature}, val/{ai,nature} ---
    zip_path = root / "genimage-stable-diffusion-v1-4.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(20):
            zf.writestr(f"train/ai/ai_{i:03d}.png", _png_bytes())
            zf.writestr(f"train/nature/nat_{i:03d}.png", _png_bytes())
        for i in range(6):
            zf.writestr(f"val/ai/ai_{i:03d}.png", _png_bytes())
            zf.writestr(f"val/nature/nat_{i:03d}.png", _png_bytes())
        # 破損画像（03_詳細設計書 6節: status=invalid として除外）
        zf.writestr("train/ai/broken.png", b"this is not a png")
        # 64x64未満（is_croppable=0）
        zf.writestr("train/ai/tiny.png", _png_bytes(32, 32))
        # 画像以外・macOSのリソースフォークは無視されること
        zf.writestr("train/ai/notes.txt", b"ignore me")
        zf.writestr("__MACOSX/train/ai/._ai_000.png", b"resource fork")

    # --- ディレクトリソース: 上位フォルダが1段挟まる構成 ---
    dir_root = root / "BigGAN" / "imagenet_ai_0419_biggan"
    for split in ("train", "val"):
        for cls, count in (("ai", 8), ("nature", 8)):
            d = dir_root / split / cls
            d.mkdir(parents=True)
            for i in range(count):
                (d / f"{cls}_{i:03d}.png").write_bytes(_png_bytes())

    return root


@pytest.fixture()
def config(tmp_path, dataset_root):
    base = Config.load("configs/base.yaml").as_dict()
    base["paths"]["dataset_root"] = str(dataset_root)
    base["paths"]["db_path"] = str(tmp_path / "test.sqlite3")
    base["dataset"]["sources"] = [
        {"name": "SDv1.4", "path": "genimage-stable-diffusion-v1-4.zip", "category": "diffusion", "type": "auto"},
        {"name": "BigGAN", "path": "BigGAN", "category": "gan", "type": "auto"},
        {"name": "Midjourney", "path": "", "category": "diffusion", "type": "auto"},
    ]
    path = tmp_path / "test_base.yaml"
    path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")

    db_module.init_db(base["paths"]["db_path"])
    return Config.load(path)


@pytest.fixture()
def ingested(config):
    stats = {s.generator: s for s in ingest(config, progress=False)}
    return config, stats


def test_scans_both_zip_and_directory(ingested):
    _, stats = ingested
    # ZIP: (20+20) train + (6+6) val + broken + tiny = 54。txt とリソースフォークは対象外
    assert stats["SDv1.4"].scanned == 54
    assert stats["SDv1.4"].inserted == 54
    # ディレクトリ: 上位フォルダが1段挟まっていても判定できること
    assert stats["BigGAN"].scanned == 32
    assert stats["BigGAN"].inserted == 32


def test_excluded_generator_is_skipped_but_registered(ingested):
    """【OI-1】Midjourneyは走査しないが、generatorsマスタには notes 付きで残ること。"""
    config, stats = ingested
    assert stats["Midjourney"].scanned == 0
    assert "除外" in stats["Midjourney"].skipped_reason

    with MetadataRepository(config.get("paths.db_path")) as repo:
        row = repo.conn.execute("SELECT * FROM generators WHERE name = 'Midjourney'").fetchone()
    assert row is not None
    assert "除外" in row["notes"]
    assert "未取得" in row["notes"]


def test_labels_and_generators(ingested):
    config, _ = ingested
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = repo.conn.execute(
            "SELECT g.name AS generator, i.label, COUNT(*) AS n "
            "FROM images i JOIN generators g USING(generator_id) GROUP BY g.name, i.label"
        ).fetchall()
    counts = {(r["generator"], r["label"]): r["n"] for r in rows}

    # ai/ は生成器そのものに、nature/ は per_source の実写生成器に紐づく
    assert counts[("SDv1.4", "ai")] == 28          # 20 + 6 + broken + tiny
    assert counts[("ImageNet(real)@SDv1.4", "real")] == 26
    assert counts[("BigGAN", "ai")] == 16
    assert counts[("ImageNet(real)@BigGAN", "real")] == 16
    assert ("SDv1.4", "real") not in counts


def test_split_assignment(ingested):
    """GenImageの val/ は test へ、train/ は一部が val へ回ること。"""
    config, _ = ingested
    with MetadataRepository(config.get("paths.db_path")) as repo:
        rows = repo.conn.execute("SELECT split, COUNT(*) AS n FROM images GROUP BY split").fetchall()
        by_split = {r["split"]: r["n"] for r in rows}

        # val/ 由来は必ず test（ZIPは '...zip!val/...'、ディレクトリは '.../val/...'）
        val_sourced = repo.conn.execute(
            "SELECT COUNT(*) AS n FROM images "
            "WHERE filepath LIKE '%val/ai/%' OR filepath LIKE '%val/nature/%'"
        ).fetchone()["n"]

    assert by_split.get("test") == val_sourced == 28  # ZIP 12 + ディレクトリ16
    assert by_split.get("train", 0) > 0
    assert by_split.get("train", 0) + by_split.get("val", 0) == 54 + 32 - 28


def test_broken_and_small_images_are_flagged(ingested):
    config, stats = ingested
    assert stats["SDv1.4"].invalid == 1
    assert stats["SDv1.4"].too_small == 1

    with MetadataRepository(config.get("paths.db_path")) as repo:
        broken = repo.conn.execute("SELECT * FROM images WHERE filepath LIKE '%broken.png'").fetchone()
        tiny = repo.conn.execute("SELECT * FROM images WHERE filepath LIKE '%tiny.png'").fetchone()
        normal = repo.conn.execute("SELECT * FROM images WHERE filepath LIKE '%ai_000.png' LIMIT 1").fetchone()
        usable = repo.list_images(label="ai", generator_names=["SDv1.4"])

    assert broken["status"] == "invalid" and broken["width"] is None
    assert tiny["status"] == "valid" and tiny["is_croppable"] == 0
    assert normal["width"] == 128 and normal["height"] == 128

    # 学習・評価対象からは破損画像と64px未満が自動で外れる
    paths = {r["filepath"] for r in usable}
    assert not any(p.endswith(("broken.png", "tiny.png")) for p in paths)
    assert len(usable) == 26


def test_zip_filepath_is_readable(ingested):
    """記録した filepath からZIP内画像を直接読めること（学習時の読み出し経路）。"""
    config, _ = ingested
    with MetadataRepository(config.get("paths.db_path")) as repo:
        row = repo.conn.execute(
            "SELECT filepath FROM images WHERE filepath LIKE '%.zip" + ZIP_SEPARATOR + "%' LIMIT 1"
        ).fetchone()

    data = open_image_bytes(row["filepath"])
    with Image.open(io.BytesIO(data)) as img:
        assert img.size == (128, 128)


def test_rerun_is_idempotent(ingested):
    """再実行しても重複登録されないこと（filepath UNIQUE）。"""
    config, _ = ingested
    second = {s.generator: s for s in ingest(config, progress=False)}
    assert second["SDv1.4"].inserted == 0
    assert second["SDv1.4"].already_registered == 54

    with MetadataRepository(config.get("paths.db_path")) as repo:
        total = repo.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
    assert total == 54 + 32


def test_split_assignment_is_deterministic(config):
    """同じseedなら再実行しても同じ画像が同じsplitに入ること（NFR-1）。"""
    ingest(config, progress=False)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        first = {r["filepath"]: r["split"] for r in repo.conn.execute("SELECT filepath, split FROM images")}
        repo.conn.execute("DELETE FROM images")
        repo.conn.commit()
    ingest(config, progress=False)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        second = {r["filepath"]: r["split"] for r in repo.conn.execute("SELECT filepath, split FROM images")}
    assert first == second


def test_limit_and_dry_run(config):
    stats = {s.generator: s for s in ingest(config, limit=5, dry_run=True, progress=False)}
    assert stats["SDv1.4"].scanned == 5
    assert stats["SDv1.4"].inserted == 0

    with MetadataRepository(config.get("paths.db_path")) as repo:
        assert repo.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == 0


def test_only_generators_filter(config):
    stats = {s.generator: s for s in ingest(config, only_generators=["BigGAN"], progress=False)}
    assert set(stats) == {"BigGAN"}
    assert stats["BigGAN"].inserted == 32


def test_missing_source_path_is_reported_not_fatal(config, tmp_path):
    """パスが存在しない生成器はスキップし、他の生成器の取り込みは続行すること。"""
    data = config.as_dict()
    data["dataset"]["sources"].append(
        {"name": "VQDM", "path": "does_not_exist", "category": "diffusion", "type": "auto"}
    )
    path = tmp_path / "with_missing.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    stats = {s.generator: s for s in ingest(Config.load(path), progress=False)}
    assert "見つかりません" in stats["VQDM"].skipped_reason
    assert stats["SDv1.4"].inserted == 54


# ------------------------------------------------- OS管理ファイルの扱い


def test_appledouble_files_are_not_scanned(tmp_path):
    """macOSが外部ディスクに作る `._xxx.png` を画像として数えないこと。

    これを数えると走査枚数がちょうど倍になり、中身が画像ではないため
    半分が「破損」に分類されてしまう。
    """
    root = tmp_path / "Extreme Pro"
    image_dir = root / "SDv15" / "train" / "ai"
    image_dir.mkdir(parents=True)
    for i in range(5):
        (image_dir / f"img_{i}.png").write_bytes(_png_bytes())
        # macOSが同じ階層に作るAppleDouble（Finderには見えない）
        (image_dir / f"._img_{i}.png").write_bytes(b"\x00\x05\x16\x07not an image")
    (image_dir / ".DS_Store").write_bytes(b"junk")
    macosx = root / "SDv15" / "__MACOSX" / "train" / "ai"
    macosx.mkdir(parents=True)
    (macosx / "img_0.png").write_bytes(b"junk")

    nature_dir = root / "SDv15" / "train" / "nature"
    nature_dir.mkdir(parents=True)
    for i in range(5):
        (nature_dir / f"nat_{i}.png").write_bytes(_png_bytes())

    base = Config.load("configs/base.yaml").as_dict()
    base["paths"]["dataset_root"] = str(root)
    base["paths"]["db_path"] = str(tmp_path / "junk.sqlite3")
    base["dataset"]["sources"] = [
        {"name": "SDv1.5", "path": "SDv15", "category": "diffusion", "type": "auto"}
    ]
    path = tmp_path / "junk.yaml"
    path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    db_module.init_db(base["paths"]["db_path"])

    stats = {s.generator: s for s in ingest(Config.load(path), progress=False)}
    assert stats["SDv1.5"].scanned == 10      # 本物の画像だけ
    assert stats["SDv1.5"].invalid == 0       # AppleDoubleを破損として数えない

    with MetadataRepository(base["paths"]["db_path"]) as repo:
        paths = [row["filepath"] for row in repo.conn.execute("SELECT filepath FROM images")]
    assert not any("._" in Path(p).name or "__MACOSX" in p for p in paths)


def test_cleanup_removes_already_registered_junk_rows(config):
    """修正前に登録されてしまったAppleDouble行を削除できること。"""
    from src.data.ingest import cleanup_junk

    ingest(config, progress=False)
    db_path = config.get("paths.db_path")

    with MetadataRepository(db_path) as repo:
        before = repo.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
        # 旧バージョンが登録していた形の行を再現する
        repo.register_images(
            [
                {
                    "dataset_id": 1,
                    "generator_id": 1,
                    "filepath": f"/Volumes/Extreme Pro/SDv15/train/ai/._img_{i}.png",
                    "label": "ai",
                    "split": "train",
                    "status": "invalid",
                    "is_croppable": 0,
                }
                for i in range(7)
            ]
        )
        assert repo.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == before + 7

    count, samples = cleanup_junk(config, dry_run=True)
    assert count == 7 and samples
    with MetadataRepository(db_path) as repo:   # dry-runでは消えない
        assert repo.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == before + 7

    count, _ = cleanup_junk(config)
    assert count == 7
    with MetadataRepository(db_path) as repo:
        assert repo.conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == before


def test_no_probe_skips_header_reading(config):
    """--no-probe ではwidth/heightを取らず、その分だけ速く登録できること。"""
    stats = {s.generator: s for s in ingest(config, progress=False, probe_header=False)}
    assert stats["SDv1.4"].scanned == 54
    assert stats["SDv1.4"].invalid == 0        # 破損判定は行わない

    with MetadataRepository(config.get("paths.db_path")) as repo:
        row = repo.conn.execute("SELECT width, height, status, is_croppable FROM images LIMIT 1").fetchone()
    assert row["width"] is None and row["height"] is None
    assert row["status"] == "valid" and row["is_croppable"] == 1


def test_directory_scan_is_lazy(tmp_path):
    """巨大なディレクトリでも、全件をリスト化せずに先頭から順に返すこと。

    GenImageは1ディレクトリに10万件以上入るため、ここで全件を材料化すると
    1件目を返すまでに長時間待たされる（外部SSDで顕著）。
    """
    from src.data.sources import DirectorySource

    image_dir = tmp_path / "gen" / "train" / "ai"
    image_dir.mkdir(parents=True)
    for i in range(50):
        (image_dir / f"img_{i:03d}.png").write_bytes(_png_bytes())

    entries = DirectorySource(tmp_path / "gen").iter_entries()
    first = next(entries)                      # 全件読まずに1件目が取れる
    assert first.relpath.startswith("train/ai/")
    assert first.size_bytes > 0

    assert len(list(entries)) == 49


def test_directory_scan_can_skip_size_lookup(tmp_path):
    """collect_size=False では stat を省くのでサイズは0になる。"""
    from src.data.sources import DirectorySource

    image_dir = tmp_path / "gen" / "train" / "ai"
    image_dir.mkdir(parents=True)
    (image_dir / "img.png").write_bytes(_png_bytes())

    entry = next(DirectorySource(tmp_path / "gen", collect_size=False).iter_entries())
    assert entry.size_bytes == 0
    assert entry.relpath == "train/ai/img.png"


def test_status_command_runs(config, capsys):
    """status が取り込み結果を集計して表示できること。"""
    from src.cli import main

    ingest(config, progress=False)
    assert main(["status", "--config", str(config.source_path)]) == 0

    out = capsys.readouterr().out
    assert "SDv1.4" in out
    assert "ImageNet(real)@SDv1.4" in out
    assert "合計" in out
    assert "Midjourney" in out            # 除外中の生成器として表示される


def test_probe_default_depends_on_source_type(config):
    """ZIPはヘッダを読み、ディレクトリは読まない（既定）。

    exFATの外部SSDでは巨大ディレクトリの1ファイルずつのstatが極端に遅くなるため、
    ディレクトリ形式では既定でヘッダ読み取りを行わない。
    """
    ingest(config, progress=False)

    with MetadataRepository(config.get("paths.db_path")) as repo:
        zip_row = repo.conn.execute(
            "SELECT width, height FROM images WHERE filepath LIKE '%.zip!%' LIMIT 1"
        ).fetchone()
        dir_row = repo.conn.execute(
            "SELECT width, height FROM images WHERE filepath NOT LIKE '%.zip!%' LIMIT 1"
        ).fetchone()

    assert zip_row["width"] == 128 and zip_row["height"] == 128   # ZIP: 読む
    assert dir_row["width"] is None and dir_row["height"] is None  # ディレクトリ: 読まない


def test_probe_can_be_forced_for_all_sources(config):
    """--probe 相当でディレクトリ側もヘッダを読むこと。"""
    ingest(config, progress=False, probe_header=True)

    with MetadataRepository(config.get("paths.db_path")) as repo:
        dir_row = repo.conn.execute(
            "SELECT width, height FROM images WHERE filepath NOT LIKE '%.zip!%' LIMIT 1"
        ).fetchone()
    assert dir_row["width"] == 128


def test_per_source_probe_setting_overrides_default(config, tmp_path):
    """ソースごとに probe を指定できること。"""
    data = config.as_dict()
    for source in data["dataset"]["sources"]:
        if source["name"] == "BigGAN":
            source["probe"] = True
    path = tmp_path / "per_source_probe.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    ingest(Config.load(path), progress=False)
    with MetadataRepository(config.get("paths.db_path")) as repo:
        dir_row = repo.conn.execute(
            "SELECT width FROM images WHERE filepath NOT LIKE '%.zip!%' LIMIT 1"
        ).fetchone()
    assert dir_row["width"] == 128

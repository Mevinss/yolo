"""Regression checks for annotation geometry and independent evaluation splits."""
from pathlib import Path

import pytest

from tools import build_dataset as legacy


def test_polygon_near_image_origin_is_not_a_degenerate_box():
    # This is a 0.3 x 0.3 polygon, not a box with width and height zero.
    assert not legacy.is_degenerate([0, 0, 0.3, 0, 0.3, 0.3, 0, 0.3])


def test_missing_sources_never_erase_existing_dataset(tmp_path, monkeypatch):
    out = tmp_path / "dataset"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("original")
    monkeypatch.setattr(legacy, "OUT_DIR", str(out))
    monkeypatch.setattr(legacy, "SOURCES", [(str(tmp_path / "missing"), {0: 0})])
    with pytest.raises(FileNotFoundError):
        legacy.build(force=True)
    assert sentinel.read_text() == "original"


def test_roboflow_variants_have_one_split():
    variants = [f"source_image.jpg.rf.{i:032x}" for i in range(40)]
    assert len({legacy.split_for(name) for name in variants}) == 1


def test_polygon_is_converted_to_enclosing_detection_box():
    from tools.prepare_dataset import detection_box
    assert detection_box([0, 0, 0.4, 0, 0.4, 0.2, 0, 0.2]) == pytest.approx(
        [0.2, 0.1, 0.4, 0.2])


@pytest.mark.parametrize("coords", [[float('nan'), .5, .1, .1],
                                    [.5, .5, -.1, .1], [0, 0, .2, .2, .3]])
def test_malformed_geometry_is_rejected(coords):
    from tools.prepare_dataset import detection_box
    with pytest.raises(ValueError):
        detection_box(coords)


def test_video_frames_and_augmentations_share_a_group():
    from tools.prepare_dataset import source_group
    a = source_group("sidewalk__May-21-2025_mp4-0001_jpg.rf.abc.jpg")
    b = source_group("sidewalk__May-21-2025_mp4-0999_jpg.rf.def.jpg")
    assert a == b
    assert a != source_group("sidewalk__May-26-2025_mp4-0001_jpg.rf.abc.jpg")


def test_preparation_preserves_input_and_has_no_group_overlap(tmp_path):
    from tools.prepare_dataset import prepare
    source, output = tmp_path / "input", tmp_path / "output"
    for split, name, content in [
        ("train", "pothole__same_jpg.rf.aaa", b"one"),
        ("test", "pothole__same_jpg.rf.bbb", b"two"),
        ("valid", "pothole__other_jpg.rf.ccc", b"one"),
    ]:
        images, labels = source / split / "images", source / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / (name + ".jpg")).write_bytes(content)
        (labels / (name + ".txt")).write_text("0 0 0 .4 0 .4 .2 0 .2\n")
    report = prepare(source, output)
    assert sum(x["images"] for x in report["splits"].values()) == 3
    assert len(list(source.glob("*/images/*.jpg"))) == 3
    assert len({x["split"] for x in report["manifest"]}) == 1
    assert report["benchmark_ready"] is False
    assert all(len(p.read_text().split()) == 5 for p in output.glob("*/labels/*.txt"))
    with pytest.raises(FileExistsError):
        prepare(source, output)


def test_missing_label_is_not_silently_treated_as_background(tmp_path):
    from tools.prepare_dataset import prepare
    image_dir = tmp_path / "input/train/images"
    image_dir.mkdir(parents=True)
    (image_dir / "x.jpg").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "input", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_training_refuses_legacy_dataset_without_manifest(tmp_path):
    from tools.train_detector import validate_dataset
    data = tmp_path / "data.yaml"
    data.write_text("names: [pothole, obstacle, stairs]\n")
    with pytest.raises(ValueError, match="manifest"):
        validate_dataset(data)


@pytest.mark.parametrize("drift", ["yaml", "extra_image", "label", "relative_root", "nested_image"])
def test_training_rejects_data_changed_after_manifest(tmp_path, drift):
    from tools.prepare_dataset import prepare
    from tools.train_detector import validate_dataset
    source, out = tmp_path/"source", tmp_path/"output"
    (source/"train/images").mkdir(parents=True)
    (source/"train/labels").mkdir(parents=True)
    (source/"train/images/one.jpg").write_bytes(b"image")
    (source/"train/labels/one.txt").write_text("0 .5 .5 .2 .2\n")
    prepare(source, out)
    validate_dataset(out/"data.yaml", exploratory=True)
    if drift == "yaml":
        (out/"data.yaml").write_text("train: ../source/train/images\nval: valid/images\ntest: test/images\nnames: [pothole, obstacle, stairs]\n")
    elif drift == "extra_image":
        (out/"valid/images/copy.jpg").write_bytes(b"image")
    elif drift == "relative_root":
        data = out/"data.yaml"
        data.write_text("path: .\n"+data.read_text())
    elif drift == "nested_image":
        (out/"valid/images/nested").mkdir()
        (out/"valid/images/nested/copy.jpg").write_bytes(b"image")
    else:
        (out/"train/labels/one.txt").write_text("0 .5 .5 .9 .9\n")
    with pytest.raises(ValueError):
        validate_dataset(out/"data.yaml", exploratory=True)

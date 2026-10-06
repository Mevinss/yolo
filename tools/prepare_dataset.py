"""Prepare a separate detection dataset with group-disjoint splits.

Keeps original exports intact. Groups known video frames, Roboflow variants
and exact file duplicates; unknown scene relationships still need human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("pothole", "obstacle", "stairs")
SPLITS = ("train", "valid", "test")


def source_group(filename: str) -> str:
    stem = Path(filename).stem.split(".rf.")[0]
    source, sep, name = stem.partition("__")
    if not sep:
        source, name = "unknown", source
    # Roboflow frame exports; no adjacent frames across partitions.
    video = re.match(r"(.*?)(?:_mp4|_mov|_avi)[-_]", name, re.I)
    if video:
        name = "video:" + video.group(1)
    elif re.match(r"youtube-\d+", name, re.I):
        name = "video:youtube"  # conservative: original video identity unknown
    return source + "__" + name


def detection_box(coords) -> list[float]:
    v = [float(x) for x in coords]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in v):
        raise ValueError("Coordinates must be finite and normalized to [0, 1]")
    if len(v) == 4:
        x, y, w, h = v
        if w <= 0 or h <= 0:
            raise ValueError("Box must have positive area")
        # Clip tiny export-rounding errors, reject genuinely out-of-frame boxes.
        if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
            raise ValueError("Box extends beyond image boundaries")
        x0, y0, x1, y1 = max(0, x-w/2), max(0, y-h/2), min(1, x+w/2), min(1, y+h/2)
    elif len(v) >= 6 and len(v) % 2 == 0:
        xs, ys = v[::2], v[1::2]
        area = abs(sum(xs[i]*ys[(i+1) % len(xs)] - xs[(i+1) % len(xs)]*ys[i]
                       for i in range(len(xs)))) / 2
        if area <= 1e-12:
            raise ValueError("Polygon must have positive area")
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    else:
        raise ValueError("Expected xywh box or polygon with at least three points")
    return [(x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0]


def annotation_text(boxes) -> str:
    lines = [str(row[0])+" "+" ".join(f"{v:.10g}" for v in row[1:]) for row in boxes]
    return "\n".join(lines) + ("\n" if lines else "")


def prepare(source: Path, output: Path, seed: int = 42) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new version: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Input and output must be separate directories")
    records = []
    rejected = []
    for split in SPLITS:
        for image in sorted((source/split/"images").glob("*")):
            if image.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            label = source/split/"labels"/(image.stem+".txt")
            if not label.exists():
                raise FileNotFoundError(f"Missing annotation: {label}")
            boxes = []
            for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                cls = float(parts[0])
                if not cls.is_integer() or not 0 <= cls < len(NAMES):
                    raise ValueError(f"Invalid class at {label}:{number}")
                try:
                    box = detection_box(parts[1:])
                except ValueError as exc:
                    # Report invalid instances; do not silently change geometry.
                    rejected.append({"file": str(label.relative_to(source)),
                                     "line": number, "reason": str(exc)})
                    continue
                boxes.append([int(cls), *box])
            records.append({"image": image, "original_split": split,
                            "relative_path": image.relative_to(source).as_posix(),
                            "group": source_group(image.name), "boxes": boxes,
                            "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
    if not records:
        raise ValueError(f"No images found in {source}")
    # Union groups connected by identical images, even with different filenames.
    parents = {r["group"]: r["group"] for r in records}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    seen = {}
    for r in records:
        if r["sha256"] in seen:
            a, b = find(r["group"]), find(seen[r["sha256"]])
            parents[max(a, b)] = min(a, b)
        seen[r["sha256"]] = r["group"]
    grouped = defaultdict(list)
    for r in records:
        r["group"] = find(r["group"])
        grouped[r["group"]].append(r)
    by_source = defaultdict(list)
    for group in grouped:
        by_source[group.partition("__")[0]].append(group)
    assignment = {}
    for groups in by_source.values():
        groups.sort(key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        n = len(groups)
        n_test = max(1, round(n*.1)) if n >= 2 else 0
        n_val = max(1, round(n*.1)) if n >= 3 else 0
        for i, group in enumerate(groups):
            assignment[group] = "test" if i < n_test else "valid" if i < n_test+n_val else "train"
    stats = {s: {"images": 0, "groups": 0, "boxes": {n: 0 for n in NAMES}, "empty_labels": 0}
             for s in SPLITS}
    old_group_splits = defaultdict(set)
    manifest = []
    # Names may collide across old splits; refuse before writing anything.
    destinations = set()
    for r in records:
        split = assignment[r["group"]]
        dest = (split, r["image"].stem.casefold())
        if dest in destinations:
            raise ValueError(f"Duplicate output name: {r['image'].name}")
        destinations.add(dest)
        stats[split]["images"] += 1
        stats[split]["empty_labels"] += not bool(r["boxes"])
        for row in r["boxes"]:
            stats[split]["boxes"][NAMES[row[0]]] += 1
        old_group_splits[r["group"]].add(r["original_split"])
        manifest.append({k: r[k] for k in ("relative_path", "original_split", "group", "sha256")}
                        | {"split": split, "box_count": len(r["boxes"]),
                           "label_sha256": hashlib.sha256(annotation_text(r["boxes"]).encode()).hexdigest()})
    for split in assignment.values():
        stats[split]["groups"] += 1
    missing = {s: [name for name, count in v["boxes"].items() if not count] for s, v in stats.items()}
    report = {"schema_version": 2, "seed": seed, "source": str(source),
              "output": str(output), "classes": list(NAMES), "splits": stats,
              "source_groups": {s: len(gs) for s, gs in by_source.items()},
              "original_cross_split_groups": sum(len(s) > 1 for s in old_group_splits.values()),
              "missing_classes": missing, "rejected_annotations": rejected,
              "benchmark_ready": False,
              "limitations": ["Source licences and original scene identities require verification.",
                              "Grouping uses filenames and exact file hashes, not all visual similarities.",
                              "Historical test images are reused; a new external test set is still needed.",
                              "Do not reuse old model metrics for this split."],
              "manifest": manifest}
    if rejected:
        # Incomplete annotations would turn real objects into false background.
        raise ValueError(f"{len(rejected)} invalid annotations; first: {rejected[0]}")
    for split in SPLITS:
        (output/split/"images").mkdir(parents=True)
        (output/split/"labels").mkdir()
    for r in records:
        split = assignment[r["group"]]
        shutil.copy2(r["image"], output/split/"images"/r["image"].name)
        (output/split/"labels"/(r["image"].stem+".txt")).write_text(
            annotation_text(r["boxes"]), encoding="utf-8", newline="\n")
    # No machine-specific drive letter. Ultralytics resolves paths from this YAML.
    (output/"data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnc: 3\nnames:\n"
        + "".join(f"  - {n}\n" for n in NAMES), encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT/"data/datasets/kz_hazards")
    ap.add_argument("--output", type=Path, default=ROOT/"data/datasets/hazards_grouped_v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    report = prepare(args.source, args.output, args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

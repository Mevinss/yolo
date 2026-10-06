# -*- coding: utf-8 -*-
"""
Пересборка датасета опасностей из трёх исходных выгрузок Roboflow.

ЗАЧЕМ ЭТО ПОНАДОБИЛОСЬ
----------------------
Унаследованный datasets/combined непригоден: в нём 3258 изображений,
но всего 469 размеченных объектов, а 2520 из 2927 файлов разметки
обучающей части пусты. Исходные выгрузки при этом целы и содержат
6549 объектов — то есть при слиянии потеряно около 93 процентов
разметки.

Причина в индексах классов. Roboflow экспортирует выгрузки с фиктивным
нулевым классом, поэтому настоящий класс имеет индекс 1:

    pothole detection    nc=2, names ['-', 'Pothole...']        -> реальный класс 1
    sidewalk obstacle    nc=2, names ['0', 'Obstacles']         -> реальный класс 1
    stairs detection     nc=1, names ['Stairs']                 -> реальный класс 0

Скрипт слияния этого не учёл, из-за чего боксы либо получили неверный
индекс, либо были отброшены. Ошибка тихая: файлы разметки существуют,
просто пусты, и обучение проходит без единой жалобы — выдавая модель,
которая почти ничего не находит.

Отсюда и отсутствие обученных весов в исходном проекте.

ПРАВИЛЬНОЕ ОТОБРАЖЕНИЕ
----------------------
    pothole detection : 1 -> 0 (pothole)
    sidewalk obstacle : 1 -> 1 (obstacle)
    stairs detection  : 0 -> 2 (stairs)

РАЗБИЕНИЕ
---------
У stairs detection нет частей valid и test — они создаются здесь
детерминированным разбиением по имени файла, чтобы результат
повторялся при повторном запуске.

Запуск:
    python tools/build_dataset.py
    python tools/build_dataset.py --check     # только проверка, без копирования
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_dataset import detection_box, source_group

SOURCES = [
    # (папка, {исходный класс: целевой класс})
    (os.path.join("..", "Проект", "datasets", "pothole detection"), {1: 0}),
    (os.path.join("..", "Проект", "datasets", "sidewalk obstacle"), {1: 1}),
    (os.path.join("..", "Проект", "datasets", "stairs detection"), {0: 2}),
]

CLASS_NAMES = ["pothole", "obstacle", "stairs"]
OUT_DIR = os.path.join("data", "datasets", "kz_hazards")
SPLITS = ("train", "valid", "test")

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")

#: Минимальная площадь бокса в долях кадра.
#:
#: В выгрузке stairs detection 15.4 % боксов вырождены — ширина или
#: высота близка к нулю. Такие аннотации не несут информации, но входят
#: и в обучение, и в тестовую часть: модель на них не учится, а метрика
#: считает их пропущенными. Отбрасываются с явным подсчётом, потому что
#: молчаливая потеря разметки — это ровно та ошибка, из-за которой
#: пересобирается весь датасет.
MIN_BOX_AREA_FRAC = 1e-5


def split_for(name: str) -> str:
    """Детерминированное разбиение по хэшу имени: 80 / 10 / 10.

    Хэш, а не случайность: при повторном запуске разбиение обязано
    совпасть, иначе цифры в статье перестают воспроизводиться.
    """
    h = int(hashlib.sha1(source_group(name).encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "valid"
    return "test"


def read_labels(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append((int(float(parts[0])), parts[1:]))
            except ValueError:
                continue
    return rows


def is_degenerate(coords: list) -> bool:
    """Бокс нулевой площади: разметка есть, информации в ней нет."""
    try:
        _, _, w, h = detection_box(coords)
    except (IndexError, ValueError):
        return True
    return w * h < MIN_BOX_AREA_FRAC


def audit() -> dict:
    """Посчитать, что есть в источниках и в старом объединении."""
    report = {"sources": {}, "old_combined": {}}

    for src, mapping in SOURCES:
        name = os.path.basename(src)
        counts = defaultdict(int)
        images = 0
        for split in SPLITS:
            lbl_dir = os.path.join(src, split, "labels")
            img_dir = os.path.join(src, split, "images")
            if not os.path.isdir(lbl_dir):
                continue
            images += len([f for f in os.listdir(img_dir)
                           if f.lower().endswith(IMAGE_EXT)]) if os.path.isdir(img_dir) else 0
            for f in os.listdir(lbl_dir):
                if not f.endswith(".txt"):
                    continue
                for cls, _ in read_labels(os.path.join(lbl_dir, f)):
                    counts[cls] += 1
        report["sources"][name] = {
            "images": images,
            "boxes_by_source_class": dict(counts),
            "mapping": {str(k): CLASS_NAMES[v] for k, v in mapping.items()},
        }

    old = os.path.join("..", "Проект", "datasets", "combined")
    if os.path.isdir(old):
        counts = defaultdict(int)
        images = empty = total_lbl = 0
        for split in SPLITS:
            lbl_dir = os.path.join(old, split, "labels")
            img_dir = os.path.join(old, split, "images")
            if os.path.isdir(img_dir):
                images += len([f for f in os.listdir(img_dir)
                               if f.lower().endswith(IMAGE_EXT)])
            if not os.path.isdir(lbl_dir):
                continue
            for f in os.listdir(lbl_dir):
                if not f.endswith(".txt"):
                    continue
                total_lbl += 1
                rows = read_labels(os.path.join(lbl_dir, f))
                if not rows:
                    empty += 1
                for cls, _ in rows:
                    counts[cls] += 1
        report["old_combined"] = {
            "images": images, "label_files": total_lbl,
            "empty_label_files": empty,
            "boxes_by_class": dict(counts),
        }
    return report


def build(force: bool = False) -> dict:
    # Validate all inputs BEFORE considering replacement of an existing output.
    for src, _ in SOURCES:
        if not any(os.path.isdir(os.path.join(src, s, "images")) for s in SPLITS):
            raise FileNotFoundError(f"Dataset source missing: {src}")
    if os.path.exists(OUT_DIR):
        raise FileExistsError("Existing datasets are immutable; use tools/prepare_dataset.py with a new output directory")
    for split in SPLITS:
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

    stats = {s: {"images": 0, "boxes": defaultdict(int), "empty": 0} for s in SPLITS}
    dropped = defaultdict(int)
    degenerate = defaultdict(int)

    for src, mapping in SOURCES:
        src_name = os.path.basename(src).replace(" ", "_")
        has_splits = [s for s in SPLITS if os.path.isdir(os.path.join(src, s, "images"))]

        for split in has_splits:
            img_dir = os.path.join(src, split, "images")
            lbl_dir = os.path.join(src, split, "labels")

            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith(IMAGE_EXT):
                    continue
                stem = os.path.splitext(fname)[0]
                rows = read_labels(os.path.join(lbl_dir, stem + ".txt"))

                mapped = []
                for cls, coords in rows:
                    if cls in mapping and is_degenerate(coords):
                        degenerate[f"{src_name}:{CLASS_NAMES[mapping[cls]]}"] += 1
                    elif cls in mapping:
                        mapped.append((mapping[cls], [f"{x:.10g}" for x in detection_box(coords)]))
                    else:
                        # Класс, не попавший в отображение. Считаем и
                        # показываем: молча отброшенная разметка —
                        # ровно та ошибка, из-за которой пересобираем.
                        dropped[f"{src_name}:{cls}"] += 1

                # Если исходная выгрузка не имела valid/test, разбиваем сами
                target = split_for(f"{src_name}__{stem}")

                new_stem = f"{src_name}__{stem}"
                shutil.copy2(os.path.join(img_dir, fname),
                             os.path.join(OUT_DIR, target, "images", new_stem + os.path.splitext(fname)[1]))
                with open(os.path.join(OUT_DIR, target, "labels", new_stem + ".txt"),
                          "w", encoding="utf-8") as fh:
                    for cls, coords in mapped:
                        fh.write(f"{cls} " + " ".join(coords) + "\n")

                stats[target]["images"] += 1
                if not mapped:
                    stats[target]["empty"] += 1
                for cls, _ in mapped:
                    stats[target]["boxes"][cls] += 1

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("train: train/images\nval: valid/images\ntest: test/images\n")
        fh.write(f"nc: {len(CLASS_NAMES)}\nnames:\n")
        for n in CLASS_NAMES:
            fh.write(f"  - {n}\n")

    return {
        "splits": {s: {"images": v["images"],
                       "empty_labels": v["empty"],
                       "boxes": {CLASS_NAMES[c]: n for c, n in sorted(v["boxes"].items())}}
                   for s, v in stats.items()},
        "dropped_unmapped": dict(dropped),
        "dropped_degenerate": dict(degenerate),
        "data_yaml": yaml_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="только аудит источников")
    ap.add_argument("--force", action="store_true", help="пересобрать с нуля")
    args = ap.parse_args()

    rep = audit()
    print("=" * 72)
    print("  АУДИТ ИСХОДНЫХ ВЫГРУЗОК")
    print("=" * 72)
    total_src = 0
    for name, v in rep["sources"].items():
        boxes = sum(v["boxes_by_source_class"].values())
        total_src += boxes
        print(f"  {name:<24}{v['images']:>6} изобр{boxes:>7} боксов   "
              f"классы {v['boxes_by_source_class']} -> {v['mapping']}")
    print(f"  {'ИТОГО В ИСТОЧНИКАХ':<24}{'':>6}     {total_src:>7} боксов")

    if rep["old_combined"]:
        o = rep["old_combined"]
        old_boxes = sum(o["boxes_by_class"].values())
        print(f"\n  Унаследованный combined: {o['images']} изобр, {old_boxes} боксов, "
              f"{o['empty_label_files']} из {o['label_files']} разметок пусты")
        print(f"  ПОТЕРЯНО ПРИ СЛИЯНИИ: {total_src - old_boxes} боксов "
              f"({100.0 * (total_src - old_boxes) / max(total_src, 1):.0f} %)")

    if args.check:
        return 0

    print("\n" + "=" * 72)
    print("  ПЕРЕСБОРКА")
    print("=" * 72)
    out = build(force=args.force)
    total = 0
    for split, v in out["splits"].items():
        n = sum(v["boxes"].values())
        total += n
        print(f"  {split:<8}{v['images']:>6} изобр{n:>7} боксов   {v['boxes']}   "
              f"без разметки: {v['empty_labels']}")
    print(f"  {'ИТОГО':<8}{'':>6}     {total:>7} боксов")
    if out["dropped_unmapped"]:
        print(f"  отброшено вне отображения: {out['dropped_unmapped']}")
    if out["dropped_degenerate"]:
        n = sum(out["dropped_degenerate"].values())
        print(f"  отброшено вырожденных боксов: {n} {dict(out['dropped_degenerate'])}")
    print(f"\n  -> {out['data_yaml']}")

    os.makedirs(os.path.join("docs", "article"), exist_ok=True)
    with open(os.path.join("docs", "article", "dataset_rebuild.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"audit": rep, "rebuilt": out}, fh, ensure_ascii=False, indent=2)
    print("  -> docs/article/dataset_rebuild.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

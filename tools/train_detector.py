# -*- coding: utf-8 -*-
"""
Дообучение детектора опасностей на пересобранном датасете.

ПОЧЕМУ ДООБУЧЕНИЕ, А НЕ ОБУЧЕНИЕ С НУЛЯ
---------------------------------------
Яма, бордюр и лестница — это не новые сущности для сети, видевшей
миллионы изображений: у них те же края, тени и текстуры. Дообучение
с весов COCO сходится за десятки эпох вместо сотен и не требует
объёмов данных, которых у нас нет.

КЛАССОВЫЙ ДИСБАЛАНС
-------------------
В датасете 7381 препятствие, 1042 ямы и 351 лестница — отношение
примерно 21 : 3 : 1. Лестница при этом опаснее прочего: пропущенная
ступень вниз означает падение.

Дисбаланс НЕ выравнивается искусственным дублированием: это
завысило бы метрики на валидации, не улучшив реальную работу.
Вместо этого он честно указывается в отчёте, а качество по классам
приводится раздельно — средняя mAP по трём классам здесь мало
что говорит.

ДВА ДЕТЕКТОРА В СИСТЕМЕ
-----------------------
Модель COCO остаётся: она находит людей, машины, скамейки — то,
для чего у нас нет своей разметки. Дообученная добавляет ямы,
препятствия тротуара и лестницы. В конвейере они дополняют друг
друга, а не заменяют.

Запуск:
    python tools/train_detector.py --epochs 60
    python tools/train_detector.py --epochs 2 --name smoke   # проверка
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_YAML = str(Path(__file__).resolve().parents[1] / "data/datasets/hazards_grouped_v1/data.yaml")
RUNS_DIR = "runs"
CLASS_NAMES = ["pothole", "obstacle", "stairs"]


def validate_dataset(path, exploratory=False):
    """Require a reviewed split description before producing new metrics."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = path.with_name("manifest.json")
    if not manifest.is_file():
        raise ValueError("Dataset manifest missing; run tools/prepare_dataset.py first")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent
    base = Path(data.get("path") or root)
    if not base.is_absolute():
        raise ValueError("Relative YAML path roots are ambiguous; omit path or use the absolute dataset directory")
    if base.resolve() != root:
        raise ValueError("YAML root differs from the prepared dataset")
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        if not isinstance(data.get(key), str) or (base/data[key]).resolve() != (root/split/"images").resolve():
            raise ValueError(f"YAML {key} path differs from the prepared dataset")
        if any(p.is_dir() for folder in ("images", "labels") for p in (root/split/folder).iterdir()):
            raise ValueError(f"Unexpected nested directory in {split}; prepared datasets are flat")
        expected = {Path(r["relative_path"]).name for r in report["manifest"] if r["split"] == split}
        actual = {p.name for p in (root/split/"images").glob("*") if p.is_file()}
        expected_labels = {Path(n).stem+".txt" for n in expected}
        actual_labels = {p.name for p in (root/split/"labels").glob("*.txt")}
        if expected != actual or expected_labels != actual_labels:
            raise ValueError(f"Dataset inventory changed in {split}")
    names = data.get("names")
    if isinstance(names, dict):
        names = [names.get(i) for i in range(len(CLASS_NAMES))]
    if names != CLASS_NAMES:
        raise ValueError("YAML class order differs from the prepared dataset")
    groups = {}
    for row in report["manifest"]:
        name = Path(row["relative_path"]).name
        image = root/row["split"]/"images"/name
        label = root/row["split"]/"labels"/(Path(name).stem+".txt")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image changed after preparation: {image}")
        if hashlib.sha256(label.read_bytes()).hexdigest() != row.get("label_sha256"):
            raise ValueError(f"Annotation changed or old manifest without label hash: {label}")
        previous = groups.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError("Dataset contains a group in multiple splits")
    if report["classes"] != CLASS_NAMES:
        raise ValueError("Dataset classes do not match the training task")
    if not exploratory and (not report["benchmark_ready"] or any(report["missing_classes"].values())):
        raise ValueError("Dataset is not ready for publication evaluation: "
                         + str(report["missing_classes"])
                         + ". Collect independent sessions; --exploratory is for pilot training only.")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/yolov8s.pt")
    ap.add_argument("--data", default=DATA_YAML)
    ap.add_argument("--exploratory", action="store_true", help="pilot only; metrics are not publication-ready")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="kz_hazards")
    ap.add_argument("--patience", type=int, default=15)
    # На Windows воркеры загрузчика падают из-за особенностей spawn:
    # безопасное значение — ноль, загрузка идёт в основном процессе.
    ap.add_argument("--workers", type=int,
                    default=0 if sys.platform == "win32" else 8)
    ap.add_argument("--cache", default="ram",
                    help="ram | disk | none — компенсирует отсутствие воркеров")
    args = ap.parse_args()

    try:
        dataset_report = validate_dataset(args.data, exploratory=args.exploratory)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc))
        return 1

    from ultralytics import YOLO

    print(f"Дообучение {args.model} на {args.data}")
    print(f"  эпох {args.epochs}, imgsz {args.imgsz}, batch {args.batch}, device {args.device}")

    t0 = time.perf_counter()
    model = YOLO(args.model)
    results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=RUNS_DIR,
        name=args.name,
        exist_ok=False,
        patience=args.patience,
        workers=args.workers,
        cache=(False if args.cache == "none" else args.cache),
        # Аугментации, осмысленные для уличной съёмки с рук:
        # телефон качается, освещение меняется, но мир не переворачивается.
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        flipud=0.0,
        hsv_v=0.5,          # разброс яркости: съёмка в разное время суток
        mosaic=1.0,
        close_mosaic=10,
        verbose=True,
    )
    train_s = time.perf_counter() - t0

    # Каталог берётся у тренера, а не собирается из имени: ultralytics
    # добавляет суффикс при коллизии, и угаданный путь оказывается пустым.
    save_dir = str(getattr(results, "save_dir", None)
                   or getattr(model.trainer, "save_dir", "")
                   or os.path.join(RUNS_DIR, args.name))
    weights = os.path.join(save_dir, "weights", "best.pt")
    print(f"\nОбучение заняло {train_s / 60:.1f} мин")
    print(f"Веса: {weights}")

    # --- оценка на отложенной части ------------------------------------
    print("\nОценка на test...")
    best = YOLO(weights)
    metrics = best.val(data=str(Path(args.data).resolve()), split="test", device=args.device, verbose=False)

    per_class = {}
    try:
        for i, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                "precision": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "mAP50_95": round(float(metrics.box.ap[i]), 4),
            }
    except (IndexError, AttributeError) as ex:
        print(f"  не удалось разложить метрики по классам: {ex}")

    report = {
        "publication_eligible": dataset_report["benchmark_ready"] and not args.exploratory,
        "dataset": str(Path(args.data).resolve()),
        "dataset_limitations": dataset_report["limitations"],
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_minutes": round(train_s / 60, 1),
        "weights": weights,
        "overall": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class": per_class,
    }

    print(f"\n{'класс':<12}{'precision':>11}{'recall':>9}{'mAP50':>9}{'mAP50-95':>11}")
    print("-" * 52)
    for name, m in per_class.items():
        print(f"{name:<12}{m['precision']:>11.3f}{m['recall']:>9.3f}"
              f"{m['mAP50']:>9.3f}{m['mAP50_95']:>11.3f}")
    o = report["overall"]
    print("-" * 52)
    print(f"{'всего':<12}{o['precision']:>11.3f}{o['recall']:>9.3f}"
          f"{o['mAP50']:>9.3f}{o['mAP50_95']:>11.3f}")

    os.makedirs(os.path.join("docs", "article"), exist_ok=True)
    out = os.path.join(save_dir, "evaluation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")

    # Keep weights with their run. A pilot must not silently replace the live model.
    return 0


if __name__ == "__main__":
    sys.exit(main())

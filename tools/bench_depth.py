# -*- coding: utf-8 -*-
"""
Проверка и замер модуля глубины: опасности без класса и цена инференса.

ДВА ВОПРОСА, НА КОТОРЫЕ ОТВЕЧАЕТ ЭТОТ ИНСТРУМЕНТ
------------------------------------------------
1. Находит ли метод отклонений от плоскости земли то, чего не находит
   детектор? Если геометрия срабатывает там же, где YOLO, она не нужна:
   уточнить расстояние до распознанного стула можно и дешевле.

2. Сколько стоит инференс и как часто его можно себе позволить.
   Это прямо задаёт depth_every_n_frames, а через него — возраст карты,
   по которой система утверждает о наличии ямы.

ЧТО СЧИТАЕТСЯ УСПЕХОМ
---------------------
Не «геометрия нашла много объектов» — их можно нашуметь, снизив порог.
Успех — это опасности, НЕ покрытые детектором: сектор, где геометрия
видит провал, а YOLO не видит ничего. Такие случаи считаются отдельно.

Запуск:
    python tools/bench_depth.py --device cpu --limit 20
    python tools/bench_depth.py --device cuda:0 --model DPT_Hybrid
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.config import Config                                     # noqa: E402
from core.perception.depth import DepthEstimator                    # noqa: E402
from core.perception.detector import Detector                       # noqa: E402
from core.risk.scoring import ObjectProfiles                        # noqa: E402
from core.types import CameraIntrinsics, Frame                      # noqa: E402

IMAGES = os.path.join("..", "Проект", "datasets", "sidewalk obstacle",
                      "test", "images", "*.jpg")

#: Угловой допуск, в пределах которого геометрическая опасность
#: считается «той же самой», что и детекция YOLO
OVERLAP_DEG = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="depth_anything_v2_s",
                    choices=["depth_anything_v2_s", "depth_anything_v2_b",
                             "MiDaS_small", "DPT_Hybrid"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--detector", default="data/models/yolov8n.pt")
    ap.add_argument("--save-vis", action="store_true",
                    help="сохранить карты глубины для просмотра")
    args = ap.parse_args()

    cfg = Config()
    cfg.perception.depth_model = args.model
    cfg.perception.depth_every_n_frames = 1        # для замера — каждый кадр
    cfg.perception.model_path = args.detector
    cfg.perception.device = args.device

    print(f"Загрузка модели глубины ({args.model})...")
    t0 = time.perf_counter()
    depth = DepthEstimator(cfg.perception).load(device=args.device)
    if not depth.available:
        print("MiDaS недоступна — замер невозможен")
        return 1
    print(f"  загружена за {time.perf_counter() - t0:.1f} с на {depth.device_name}")
    print(f"  бэкенд: {depth.backend}, модель: {depth.model_id}")

    profiles = ObjectProfiles.load(cfg.perception.profiles_path)
    det = Detector(cfg.perception, profiles).load()
    print(f"  детектор на {det.device}")

    images = sorted(glob.glob(IMAGES))[:args.limit]
    if not images:
        print(f"нет кадров по маске {IMAGES}")
        return 1

    infer_ms, hazard_counts = [], []
    total_geo = total_uncovered = total_yolo = 0
    by_kind = {"drop": 0, "rise": 0}
    uncovered_examples = []

    print(f"\n{'кадр':<26}{'depth мс':>10}{'YOLO':>6}{'геом':>6}"
          f"{'провал':>8}{'прегр':>7}{'вне YOLO':>10}")
    print("-" * 76)

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        intr = CameraIntrinsics(
            focal_px=cfg.io.camera_focal_px, width=w, height=h,
            hfov_deg=cfg.io.camera_hfov_deg,
            height_above_ground_m=cfg.io.camera_height_m,
            pitch_deg=cfg.io.camera_pitch_deg,
        )

        detections = det.detect(Frame(image=img, ts=time.monotonic(),
                                      intrinsics=intr, seq=0))
        risky = [d for d in detections if profiles.contributes_to_risk(d.label)]

        depth.  _last_seq = -10_000            # принудительно пересчитать
        dmap, _age = depth.estimate(img, seq=0, ts=time.monotonic())
        infer_ms.append(depth.last_infer_ms)

        geo = depth.geometric_hazards(dmap, intr, n_sectors=cfg.risk.n_sectors)
        drops = sum(1 for g in geo if g.label == "drop")
        rises = len(geo) - drops

        # Главный вопрос: сколько геометрических опасностей НЕ покрыты
        # детектором. Совпадающие ничего не добавляют.
        uncovered = [g for g in geo
                     if not any(abs(g.bearing_deg - d.bearing_deg) < OVERLAP_DEG
                                for d in risky)]

        total_geo += len(geo)
        total_yolo += len(risky)
        total_uncovered += len(uncovered)
        by_kind["drop"] += drops
        by_kind["rise"] += rises
        hazard_counts.append(len(geo))

        for g in uncovered[:2]:
            uncovered_examples.append({
                "image": os.path.basename(path), "kind": g.label,
                "bearing_deg": round(g.bearing_deg, 1),
                "distance_m": None if g.distance_m is None else round(g.distance_m, 1),
                "confidence": round(g.confidence, 2),
            })

        print(f"{os.path.basename(path)[:24]:<26}{depth.last_infer_ms:>10.0f}"
              f"{len(risky):>6}{len(geo):>6}{drops:>8}{rises:>7}{len(uncovered):>10}")

        if args.save_vis and dmap is not None:
            os.makedirs(os.path.join("docs", "article", "depth_vis"), exist_ok=True)
            norm = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            for g in geo:
                x1, y1, x2, y2 = g.bbox
                color = (0, 0, 255) if g.label == "drop" else (0, 255, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.imwrite(os.path.join("docs", "article", "depth_vis",
                                     os.path.basename(path)), vis)

    ms = sorted(infer_ms)
    n = len(ms)
    report = {
        "model": args.model,
        "model_id": depth.model_id,
        "backend": depth.backend,
        "device": depth.device_name,
        # Конвенция определена эмпирически, а не взята из документации:
        # при обратном знаке «провал» и «преграда» поменялись бы местами,
        # и система предупреждала бы наоборот, ничего не роняя.
        "higher_is_closer": depth.higher_is_closer,
        "frames": n,
        "infer_median_ms": round(ms[n // 2], 1),
        "infer_p95_ms": round(ms[int(n * 0.95) - 1], 1),
        "yolo_hazards_total": total_yolo,
        "geometric_hazards_total": total_geo,
        "geometric_not_covered_by_yolo": total_uncovered,
        "by_kind": by_kind,
        "uncovered_examples": uncovered_examples[:12],
    }

    print("-" * 76)
    print(f"инференс: медиана {report['infer_median_ms']} мс, "
          f"p95 {report['infer_p95_ms']} мс")
    print(f"опасностей: YOLO {total_yolo}, геометрия {total_geo}, "
          f"из них ВНЕ YOLO {total_uncovered}")
    if total_geo:
        print(f"доля геометрических, не покрытых детектором: "
              f"{100.0 * total_uncovered / total_geo:.0f} %")

    # Сколько кадров можно позволить между расчётами глубины,
    # чтобы карта не устаревала сверх допустимого
    max_age = cfg.perception.depth_max_age_s
    print(f"\nПри бюджете возраста карты {max_age} с и медиане "
          f"{report['infer_median_ms']:.0f} мс:")
    for fps in (10, 20, 30):
        every_n = max(1, int(max_age * fps))
        cost_share = report["infer_median_ms"] / (1000.0 / fps) / every_n
        print(f"  при {fps} к/с: раз в {every_n} кадров, "
              f"доля бюджета кадра {100 * cost_share:.0f} %")

    os.makedirs(os.path.join("docs", "article"), exist_ok=True)
    out = os.path.join("docs", "article",
                       f"depth_bench_{args.model}_{args.device.replace(':', '')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

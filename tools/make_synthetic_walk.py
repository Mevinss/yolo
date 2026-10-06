# -*- coding: utf-8 -*-
"""
Синтетическая запись прогулки — для проверки экспериментальной машинерии.

ЗАЧЕМ
-----
Прогон, метрики и ablation должны быть проверены ДО того, как будут
записаны настоящие прогулки. Иначе ошибка в подсчёте обнаружится
после часов ходьбы, а переснять их нельзя.

Здесь собирается запись в точности того формата, который пишет сервер:
video.mp4 + track.jsonl + meta.json + hazards.jsonl. Кадры берутся
из уличного датасета, положения — с настоящего маршрута по OSM Астаны,
разметка опасностей строится по разметке датасета.

ЧЕМ ЭТО НЕ ЯВЛЯЕТСЯ
-------------------
Это НЕ данные для статьи. Кадр и координата здесь не принадлежат
одному моменту: человек идёт по маршруту, а видит произвольные кадры
датасета. Любая метрика, посчитанная по такой записи, характеризует
работоспособность кода, а не качество навигации.

Чтобы это нельзя было спутать, в meta.json ставится
"synthetic": true, а имя папки начинается с "synthetic_".
Все инструменты обязаны исключать такие записи из итоговых таблиц.

Запуск:
    python tools/make_synthetic_walk.py --frames 400
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.config import Config                                     # noqa: E402
from core.routing.graph import PedestrianGraph                     # noqa: E402
from core.routing.maneuvers import bearing, haversine_m            # noqa: E402
from core.routing.planner import Router                            # noqa: E402
from core.routing.weights import AccessibilityWeights              # noqa: E402

DATASET_DIR = os.path.join("data", "datasets", "kz_hazards", "test")
GRAPH_CACHE = os.path.join("data", "osm", "graph_cache.pkl")
OUT_ROOT = os.path.join("data", "walks")

BAYTEREK = (51.1283, 71.4305)
KHAN_SHATYR = (51.1325, 71.4044)

FPS = 10.0
WALK_SPEED_MPS = 1.2
CLASS_NAMES = ["pothole", "obstacle", "stairs"]
SEVERITY = {"pothole": "danger", "stairs": "danger", "obstacle": "obstacle"}

#: За сколько секунд до появления опасности в кадре предупреждение
#: ещё полезно. Два шага обычной ходьбы.
WARN_LEAD_S = 1.5


def load_dataset_frames(limit: int) -> list:
    """-> [(изображение, [(класс, xc, yc, w, h), ...]), ...]"""
    img_dir = os.path.join(DATASET_DIR, "images")
    lbl_dir = os.path.join(DATASET_DIR, "labels")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"нет {img_dir} — сначала tools/build_dataset.py")

    out = []
    for path in sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:limit]:
        img = cv2.imread(path)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        boxes = []
        lbl = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lbl):
            with open(lbl, encoding="utf-8") as fh:
                for line in fh:
                    p = line.split()
                    if len(p) >= 5:
                        boxes.append((int(float(p[0])), *[float(x) for x in p[1:5]]))
        out.append((img, boxes))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--name", default="synthetic_bayterek_khanshatyr")
    args = ap.parse_args()

    cfg = Config()
    weights = AccessibilityWeights(cfg.routing)
    graph = PedestrianGraph.load_cache(GRAPH_CACHE, weights)
    router = Router(cfg.routing, graph, weights)

    print("Построение маршрута...")
    route = router.plan(BAYTEREK, KHAN_SHATYR)
    print(f"  {route.total_distance_m:.0f} м")

    frames = load_dataset_frames(args.frames)
    if not frames:
        print("нет кадров датасета")
        return 1
    print(f"  кадров датасета: {len(frames)}")

    walk_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(walk_dir, exist_ok=True)

    h, w = frames[0][0].shape[:2]
    video = cv2.VideoWriter(os.path.join(walk_dir, "video.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    pts = route.polyline
    seg_len = np.array([haversine_m(*pts[k], *pts[k + 1]) for k in range(len(pts) - 1)])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    ts0 = 1000.0
    track, hazards = [], []
    hfov = cfg.io.camera_hfov_deg

    n = min(args.frames, len(frames))
    for i in range(n):
        img, boxes = frames[i % len(frames)]
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        video.write(img)

        ts = ts0 + i / FPS
        s = min(WALK_SPEED_MPS * (i / FPS), total - 1.0)
        seg = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(pts) - 2))
        f = (s - float(cum[seg])) / max(float(seg_len[seg]), 1e-6)
        lat = pts[seg][0] + (pts[seg + 1][0] - pts[seg][0]) * f
        lon = pts[seg][1] + (pts[seg + 1][1] - pts[seg][1]) * f
        head = bearing(*pts[seg], *pts[seg + 1])

        track.append({"seq": i, "ts": round(ts, 3),
                      "lat": round(lat, 7), "lon": round(lon, 7),
                      "heading": round(head, 2), "accuracy": 8.0})

        # Разметка опасностей строится из разметки датасета: угол —
        # из положения бокса в кадре, дальность — из его нижнего края.
        for cls, xc, yc, bw, bh in boxes:
            name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "obstacle"
            bearing_deg = (xc - 0.5) * hfov
            bottom_y = (yc + bh / 2) * h
            approx_d = max(1.0, 12.0 * (1.0 - min(bottom_y / h, 0.99)) + 1.0)
            hazards.append({
                "ts": round(ts, 3), "seq": i, "type": name,
                "severity": SEVERITY.get(name, "obstacle"),
                "bearing_deg": round(bearing_deg, 1),
                "distance_m": round(approx_d, 1),
                "warn_by_ts": round(ts - WARN_LEAD_S, 3),
                "synthetic": True,
            })

    video.release()

    with open(os.path.join(walk_dir, "track.jsonl"), "w", encoding="utf-8") as fh:
        for r in track:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(walk_dir, "hazards.jsonl"), "w", encoding="utf-8") as fh:
        for r in hazards:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(walk_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "walk_id": args.name,
            # Метка, по которой все инструменты исключают запись
            # из итоговых таблиц статьи.
            "synthetic": True,
            "warning": ("Кадр и координата не принадлежат одному моменту. "
                        "Пригодно только для проверки кода, не для выводов."),
            "destination": {"lat": KHAN_SHATYR[0], "lon": KHAN_SHATYR[1]},
            "route_description": "Байтерек -> Хан Шатыр (синтетика)",
            "fps": FPS, "frames": n,
            "duration_s": round(n / FPS, 1),
        }, fh, ensure_ascii=False, indent=2)

    print(f"\n  {walk_dir}")
    print(f"  кадров {n}, {n / FPS:.0f} с, разметка: {len(hazards)} опасностей")
    return 0


if __name__ == "__main__":
    sys.exit(main())

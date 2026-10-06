# -*- coding: utf-8 -*-
"""
Демонстрация шага 4: кадр -> распознавание -> расстояние -> речь.

Проверяет сквозную работу восприятия и синтеза БЕЗ слоёв слияния,
которые реализуются на шаге 5. По функциональности это примерно
то, что умел проект-предшественника, плюс углы, неопределённость
расстояния, категории объектов и казахская речь.

ВАЖНАЯ ОГОВОРКА
---------------
Построение фразы здесь временное и намеренно примитивное: настоящее
живёт в core/guidance/phrasing.py и подчиняется правилам из
docs/mockups/voice_scenarios.md (действие раньше причины, бюджет речи,
смягчение при низком доверии). Здесь оно нужно лишь для того, чтобы
измерить сквозную задержку от кадра до готового звука.

Запуск:
    python tools/demo_perception.py --image путь.jpg
    python tools/demo_perception.py --dataset       # выборка уличных кадров
    python tools/demo_perception.py --image x.jpg --lang kk --speak
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

from core.config import Config                                    # noqa: E402
from core.guidance.tts import TTSEngine                            # noqa: E402
from core.perception.detector import Detector                      # noqa: E402
from core.risk.scoring import ObjectProfiles                       # noqa: E402
from core.types import CameraIntrinsics, Frame, Lang, Urgency, Utterance  # noqa: E402

DATASET_GLOB = os.path.join("..", "Проект", "datasets", "sidewalk obstacle",
                            "test", "images", "*.jpg")

POSITION_RU = {"left": "слева", "center": "прямо", "right": "справа"}
POSITION_KK = {"left": "солда", "center": "алдыда", "right": "оңда"}


def position_of(bearing_deg: float) -> str:
    if bearing_deg < -12:
        return "left"
    if bearing_deg > 12:
        return "right"
    return "center"


def build_phrase(dets: list, profiles: ObjectProfiles, lang: Lang,
                 sigma_threshold: float) -> tuple:
    """Временная сборка фразы. Настоящая — в core/guidance/phrasing.py."""
    risky = [d for d in dets if profiles.contributes_to_risk(d.label) and d.distance_m]
    if not risky:
        landmarks = [d for d in dets if profiles.is_landmark(d.label)]
        if landmarks:
            d = landmarks[0]
            pos = (POSITION_KK if lang is Lang.KK else POSITION_RU)[position_of(d.bearing_deg)]
            name = profiles.name(d.label, lang.value)
            text = f"{pos} {name}" if lang is Lang.KK else f"{pos.capitalize()} {name}"
            return text, Urgency.INFO
        return ("Кедергі жоқ" if lang is Lang.KK else "Препятствий не вижу"), Urgency.INFO

    # ближайший опасный объект
    d = min(risky, key=lambda x: x.distance_m)
    name = profiles.name(d.label, lang.value)
    pos = (POSITION_KK if lang is Lang.KK else POSITION_RU)[position_of(d.bearing_deg)]

    # Метры произносятся, только если оценке можно верить —
    # иначе «близко». Ложная точность подрывает доверие.
    trust = d.distance_sigma_m is not None and d.distance_sigma_m <= sigma_threshold
    if lang is Lang.KK:
        dist = f"{d.distance_m:.0f} метр" if trust else "жақын"
        text = f"{pos} {name}, {dist}"
    else:
        dist = f"{d.distance_m:.1f} метра" if trust else "близко"
        text = f"{pos.capitalize()} {name}, {dist}"

    urgency = Urgency.EMERGENCY if (profiles.is_danger(d.label) and d.distance_m < 2.0) \
        else Urgency.CAUTION
    return text, urgency


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--dataset", action="store_true")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--lang", default="ru", choices=["ru", "kk", "en"])
    ap.add_argument("--model", default="data/models/yolov8n.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speak", action="store_true", help="сохранить WAV рядом")
    args = ap.parse_args()

    lang = Lang(args.lang)
    cfg = Config()
    cfg.perception.model_path = args.model
    cfg.perception.device = args.device
    cfg.guidance.lang = lang

    print("Загрузка...")
    t0 = time.perf_counter()
    profiles = ObjectProfiles.load(cfg.perception.profiles_path)
    detector = Detector(cfg.perception, profiles).load()
    tts = TTSEngine(cfg.guidance).load(langs=[lang])
    print(f"  готово за {time.perf_counter() - t0:.1f} с "
          f"| детектор: {detector.device} | голоса: {[l.value for l in tts.loaded_langs]}")

    if args.dataset:
        images = sorted(glob.glob(DATASET_GLOB))[:args.limit]
        if not images:
            print(f"нет изображений по маске {DATASET_GLOB}")
            return 1
    elif args.image:
        images = [args.image]
    else:
        images = [os.path.join("..", "Проект", "yolo-main", "mobile_app", "latest_frame.jpg")]

    rows = []
    print(f"\n{'кадр':<26}{'объектов':>9}{'детекц':>9}{'синтез':>9}{'всего':>9}  реплика")
    print("-" * 108)

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"  не читается: {path}")
            continue
        h, w = img.shape[:2]
        intr = CameraIntrinsics(
            focal_px=cfg.io.camera_focal_px, width=w, height=h,
            hfov_deg=cfg.io.camera_hfov_deg,
            height_above_ground_m=cfg.io.camera_height_m,
            pitch_deg=cfg.io.camera_pitch_deg,
        )

        t_all = time.perf_counter()
        t = time.perf_counter()
        dets = detector.detect(Frame(image=img, ts=time.monotonic(), intrinsics=intr, seq=0))
        t_det = (time.perf_counter() - t) * 1000

        text, urgency = build_phrase(dets, profiles, lang,
                                     cfg.guidance.metric_sigma_threshold_m)

        t = time.perf_counter()
        wav = tts.synthesize(Utterance(text=text, lang=lang, urgency=urgency, ts=0.0))
        t_tts = (time.perf_counter() - t) * 1000
        t_total = (time.perf_counter() - t_all) * 1000

        risky = sum(1 for d in dets if profiles.contributes_to_risk(d.label))
        print(f"{os.path.basename(path)[:24]:<26}{len(dets):>4}/{risky:<4}"
              f"{t_det:>8.0f}м{t_tts:>8.0f}м{t_total:>8.0f}м  "
              f"[{urgency.name[:4]}] {text}")

        rows.append({
            "image": os.path.basename(path), "detections": len(dets), "risky": risky,
            "detect_ms": round(t_det, 1), "tts_ms": round(t_tts, 1),
            "total_ms": round(t_total, 1), "text": text, "urgency": urgency.name,
        })

        if args.speak and wav:
            out = os.path.join("data", "models", "tts", "samples",
                               f"demo_{lang.value}_{os.path.basename(path)}.wav")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as fh:
                fh.write(wav)

    if rows:
        det = sorted(r["detect_ms"] for r in rows)
        tot = sorted(r["total_ms"] for r in rows)
        mid = len(rows) // 2
        print("-" * 108)
        print(f"медиана: детекция {det[mid]:.0f} мс ({1000 / det[mid]:.1f} к/с), "
              f"кадр -> звук {tot[mid]:.0f} мс")

        os.makedirs(os.path.join("docs", "article"), exist_ok=True)
        out = os.path.join("docs", "article", f"perception_demo_{lang.value}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"device": detector.device, "model": args.model,
                       "lang": lang.value, "rows": rows}, fh,
                      ensure_ascii=False, indent=2)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

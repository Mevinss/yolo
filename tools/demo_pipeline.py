# -*- coding: utf-8 -*-
"""
Сквозной прогон ВСЕЙ системы: маршрут Астаны + реальные кадры -> речь.

Это первая проверка, в которой работают одновременно оба уровня.
Пользователь виртуально идёт по настоящему маршруту, построенному
по выгрузке OSM, а камера показывает настоящие уличные кадры.

ЧТО ИМЕННО ПРОВЕРЯЕТСЯ
----------------------
Не то, что каждый модуль работает по отдельности — это покрыто тестами.
Проверяется, что слияние выдаёт ОСМЫСЛЕННУЮ последовательность реплик:
маршрутные подсказки перемежаются предупреждениями об обстановке,
речь не льётся сплошным потоком, а отклонения объясняются причиной.

ОГОВОРКА О ЧЕСТНОСТИ
--------------------
Положение здесь синтетическое: пользователь движется по полилинии
маршрута с постоянной скоростью, а кадры берутся из датасета и
не соответствуют этому положению. Это демонстрация связности системы,
а НЕ измерение качества навигации. Настоящие цифры дадут записанные
прогулки на шаге 6, где кадр и координата принадлежат одному моменту.

Запуск:
    python tools/demo_pipeline.py
    python tools/demo_pipeline.py --lang kk --device cuda:0 --steps 60
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.config import Config                                     # noqa: E402
from core.fusion.policy import FusionPolicy                        # noqa: E402
from core.guidance.phrasing import Phrasebook                      # noqa: E402
from core.guidance.scheduler import UtteranceScheduler             # noqa: E402
from core.io.health import HealthMonitor                           # noqa: E402
from core.local_planner.corridor import CorridorPlanner            # noqa: E402
from core.perception.depth import DepthEstimator                   # noqa: E402
from core.perception.detector import Detector                      # noqa: E402
from core.pipeline import Pipeline                                 # noqa: E402
from core.risk.field import RiskFieldBuilder                       # noqa: E402
from core.risk.scoring import ObjectProfiles                       # noqa: E402
from core.routing.graph import PedestrianGraph                     # noqa: E402
from core.routing.maneuvers import bearing, haversine_m            # noqa: E402
from core.routing.planner import Router                            # noqa: E402
from core.routing.weights import AccessibilityWeights              # noqa: E402
from core.types import CameraIntrinsics, Frame, Lang, Pose, PoseSource  # noqa: E402

IMAGES_GLOB = os.path.join("..", "Проект", "datasets", "sidewalk obstacle",
                           "test", "images", "*.jpg")
GRAPH_CACHE = os.path.join("data", "osm", "graph_cache.pkl")

BAYTEREK = (51.1283, 71.4305)
KHAN_SHATYR = (51.1325, 71.4044)

WALK_SPEED_MPS = 1.2
STEP_DT_S = 0.5          # шаг симуляции: реплики разнесены во времени


def build(cfg: Config):
    profiles = ObjectProfiles.load(cfg.perception.profiles_path)
    weights = AccessibilityWeights(cfg.routing)
    graph = PedestrianGraph.load_cache(GRAPH_CACHE, weights)
    router = Router(cfg.routing, graph, weights)

    detector = Detector(cfg.perception, profiles).load()
    book = Phrasebook(cfg.guidance, profiles)

    depth = None
    if cfg.perception.use_depth:
        depth = DepthEstimator(cfg.perception).load(device=cfg.perception.device)
        if not depth.available:
            # Отсутствие глубины не останавливает систему, но и молчать
            # об этом нельзя: без неё не видны опасности без класса.
            print("  глубина недоступна — прогон без опасностей без класса")

    pipe = Pipeline(
        config=cfg,
        detector=detector,
        risk_builder=RiskFieldBuilder(cfg.risk, profiles),
        router=router,
        local_planner=CorridorPlanner(cfg.local_planner),
        fusion=FusionPolicy(cfg.fusion),
        guidance=UtteranceScheduler(cfg.guidance, book),
        health_monitor=HealthMonitor(),
        depth_estimator=depth,
    )
    return pipe, detector, profiles, depth


def walk_poses(route, n_steps: int, ts0: float):
    """Идём по полилинии маршрута с постоянной скоростью.

    Положение задаётся кумулятивным расстоянием от начала маршрута,
    а сегмент находится двоичным поиском. Инкрементальный счётчик здесь
    ошибочен: он смешивает «пройдено до начала сегмента» и «пройдено
    всего», из-за чего пользователь топчется на месте, а система
    бесконечно повторяет одну и ту же подсказку.
    """
    pts = route.polyline
    seg_len = np.array([haversine_m(*pts[k], *pts[k + 1]) for k in range(len(pts) - 1)])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    for i in range(n_steps):
        s = WALK_SPEED_MPS * STEP_DT_S * i
        if s >= total:
            break
        seg = int(np.searchsorted(cum, s, side="right") - 1)
        seg = max(0, min(seg, len(pts) - 2))

        lat1, lon1 = pts[seg]
        lat2, lon2 = pts[seg + 1]
        d = max(float(seg_len[seg]), 1e-6)
        f = min(max((s - float(cum[seg])) / d, 0.0), 1.0)

        yield Pose(
            lat=lat1 + (lat2 - lat1) * f,
            lon=lon1 + (lon2 - lon1) * f,
            heading_deg=bearing(lat1, lon1, lat2, lon2),
            ts=ts0 + i * STEP_DT_S,
            accuracy_m=8.0,
            heading_sigma_deg=14.0,
            speed_mps=WALK_SPEED_MPS,
            source=PoseSource.REPLAY,
        )


def guard_poses(start: Pose, n_steps: int):
    """Ходьба вперёд без маршрута: курс держится, положение почти не важно.

    В режиме стража координата нужна лишь для журнала: решения
    принимаются по кадру и курсу взгляда.
    """
    for i in range(n_steps):
        yield Pose(
            lat=start.lat, lon=start.lon,
            heading_deg=(start.heading_deg + 4.0 * np.sin(i / 12.0)) % 360.0,
            ts=start.ts + i * STEP_DT_S,
            accuracy_m=8.0, heading_sigma_deg=14.0,
            speed_mps=WALK_SPEED_MPS, source=PoseSource.REPLAY,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ru", choices=["ru", "kk", "en"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="data/models/yolov8m.pt")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--with-depth", action="store_true",
                    help="включить оценку глубины и опасности без класса")
    ap.add_argument("--depth-model", default="depth_anything_v2_s")
    ap.add_argument("--guard", action="store_true",
                    help="режим стража: без маршрута, только предупреждения")
    ap.add_argument("--audio", metavar="OUT.wav",
                    help="синтезировать звук всей прогулки в один файл")
    args = ap.parse_args()

    cfg = Config()
    cfg.perception.model_path = args.model
    cfg.perception.device = args.device
    cfg.guidance.lang = Lang(args.lang)
    cfg.perception.use_depth = args.with_depth
    cfg.perception.depth_model = args.depth_model

    print("Сборка системы...")
    t0 = time.perf_counter()
    pipe, detector, profiles, depth = build(cfg)
    print(f"  готово за {time.perf_counter() - t0:.1f} с | детектор: {detector.device}")
    if depth is not None and depth.available:
        print(f"  глубина: {depth.model_id} на {depth.device_name} "
              f"(раз в {cfg.perception.depth_every_n_frames} кадров)")

    images = sorted(glob.glob(IMAGES_GLOB))
    if not images:
        print(f"нет кадров по маске {IMAGES_GLOB}")
        return 1
    frames = [cv2.imread(p) for p in images[:40]]
    frames = [f for f in frames if f is not None]

    ts0 = time.monotonic()
    start_pose = Pose(lat=BAYTEREK[0], lon=BAYTEREK[1], heading_deg=0.0, ts=ts0,
                      accuracy_m=8.0, heading_sigma_deg=14.0)

    if args.guard:
        # Режим стража: цель не ставится вовсе. Человек идёт своей
        # дорогой, а система следит за тем, что перед ним.
        print("\nРежим стража: маршрут не строится, только предупреждения")
        route = None
    else:
        print("\nПостроение маршрута Байтерек -> Хан Шатыр...")
        route = pipe.set_destination(start_pose, *KHAN_SHATYR)
        print(f"  {route.total_distance_m:.0f} м, {len(route.maneuvers)} манёвров, "
              f"{route.profile.get('pedestrian_share_pct')}% по пешеходной инфраструктуре")

    h, w = frames[0].shape[:2]
    intr = CameraIntrinsics(
        focal_px=cfg.io.camera_focal_px, width=w, height=h,
        hfov_deg=cfg.io.camera_hfov_deg,
        height_above_ground_m=cfg.io.camera_height_m,
        pitch_deg=cfg.io.camera_pitch_deg,
    )

    print(f"\n{'t':>6}{'действие':<14}{'фаза':<13}{'дов':>5}{'мс':>6}  реплика")
    print("-" * 100)

    spoken = 0
    timings: list = []
    layer_ms: dict = {}
    geo_hazards = 0
    narration: list = []          # (секунда от старта, реплика)
    poses = (walk_poses(route, args.steps, ts0) if route is not None
             else guard_poses(start_pose, args.steps))
    for i, p in enumerate(poses):
        img = frames[i % len(frames)]
        tick = pipe.step(Frame(image=img, ts=p.ts, intrinsics=intr, seq=i), p)
        timings.append(tick.timings_ms["total_ms"])
        for k, v in tick.timings_ms.items():
            layer_ms.setdefault(k, []).append(v)
        geo_hazards += sum(1 for d in tick.detections if d.label in ("drop", "rise"))

        text = tick.utterance.text if tick.utterance else ""
        if tick.utterance:
            spoken += 1
        act = tick.fusion.action.value if tick.fusion else "—"
        phase = (tick.fusion.crossing_phase.value
                 if tick.fusion and tick.fusion.crossing_phase else "")
        if tick.utterance is not None:
            narration.append((i * STEP_DT_S, tick.utterance))
        mark = "*" if tick.utterance else " "
        print(f"{i * STEP_DT_S:>6.1f}{act:<14}{phase:<13}"
              f"{tick.health.confidence:>5.2f}{tick.timings_ms['total_ms']:>6.0f}{mark} {text}")

    stats = pipe.guidance.stats()
    ts_sorted = sorted(timings)
    duration_min = (len(timings) * STEP_DT_S) / 60.0
    print("-" * 100)
    print(f"кадров: {len(timings)} | медиана {ts_sorted[len(ts_sorted) // 2]:.0f} мс "
          f"| p95 {ts_sorted[int(len(ts_sorted) * .95) - 1]:.0f} мс")
    print(f"реплик: {spoken} за {duration_min:.1f} мин = "
          f"{spoken / max(duration_min, 1e-6):.1f} в минуту")
    print(f"подавлено: бюджет {stats['suppressed_by_budget']}, "
          f"повтор {stats['suppressed_by_dedup']}, "
          f"нечего сказать {stats['nothing_to_say']}")
    if cfg.perception.use_depth:
        print(f"опасностей без класса найдено: {geo_hazards}")

    # Медиана и p95 по слоям, а не последний кадр.
    #
    # Глубина считается раз в N кадров, поэтому на отдельном кадре её
    # стоимость либо вся целиком, либо ноль. Одиночный замер вводит
    # в заблуждение в обе стороны, а для аида важнее хвост, а не середина:
    # медиана скрывает именно те кадры, на которых система задумывается.
    print(f"\n{'слой':<14}{'медиана':>10}{'p95':>9}{'макс':>9}")
    print("-" * 42)
    for key in ("health_ms", "perception_ms", "depth_ms", "risk_ms",
                "local_ms", "routing_ms", "fusion_ms", "guidance_ms", "total_ms"):
        vals = sorted(v for v in layer_ms.get(key, []) if v is not None)
        if not vals:
            continue
        n = len(vals)
        print(f"{key.replace('_ms', ''):<14}{vals[n // 2]:>10.2f}"
              f"{vals[int(n * 0.95) - 1]:>9.2f}{vals[-1]:>9.2f}")

    if args.audio:
        write_narration(narration, cfg, args.audio)
    return 0


def write_narration(narration: list, cfg: Config, out_path: str) -> None:
    """Собрать звуковую дорожку прогулки: реплики на своих секундах.

    Реплики расставляются по РЕАЛЬНОМУ времени, а между ними тишина.
    Склеивать их подряд было бы бессмысленно: половина смысла системы —
    в том, КОГДА она молчит. Дорожка, где реплики идут вплотную,
    показывает совсем другой продукт.
    """
    import wave
    from core.guidance.tts import TTSEngine

    if not narration:
        print("\nреплик не было — дорожка не собрана")
        return

    print(f"\nСинтез {len(narration)} реплик...")
    tts = TTSEngine(cfg.guidance).load(langs=[cfg.guidance.lang])
    if not tts.available:
        print("  синтез недоступен — нет голосов, см. tools/fetch_voices.py")
        return

    clips = []
    rate = None
    for t_rel, utt in narration:
        data = tts.synthesize(utt)
        if not data:
            continue
        with wave.open(io.BytesIO(data), "rb") as w:
            r = w.getframerate()
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if rate is None:
            rate = r
        elif r != rate:
            # Быстрый и качественный голоса могут иметь разную частоту;
            # приводим к первой, иначе дорожка поедет по времени.
            idx = np.linspace(0, len(frames) - 1, int(len(frames) * rate / r))
            frames = np.interp(idx, np.arange(len(frames)), frames).astype(np.int16)
        clips.append((t_rel, frames))

    if not clips:
        print("  ни одна реплика не синтезировалась")
        return

    total_s = clips[-1][0] + len(clips[-1][1]) / rate + 1.0
    track = np.zeros(int(total_s * rate), dtype=np.int32)
    for t_rel, frames in clips:
        start = int(t_rel * rate)
        end = min(start + len(frames), len(track))
        track[start:end] += frames[:end - start]
    track = np.clip(track, -32768, 32767).astype(np.int16)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(track.tobytes())

    speech_s = sum(len(f) for _, f in clips) / rate
    print(f"  -> {out_path}  ({total_s:.0f} с, речь занимает "
          f"{speech_s:.0f} с = {100 * speech_s / total_s:.0f} % времени)")


if __name__ == "__main__":
    sys.exit(main())

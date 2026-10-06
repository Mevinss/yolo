# -*- coding: utf-8 -*-
"""
Офлайн-прогон записанной прогулки. ГЛАВНЫЙ ИНСТРУМЕНТ ДЛЯ СТАТЬИ.

Берёт запись из data/walks/<walk_id>/ и прогоняет её через тот же
Pipeline, что работает в поле. Итог — JSONL с Tick'ами, который читает
eval/metrics.py.

ЗАЧЕМ
-----
  1. Воспроизводимость: результат зависит от записи и конфига,
     а не от погоды в день эксперимента. Рецензент может повторить.
  2. Ablation: один и тот же маршрут прогоняется с отключёнными
     слагаемыми. В поле такое воспроизвести невозможно — нельзя
     дважды пройти одну улицу в одинаковых условиях.
  3. Baseline: те же записи через конфигурации «только маршрут»
     и «только зрение» дают честное сравнение с парадигмами
     конкурентов, без тестирования чужих приложений.
  4. Скорость: перебор параметров занимает минуты, а не дни выходов.

ФОРМАТ ЗАПИСИ
-------------
    data/walks/<walk_id>/
        video.mp4          кадры, индекс = Tick.seq
        track.jsonl        GPS, компас, метки времени
        meta.json          маршрут, погода, освещённость, время суток
        hazards.jsonl      ручная разметка опасностей (ground truth)

ВРЕМЯ ВОСПРОИЗВОДИТСЯ, А НЕ ПЕРЕСЧИТЫВАЕТСЯ
-------------------------------------------
Кадры подаются с ИСХОДНЫМИ метками времени, а не так быстро, как
читается файл. Арбитраж речи целиком построен на интервалах: при
сжатом времени он подавит всё подряд, и цифры в статье не будут
соответствовать полевому поведению.

Запуск:
    python -m eval.replay --walk data/walks/walk_20260823_101500
    python -m eval.replay --walk ... --profile no_fusion_risk
    python -m eval.replay --walk ... --set fusion.w_route=0.2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ABLATION_PROFILES, Config                  # noqa: E402
from core.fusion.policy import FusionPolicy                        # noqa: E402
from core.guidance.phrasing import Phrasebook                      # noqa: E402
from core.guidance.scheduler import UtteranceScheduler             # noqa: E402
from core.io.health import HealthMonitor                           # noqa: E402
from core.io.pose import PoseEstimator                             # noqa: E402
from core.local_planner.corridor import CorridorPlanner            # noqa: E402
from core.perception.detector import Detector                      # noqa: E402
from core.pipeline import Pipeline                                 # noqa: E402
from core.risk.field import RiskFieldBuilder                       # noqa: E402
from core.risk.scoring import ObjectProfiles                       # noqa: E402
from core.routing.graph import PedestrianGraph                     # noqa: E402
from core.routing.planner import Router                            # noqa: E402
from core.routing.weights import AccessibilityWeights              # noqa: E402
from core.serialization import TickWriter                          # noqa: E402
from core.types import CameraIntrinsics, Frame, Lang, Pose, PoseSource  # noqa: E402

GRAPH_CACHE = os.path.join("data", "osm", "graph_cache.pkl")


# ---------------------------------------------------------------------------
# Конфигурация прогона
# ---------------------------------------------------------------------------

def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Применить точечные правки вида {"fusion.w_route": 0.0}.

    Правки применяются к КОПИИ: один процесс прогоняет несколько
    конфигураций подряд, и утечка настроек между ними испортила бы
    таблицу ablation незаметно.
    """
    cfg = copy.deepcopy(cfg)
    for path, value in overrides.items():
        section, _, field = path.partition(".")
        if not field:
            setattr(cfg, section, value)
            continue
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, field):
            raise KeyError(f"неизвестный параметр: {path}")
        setattr(target, field, value)
    return cfg


def profile_config(base: Config, profile: str) -> Config:
    if profile not in ABLATION_PROFILES:
        raise KeyError(f"нет профиля {profile}; есть: {list(ABLATION_PROFILES)}")
    cfg = apply_overrides(base, ABLATION_PROFILES[profile])
    cfg.profile_name = profile
    return cfg


# ---------------------------------------------------------------------------
# Источник записи
# ---------------------------------------------------------------------------

class WalkRecording:
    """Кадры и положения одной прогулки, сведённые по времени."""

    def __init__(self, walk_dir: str):
        self.dir = walk_dir
        self.meta = self._load_json("meta.json", {})
        self.track = self._load_jsonl("track.jsonl")
        if not self.track:
            raise FileNotFoundError(f"нет track.jsonl в {walk_dir}")
        self.video_path = os.path.join(walk_dir, "video.mp4")
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"нет video.mp4 в {walk_dir}")

    def _load_json(self, name: str, default):
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def _load_jsonl(self, name: str) -> list:
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            return []
        rows = []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def frames(self):
        """-> (seq, image, ts, track_row)

        Каждому кадру сопоставляется ближайшая по времени запись трека.
        GPS приходит раз в секунду, кадры — двадцать раз, поэтому
        сопоставление обязательно интерполирующее или ближайшее;
        брать «последнюю пришедшую» значит систематически отставать.
        """
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        previous_ts = None
        try:
            if not cap.isOpened():
                raise ValueError("Video cannot be opened")
            # Session.write_frame writes one track row per stored video frame.
            # Container FPS is only a playback setting; acquisition is irregular.
            for index, row in enumerate(self.track):
                ok, img = cap.read()
                if not ok:
                    raise ValueError(f"Video ended before track row {index}")
                ts = float(row["ts"])
                if not math.isfinite(ts) or (previous_ts is not None and ts < previous_ts):
                    raise ValueError(f"Track timestamp is invalid at row {index}")
                previous_ts = ts
                yield int(row["seq"]), img, ts, row
            if cap.read()[0]:
                raise ValueError("Video has frames without track metadata")
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------

def build_pipeline(cfg: Config):
    profiles = ObjectProfiles.load(cfg.perception.profiles_path)
    weights = AccessibilityWeights(cfg.routing)
    graph = PedestrianGraph.load_cache(GRAPH_CACHE, weights)
    router = Router(cfg.routing, graph, weights)
    detector = Detector(cfg.perception, profiles).load()
    depth_estimator = None
    if cfg.perception.use_depth:
        from core.perception.depth import DepthEstimator
        depth_estimator = DepthEstimator(cfg.perception).load(device=detector.device)

    return Pipeline(
        config=cfg,
        detector=detector,
        risk_builder=RiskFieldBuilder(cfg.risk, profiles),
        router=router,
        local_planner=CorridorPlanner(cfg.local_planner),
        fusion=FusionPolicy(cfg.fusion),
        guidance=UtteranceScheduler(cfg.guidance, Phrasebook(cfg.guidance, profiles)),
        health_monitor=HealthMonitor(),
        depth_estimator=depth_estimator,
    )


def replay(walk_dir: str, cfg: Config, out_path: Optional[str] = None,
           verbose: bool = False) -> str:
    """Прогнать запись через конвейер. Возвращает путь к логу Tick'ов."""
    rec = WalkRecording(walk_dir)
    pipe = build_pipeline(cfg)
    pose_est = PoseEstimator(cfg.io)

    dest = rec.meta.get("destination")
    route = None

    out_path = out_path or os.path.join(walk_dir, f"ticks_{cfg.profile_name}.jsonl.gz")
    n = 0
    t0 = time.perf_counter()

    import cv2
    writer = None
    try:
        for seq, img, ts, row in rec.frames():
            h, w = img.shape[:2]
            intr = CameraIntrinsics(
                focal_px=cfg.io.camera_focal_px, width=w, height=h,
                hfov_deg=cfg.io.camera_hfov_deg,
                height_above_ground_m=cfg.io.camera_height_m,
                pitch_deg=cfg.io.camera_pitch_deg,
            )
            gray = cv2.cvtColor(cv2.resize(img, (160, 120)), cv2.COLOR_BGR2GRAY)
            pose = pose_est.update(
                lat=float(row["lat"]), lon=float(row["lon"]),
                compass_deg=row.get("heading"),
                accuracy_m=float(row.get("accuracy", 15.0)),
                ts=ts, gray=gray,
            )
            pose = Pose(**{**pose.__dict__, "source": PoseSource.REPLAY})

            if route is None and dest:
                try:
                    route = pipe.set_destination(pose, float(dest["lat"]), float(dest["lon"]))
                except ValueError as ex:
                    print(f"  маршрут не построен: {ex}")
                    dest = None

            if writer is None:
                writer = TickWriter(out_path, config=cfg.to_dict(), route=route,
                                    meta={**rec.meta, "replay_of": walk_dir,
                                          "profile": cfg.profile_name})

            tick = pipe.step(Frame(image=img, ts=ts, intrinsics=intr, seq=seq), pose)
            writer.write(tick)
            n += 1
            if verbose and tick.utterance:
                print(f"  {ts - rec.track[0]['ts']:>7.1f}  {tick.utterance.text}")
    finally:
        if writer:
            writer.close()

    dt = time.perf_counter() - t0
    print(f"  {cfg.profile_name:<18}{n:>6} кадров за {dt:.0f} с -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Офлайн-прогон записанной прогулки")
    ap.add_argument("--walk", required=True)
    ap.add_argument("--profile", default="full", choices=list(ABLATION_PROFILES))
    ap.add_argument("--set", action="append", default=[],
                    help="точечная правка, напр. fusion.w_route=0.2")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="data/models/yolov8m.pt")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.perception.model_path = args.model
    cfg.perception.device = args.device
    cfg.guidance.lang = Lang(args.lang)
    cfg = profile_config(cfg, args.profile)

    manual = {}
    for item in args.set:
        k, _, v = item.partition("=")
        try:
            manual[k] = json.loads(v)
        except json.JSONDecodeError:
            manual[k] = v
    if manual:
        cfg = apply_overrides(cfg, manual)
        cfg.profile_name = f"{args.profile}+" + ",".join(f"{k}={v}" for k, v in manual.items())

    replay(args.walk, cfg, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())

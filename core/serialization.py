# -*- coding: utf-8 -*-
"""
Сериализация Tick в JSONL и обратно.

ЗАЧЕМ ЯВНЫЕ ПРЕОБРАЗОВАТЕЛИ, А НЕ asdict()
------------------------------------------
dataclasses.asdict() не справляется с numpy-массивами внутри RiskField
и LocalGuidance, а обратного преобразования не даёт вовсе. Кроме того,
запись прогулки — это десятки тысяч строк, и контроль над точностью
чисел сокращает файл в разы без потери смысла: широта с семью знаками
после запятой различает сантиметры, а угол с двумя — сотые доли градуса,
чего более чем достаточно.

Явные преобразователи дают ещё одно: при изменении контракта старые
записи не «молча читаются неправильно», а падают на отсутствующем поле.
Для данных, на которых строятся выводы статьи, это желаемое поведение.

ФОРМАТ
------
Одна строка JSONL = один Tick. Файл может быть сжат gzip: записи
прогулок сжимаются примерно в десять раз и не мешают версионированию.

ВЕРСИЯ ФОРМАТА
--------------
FORMAT_VERSION пишется в заголовочную строку файла. При несовпадении
чтение отказывает явно: пересчитывать метрики статьи по записи,
собранной другим форматом, нельзя.
"""

from __future__ import annotations

import gzip
import json
from typing import Iterable, Iterator, Optional

import numpy as np

from core.types import (
    Corridor, CrossingPhase, Detection, DistanceSource, FusionResult,
    GlobalGuidance, HealthFlag, Lang, LocalGuidance, Maneuver, ManeuverType,
    Pose, PoseSource, RiskField, Route, SystemHealth, Tick, Urgency, Utterance,
    Action,
)

FORMAT_VERSION = "1.0"

# Точность: подобрана так, чтобы не терять смысл и не раздувать файл
P_COORD = 7      # ~1 см
P_ANGLE = 2      # сотые градуса
P_DIST = 3       # миллиметры
P_RISK = 4
P_TIME = 3       # миллисекунды


def _r(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _arr(a: Optional[np.ndarray], digits: int) -> Optional[list]:
    return None if a is None else np.round(np.asarray(a, dtype=float), digits).tolist()


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def pose_to_dict(p: Pose) -> dict:
    return {
        "lat": _r(p.lat, P_COORD),
        "lon": _r(p.lon, P_COORD),
        "heading_deg": _r(p.heading_deg, P_ANGLE),
        "ts": _r(p.ts, P_TIME),
        "accuracy_m": _r(p.accuracy_m, P_DIST),
        "heading_sigma_deg": _r(p.heading_sigma_deg, P_ANGLE),
        "speed_mps": _r(p.speed_mps, P_DIST),
        "source": p.source.value,
    }


def pose_from_dict(d: dict) -> Pose:
    return Pose(
        lat=d["lat"], lon=d["lon"], heading_deg=d["heading_deg"], ts=d["ts"],
        accuracy_m=d["accuracy_m"], heading_sigma_deg=d["heading_sigma_deg"],
        speed_mps=d["speed_mps"], source=PoseSource(d["source"]),
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detection_to_dict(x: Detection) -> dict:
    return {
        "label": x.label,
        "confidence": _r(x.confidence, P_RISK),
        "bbox": [int(v) for v in x.bbox],
        "bearing_deg": _r(x.bearing_deg, P_ANGLE),
        "distance_m": _r(x.distance_m, P_DIST),
        "distance_source": x.distance_source.value,
        "distance_sigma_m": _r(x.distance_sigma_m, P_DIST),
        "track_id": x.track_id,
        "radial_velocity_mps": _r(x.radial_velocity_mps, P_DIST),
        "ts": _r(x.ts, P_TIME),
    }


def detection_from_dict(d: dict) -> Detection:
    return Detection(
        label=d["label"], confidence=d["confidence"], bbox=tuple(d["bbox"]),
        bearing_deg=d["bearing_deg"], distance_m=d["distance_m"],
        distance_source=DistanceSource(d["distance_source"]),
        distance_sigma_m=d["distance_sigma_m"], track_id=d["track_id"],
        radial_velocity_mps=d["radial_velocity_mps"], ts=d["ts"],
    )


# ---------------------------------------------------------------------------
# RiskField
# ---------------------------------------------------------------------------

def risk_to_dict(r: Optional[RiskField]) -> Optional[dict]:
    if r is None:
        return None
    return {
        "cells": _arr(r.cells, P_RISK),
        "sector_edges_deg": _arr(r.sector_edges_deg, P_ANGLE),
        "band_edges_m": _arr(r.band_edges_m, P_DIST),
        "ts": _r(r.ts, P_TIME),
        # ключи вида "сектор,полоса" — JSON не допускает кортежей в ключах
        "contributors": {str(k): list(v) for k, v in r.contributors.items()},
    }


def risk_from_dict(d: Optional[dict]) -> Optional[RiskField]:
    if d is None:
        return None
    return RiskField(
        cells=np.asarray(d["cells"], dtype=float),
        sector_edges_deg=np.asarray(d["sector_edges_deg"], dtype=float),
        band_edges_m=np.asarray(d["band_edges_m"], dtype=float),
        ts=d["ts"],
        contributors={k: list(v) for k, v in d["contributors"].items()},
    )


# ---------------------------------------------------------------------------
# Маршрут
# ---------------------------------------------------------------------------

def maneuver_to_dict(m: Maneuver) -> dict:
    return {
        "type": m.type.value,
        "lat": _r(m.lat, P_COORD),
        "lon": _r(m.lon, P_COORD),
        "exit_bearing_deg": _r(m.exit_bearing_deg, P_ANGLE),
        "distance_from_prev_m": _r(m.distance_from_prev_m, P_DIST),
        "landmark": m.landmark,
        "accessibility": dict(m.accessibility),
    }


def maneuver_from_dict(d: dict) -> Maneuver:
    return Maneuver(
        type=ManeuverType(d["type"]), lat=d["lat"], lon=d["lon"],
        exit_bearing_deg=d["exit_bearing_deg"],
        distance_from_prev_m=d["distance_from_prev_m"],
        landmark=d["landmark"], accessibility=d["accessibility"],
    )


def route_to_dict(r: Route) -> dict:
    """Маршрут пишется один раз в заголовок файла, а не в каждый Tick."""
    return {
        "maneuvers": [maneuver_to_dict(m) for m in r.maneuvers],
        "total_distance_m": _r(r.total_distance_m, P_DIST),
        "total_cost": _r(r.total_cost, P_DIST),
        "polyline": [[_r(a, P_COORD), _r(b, P_COORD)] for a, b in r.polyline],
        "profile": dict(r.profile),
    }


def route_from_dict(d: dict) -> Route:
    return Route(
        maneuvers=[maneuver_from_dict(m) for m in d["maneuvers"]],
        total_distance_m=d["total_distance_m"], total_cost=d["total_cost"],
        polyline=[tuple(p) for p in d["polyline"]], profile=d["profile"],
    )


def global_to_dict(g: Optional[GlobalGuidance]) -> Optional[dict]:
    if g is None:
        return None
    return {
        "theta_route_deg": _r(g.theta_route_deg, P_ANGLE),
        "next_maneuver": maneuver_to_dict(g.next_maneuver) if g.next_maneuver else None,
        "distance_to_maneuver_m": _r(g.distance_to_maneuver_m, P_DIST),
        "cross_track_error_m": _r(g.cross_track_error_m, P_DIST),
        "off_route": bool(g.off_route),
        "arrived": bool(g.arrived),
    }


def global_from_dict(d: Optional[dict]) -> Optional[GlobalGuidance]:
    if d is None:
        return None
    nm = d["next_maneuver"]
    return GlobalGuidance(
        theta_route_deg=d["theta_route_deg"],
        next_maneuver=maneuver_from_dict(nm) if nm else None,
        distance_to_maneuver_m=d["distance_to_maneuver_m"],
        cross_track_error_m=d["cross_track_error_m"],
        off_route=d["off_route"], arrived=d["arrived"],
    )


# ---------------------------------------------------------------------------
# Локальный уровень
# ---------------------------------------------------------------------------

def corridor_to_dict(c: Corridor) -> dict:
    return {
        "center_deg": _r(c.center_deg, P_ANGLE),
        "left_deg": _r(c.left_deg, P_ANGLE),
        "right_deg": _r(c.right_deg, P_ANGLE),
        "clearance_m": _r(c.clearance_m, P_DIST),
        "mean_risk": _r(c.mean_risk, P_RISK),
    }


def corridor_from_dict(d: dict) -> Corridor:
    return Corridor(**d)


def local_to_dict(l: Optional[LocalGuidance]) -> Optional[dict]:
    if l is None:
        return None
    clearance = l.clearance_m
    return {
        "risk_profile": _arr(l.risk_profile, P_RISK),
        "profile_angles_deg": _arr(l.profile_angles_deg, P_ANGLE),
        "corridors": [corridor_to_dict(c) for c in l.corridors],
        "theta_free_deg": _r(l.theta_free_deg, P_ANGLE),
        "corridor_width_deg": _r(l.corridor_width_deg, P_ANGLE),
        # inf не входит в стандарт JSON — пишем null и восстанавливаем при чтении
        "clearance_m": None if clearance == float("inf") else _r(clearance, P_DIST),
        "blocking_indices": [int(i) for i in l.blocking] if l.blocking else [],
        "blocked": bool(l.blocked),
    }


def local_from_dict(d: Optional[dict]) -> Optional[LocalGuidance]:
    if d is None:
        return None
    return LocalGuidance(
        risk_profile=np.asarray(d["risk_profile"], dtype=float),
        profile_angles_deg=np.asarray(d["profile_angles_deg"], dtype=float),
        corridors=[corridor_from_dict(c) for c in d["corridors"]],
        theta_free_deg=d["theta_free_deg"],
        corridor_width_deg=d["corridor_width_deg"],
        clearance_m=float("inf") if d["clearance_m"] is None else d["clearance_m"],
        blocking=list(d["blocking_indices"]),
        blocked=d["blocked"],
    )


# ---------------------------------------------------------------------------
# Здоровье и слияние
# ---------------------------------------------------------------------------

def health_to_dict(h: SystemHealth) -> dict:
    return {
        "flags": [f.value if isinstance(f, HealthFlag) else str(f) for f in h.flags],
        "confidence": _r(h.confidence, P_RISK),
        "degraded_since_ts": _r(h.degraded_since_ts, P_TIME),
        "detail": h.detail,
    }


def health_from_dict(d: dict) -> SystemHealth:
    return SystemHealth(
        flags=[HealthFlag(f) for f in d["flags"]],
        confidence=d["confidence"],
        degraded_since_ts=d["degraded_since_ts"],
        detail=d["detail"],
    )


def fusion_to_dict(f: Optional[FusionResult]) -> Optional[dict]:
    if f is None:
        return None
    return {
        "action": f.action.value,
        "theta_star_deg": _r(f.theta_star_deg, P_ANGLE),
        "urgency": int(f.urgency),
        "deviation_from_route_deg": _r(f.deviation_from_route_deg, P_ANGLE),
        "reason": f.reason,
        "cause_indices": [int(i) for i in f.cause_indices],
        "replan_needed": bool(f.replan_needed),
        "crossing_phase": f.crossing_phase.value if f.crossing_phase else None,
        "cost_breakdown": {k: _r(v, P_RISK) for k, v in f.cost_breakdown.items()},
    }


def fusion_from_dict(d: Optional[dict]) -> Optional[FusionResult]:
    if d is None:
        return None
    return FusionResult(
        action=Action(d["action"]), theta_star_deg=d["theta_star_deg"],
        urgency=Urgency(d["urgency"]),
        deviation_from_route_deg=d["deviation_from_route_deg"],
        reason=d["reason"], cause_indices=list(d["cause_indices"]),
        replan_needed=d["replan_needed"],
        crossing_phase=CrossingPhase(d["crossing_phase"]) if d["crossing_phase"] else None,
        cost_breakdown=d["cost_breakdown"],
    )


def utterance_to_dict(u: Optional[Utterance]) -> Optional[dict]:
    if u is None:
        return None
    return {
        "text": u.text, "lang": u.lang.value, "urgency": int(u.urgency),
        "ts": _r(u.ts, P_TIME), "ttl_s": _r(u.ttl_s, P_TIME),
        "dedup_key": u.dedup_key, "haptic": u.haptic,
    }


def utterance_from_dict(d: Optional[dict]) -> Optional[Utterance]:
    if d is None:
        return None
    return Utterance(
        text=d["text"], lang=Lang(d["lang"]), urgency=Urgency(d["urgency"]),
        ts=d["ts"], ttl_s=d["ttl_s"], dedup_key=d["dedup_key"], haptic=d["haptic"],
    )


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

def tick_to_dict(t: Tick) -> dict:
    return {
        "seq": int(t.seq),
        "ts": _r(t.ts, P_TIME),
        "pose": pose_to_dict(t.pose),
        "health": health_to_dict(t.health),
        "detections": [detection_to_dict(x) for x in t.detections],
        "risk": risk_to_dict(t.risk),
        "global_guidance": global_to_dict(t.global_guidance),
        "local_guidance": local_to_dict(t.local_guidance),
        "fusion": fusion_to_dict(t.fusion),
        "utterance": utterance_to_dict(t.utterance),
        "timings_ms": {k: _r(v, 2) for k, v in t.timings_ms.items()},
    }


def tick_from_dict(d: dict) -> Tick:
    return Tick(
        seq=d["seq"], ts=d["ts"],
        pose=pose_from_dict(d["pose"]),
        health=health_from_dict(d["health"]),
        detections=[detection_from_dict(x) for x in d["detections"]],
        risk=risk_from_dict(d["risk"]),
        global_guidance=global_from_dict(d["global_guidance"]),
        local_guidance=local_from_dict(d["local_guidance"]),
        fusion=fusion_from_dict(d["fusion"]),
        utterance=utterance_from_dict(d["utterance"]),
        timings_ms=d["timings_ms"],
    )


# ---------------------------------------------------------------------------
# Файлы
# ---------------------------------------------------------------------------

def _open(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class TickWriter:
    """Пишет заголовок и поток Tick'ов.

    Заголовок содержит конфигурацию прогона и маршрут: без них лог
    невозможно интерпретировать, а хранить их в каждой строке
    расточительно. Именно заголовок делает запись самодостаточной —
    результат воспроизводится по одному файлу.
    """

    def __init__(self, path: str, config: dict, route: Optional[Route] = None,
                 meta: Optional[dict] = None):
        self.path = path
        self._fh = _open(path, "w")
        header = {
            "_format_version": FORMAT_VERSION,
            "config": config,
            "route": route_to_dict(route) if route else None,
            "meta": meta or {},
        }
        self._fh.write(json.dumps(header, ensure_ascii=False) + "\n")

    def write(self, tick: Tick) -> None:
        self._fh.write(json.dumps(tick_to_dict(tick), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TickWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_ticks(path: str) -> tuple:
    """-> (header: dict, ticks: Iterator[Tick])

    Заголовок читается сразу, Tick'и — лениво: запись получасовой
    прогулки не обязана помещаться в память целиком.
    """
    fh = _open(path, "r")
    header = json.loads(fh.readline())

    version = header.get("_format_version")
    if version != FORMAT_VERSION:
        fh.close()
        raise ValueError(
            f"Формат записи {version} не совпадает с текущим {FORMAT_VERSION}. "
            "Пересчитывать метрики по такой записи нельзя."
        )

    def gen() -> Iterator[Tick]:
        try:
            for line in fh:
                line = line.strip()
                if line:
                    yield tick_from_dict(json.loads(line))
        finally:
            fh.close()

    return header, gen()


def write_ticks(path: str, ticks: Iterable, config: dict,
                route: Optional[Route] = None, meta: Optional[dict] = None) -> int:
    n = 0
    with TickWriter(path, config, route, meta) as w:
        for t in ticks:
            w.write(t)
            n += 1
    return n

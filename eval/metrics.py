# -*- coding: utf-8 -*-
"""
Метрики для статьи. Считаются из JSONL-лога Tick'ов и ручной разметки.

ПОЧЕМУ ОПРЕДЕЛЕНИЯ ВАЖНЕЕ КОДА
------------------------------
Каждая метрика ниже — это утверждение о том, что считать успехом.
Неверно выбранная метрика даёт систему, оптимизированную не туда,
и обнаруживается это уже после сбора данных. На шаге 3 мы это
проходили: подсчёт переходов штуками объявлял маршрут по осевой
линии проезжей части самым безопасным.

Поэтому каждая метрика здесь снабжена оговоркой, чего она НЕ измеряет.

ФОРМАТ РУЧНОЙ РАЗМЕТКИ (data/walks/<id>/hazards.jsonl)
------------------------------------------------------
    {"ts": 123.5,                   момент, когда опасность видна в кадре
     "seq": 2470,                   номер кадра в video.mp4
     "type": "pothole",
     "severity": "danger",          danger | obstacle
     "bearing_deg": -12.0,          где относительно направления взгляда
     "distance_m": 3.5,             оценка размечающего
     "warn_by_ts": 122.0,           ПОЗЖЕ ЭТОГО предупреждать бесполезно
     "note": "открытый люк у края"}

warn_by_ts — ключевое поле. Предупреждение, прозвучавшее после него,
формально относится к опасности, но человек уже сделал шаг. Считать
такое успехом значит обманывать себя.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Iterable, Optional

from core.serialization import read_ticks
from core.types import Action, Tick, Urgency

#: Насколько угол реплики должен совпасть с углом опасности, градусы.
#: Шире сектора поля риска: важно, что предупредили про то самое,
#: а не про соседнее.
MATCH_ANGLE_DEG = 25.0

#: Окно вокруг момента опасности, в котором предупреждение считается
#: относящимся к ней, с. Раньше — предвидение, позже — опоздание.
MATCH_WINDOW_BEFORE_S = 8.0
MATCH_WINDOW_AFTER_S = 1.0

#: Две подсказки противоположного направления ближе этого интервала —
#: противоречие. Человек не успевает отработать первую до отмены.
CONTRADICTION_WINDOW_S = 1.5

#: Скорость речи для оценки занятости канала, слов в секунду
SPEECH_WORDS_PER_S = 2.6


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load_hazards(walk_dir: str) -> list:
    path = os.path.join(walk_dir, "hazards.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                out.append(json.loads(line))
    return out


def load_walk(walk_dir: str) -> tuple:
    """-> (header, list[Tick], list[hazard])"""
    ticks_path = None
    for name in ("ticks.jsonl.gz", "ticks.jsonl"):
        p = os.path.join(walk_dir, name)
        if os.path.exists(p):
            ticks_path = p
            break
    if ticks_path is None:
        raise FileNotFoundError(f"нет ticks.jsonl(.gz) в {walk_dir}")

    header, stream = read_ticks(ticks_path)
    return header, list(stream), load_hazards(walk_dir)


# ---------------------------------------------------------------------------
# Безопасность
# ---------------------------------------------------------------------------

def _utterances(ticks: list) -> list:
    return [t for t in ticks if t.utterance is not None]


def _warning_bearing(tick: Tick) -> Optional[float]:
    """Угол, о котором предупреждала реплика, относительно взгляда.

    Берётся у объекта-причины: именно он породил предупреждение.
    Если причина не указана (например, «стоп»), считаем, что реплика
    относится к направлению прямо по курсу.
    """
    f = tick.fusion
    if f is None:
        return 0.0
    for i in f.cause_indices:
        if 0 <= i < len(tick.detections):
            return tick.detections[i].bearing_deg
    return 0.0


def hazard_metrics(ticks: list, hazards: list) -> dict:
    """Полнота, точность и своевременность предупреждений.

    ЧТО ЭТО НЕ ИЗМЕРЯЕТ
    Не измеряет, избежал ли человек опасности: система могла
    предупредить, а человек не успеть. Это разные вещи, и вторая
    требует участия людей (шаг 7).
    """
    warned = [t for t in _utterances(ticks)
              if t.utterance.urgency >= Urgency.CAUTION]

    matched_hazards = set()
    matched_warnings = set()
    lead_times = []
    late = 0

    for hi, hz in enumerate(hazards):
        hz_ts = float(hz["ts"])
        hz_bearing = float(hz.get("bearing_deg", 0.0))
        warn_by = float(hz.get("warn_by_ts", hz_ts))

        best = None
        for wi, t in enumerate(warned):
            if wi in matched_warnings:
                continue
            dt = hz_ts - t.ts
            if not (-MATCH_WINDOW_AFTER_S <= dt <= MATCH_WINDOW_BEFORE_S):
                continue
            wb = _warning_bearing(t)
            if wb is not None and abs(wb - hz_bearing) > MATCH_ANGLE_DEG:
                continue
            if best is None or t.ts < warned[best].ts:
                best = wi

        if best is not None:
            matched_hazards.add(hi)
            matched_warnings.add(best)
            lead_times.append(warn_by - warned[best].ts)
            if warned[best].ts > warn_by:
                late += 1

    n_hz = len(hazards)
    n_warn = len(warned)
    critical = [h for h in hazards if h.get("severity") == "danger"]
    missed_critical = sum(
        1 for hi, h in enumerate(hazards)
        if h.get("severity") == "danger" and hi not in matched_hazards
    )

    lead_sorted = sorted(lead_times)
    return {
        "hazards_total": n_hz,
        "hazards_critical": len(critical),
        "warnings_total": n_warn,
        "hazard_recall": round(len(matched_hazards) / n_hz, 4) if n_hz else None,
        "hazard_precision": round(len(matched_warnings) / n_warn, 4) if n_warn else None,
        "missed_critical": missed_critical,
        # Запас времени: положительный — успели, отрицательный — опоздали
        "lead_time_median_s": round(lead_sorted[len(lead_sorted) // 2], 2) if lead_sorted else None,
        "lead_time_min_s": round(lead_sorted[0], 2) if lead_sorted else None,
        "warnings_after_deadline": late,
    }


# ---------------------------------------------------------------------------
# Навигация
# ---------------------------------------------------------------------------

def navigation_metrics(ticks: list) -> dict:
    """Насколько точно система вела по маршруту.

    ЧТО ЭТО НЕ ИЗМЕРЯЕТ
    Отклонение считается от маршрута, который система СЧИТАЛА верным.
    Если маршрут был перестроен в обход, отклонение обнулится, хотя
    человек прошёл лишние сто метров. Поэтому рядом приводится
    число перестроений и пройденный путь.
    """
    cte = [t.global_guidance.cross_track_error_m for t in ticks
           if t.global_guidance is not None]
    off = [t for t in ticks if t.global_guidance and t.global_guidance.off_route]
    replans = sum(1 for t in ticks if t.fusion and t.fusion.replan_needed)
    arrived = any(t.global_guidance and t.global_guidance.arrived for t in ticks)

    # пройденный путь по позициям
    dist = 0.0
    for a, b in zip(ticks, ticks[1:]):
        dist += _haversine(a.pose.lat, a.pose.lon, b.pose.lat, b.pose.lon)

    stops = sum(1 for t in ticks if t.fusion and t.fusion.action is Action.STOP)
    deviations = [abs(t.fusion.deviation_from_route_deg) for t in ticks
                  if t.fusion is not None]

    return {
        "arrived": arrived,
        "distance_walked_m": round(dist, 1),
        "cross_track_rmse_m": round(math.sqrt(sum(x * x for x in cte) / len(cte)), 2) if cte else None,
        "cross_track_max_m": round(max(cte), 2) if cte else None,
        "off_route_fraction": round(len(off) / len(ticks), 4) if ticks else None,
        "replan_count": replans,
        "stop_commands": stops,
        "mean_deviation_deg": round(sum(deviations) / len(deviations), 2) if deviations else None,
    }


# ---------------------------------------------------------------------------
# Нагрузка на пользователя
# ---------------------------------------------------------------------------

_OPPOSITE = {
    (Action.BEAR_LEFT, Action.BEAR_RIGHT), (Action.BEAR_RIGHT, Action.BEAR_LEFT),
    (Action.TURN_LEFT, Action.TURN_RIGHT), (Action.TURN_RIGHT, Action.TURN_LEFT),
}


def load_metrics(ticks: list) -> dict:
    """Сколько система говорила и насколько связно.

    ЧТО ЭТО НЕ ИЗМЕРЯЕТ
    Не измеряет субъективную нагрузку: две реплики в минуту могут
    раздражать, а десять — успокаивать, в зависимости от их полезности.
    Это выясняется только с людьми (шаг 7).
    """
    utts = _utterances(ticks)
    if not ticks:
        return {}

    duration_s = max(ticks[-1].ts - ticks[0].ts, 1e-6)
    words = sum(len(t.utterance.text.split()) for t in utts)
    speech_s = words / SPEECH_WORDS_PER_S

    contradictions = 0
    for a, b in zip(utts, utts[1:]):
        if b.ts - a.ts > CONTRADICTION_WINDOW_S:
            continue
        fa = a.fusion.action if a.fusion else None
        fb = b.fusion.action if b.fusion else None
        if (fa, fb) in _OPPOSITE:
            contradictions += 1

    by_urgency = defaultdict(int)
    for t in utts:
        by_urgency[t.utterance.urgency.name] += 1

    gaps = [b.ts - a.ts for a, b in zip(utts, utts[1:])]
    return {
        "duration_s": round(duration_s, 1),
        "utterances_total": len(utts),
        "utterances_per_min": round(len(utts) / (duration_s / 60.0), 2),
        # Доля времени, в течение которой канал слуха занят системой.
        # Для незрячего это прямая цена: слух — основной канал ориентации.
        "speech_duty_cycle": round(speech_s / duration_s, 4),
        "contradiction_count": contradictions,
        "contradiction_rate_per_min": round(contradictions / (duration_s / 60.0), 3),
        "by_urgency": dict(by_urgency),
        "min_gap_s": round(min(gaps), 2) if gaps else None,
        "median_gap_s": round(sorted(gaps)[len(gaps) // 2], 2) if gaps else None,
    }


# ---------------------------------------------------------------------------
# Производительность
# ---------------------------------------------------------------------------

def performance_metrics(ticks: list) -> dict:
    if not ticks:
        return {}
    totals = sorted(t.timings_ms.get("total_ms", 0.0) for t in ticks)
    n = len(totals)

    layers = defaultdict(list)
    for t in ticks:
        for k, v in t.timings_ms.items():
            if k != "total_ms":
                layers[k].append(v)

    out = {
        "frames": n,
        "latency_p50_ms": round(totals[n // 2], 1),
        "latency_p95_ms": round(totals[int(n * 0.95) - 1], 1),
        "latency_max_ms": round(totals[-1], 1),
        "fps_equivalent": round(1000.0 / totals[n // 2], 1) if totals[n // 2] else None,
        "by_layer_median_ms": {
            k: round(sorted(v)[len(v) // 2], 2) for k, v in sorted(layers.items())
        },
    }

    degraded = sum(1 for t in ticks if t.health.degraded)
    out["degraded_fraction"] = round(degraded / n, 4)
    flags = defaultdict(int)
    for t in ticks:
        for f in t.health.flags:
            flags[f.value] += 1
    out["health_flags"] = dict(flags)
    return out


# ---------------------------------------------------------------------------
# Всё вместе
# ---------------------------------------------------------------------------

def evaluate(walk_dir: str) -> dict:
    header, ticks, hazards = load_walk(walk_dir)
    return {
        "walk": os.path.basename(walk_dir.rstrip(os.sep)),
        "meta": header.get("meta", {}),
        "config_profile": (header.get("config") or {}).get("profile_name"),
        "safety": hazard_metrics(ticks, hazards),
        "navigation": navigation_metrics(ticks),
        "load": load_metrics(ticks),
        "performance": performance_metrics(ticks),
    }


def evaluate_many(walk_dirs: Iterable) -> dict:
    """Свод по нескольким прогулкам.

    Усредняются НЕ доли, а исходные счётчики: среднее по долям
    придаёт короткой прогулке тот же вес, что и длинной.
    """
    per_walk = [evaluate(d) for d in walk_dirs]
    if not per_walk:
        return {"walks": 0}

    hz_total = sum(w["safety"]["hazards_total"] or 0 for w in per_walk)
    hz_found = sum(round((w["safety"]["hazard_recall"] or 0)
                         * (w["safety"]["hazards_total"] or 0)) for w in per_walk)
    warn_total = sum(w["safety"]["warnings_total"] or 0 for w in per_walk)
    warn_true = sum(round((w["safety"]["hazard_precision"] or 0)
                          * (w["safety"]["warnings_total"] or 0)) for w in per_walk)
    dur = sum(w["load"].get("duration_s", 0) for w in per_walk)
    utts = sum(w["load"].get("utterances_total", 0) for w in per_walk)

    return {
        "walks": len(per_walk),
        "total_duration_s": round(dur, 1),
        "hazard_recall": round(hz_found / hz_total, 4) if hz_total else None,
        "hazard_precision": round(warn_true / warn_total, 4) if warn_total else None,
        "missed_critical": sum(w["safety"]["missed_critical"] for w in per_walk),
        "utterances_per_min": round(utts / (dur / 60.0), 2) if dur else None,
        "arrived_rate": round(sum(1 for w in per_walk if w["navigation"]["arrived"])
                              / len(per_walk), 4),
        "replan_total": sum(w["navigation"]["replan_count"] for w in per_walk),
        "per_walk": per_walk,
    }


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

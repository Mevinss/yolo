# -*- coding: utf-8 -*-
"""
Последовательность узлов -> человекопонятные манёвры.

Отдельный слой, потому что «повернуть направо» — не свойство графа,
а интерпретация изменения азимута между рёбрами.

ОСОБЫЙ СЛУЧАЙ: ПЕШЕХОДНЫЙ ПЕРЕХОД
---------------------------------
Для зрячего переход — обычная точка маршрута. Для незрячего это самый
опасный момент пути, требующий отдельного протокола:

  1. заранее:   «через 15 метров переход, регулируемый»
  2. на месте:  «переход. Ждите сигнала» / «переход нерегулируемый, будьте осторожны»
  3. подтверждение начала движения пользователем
  4. во время:  удержание курса — здесь слияние особенно важно,
                потому что сойти с курса на проезжей части опаснее всего

Поэтому ManeuverType.CROSSING обрабатывается не как поворот, а как
отдельное состояние в fusion (шаг 5). Здесь мы его только распознаём
и помечаем, вместе с тегами доступности, которые понадобятся для фразы.

ПОРОГИ УГЛОВ
------------
Пороги подобраны под восприятие без зрения: человек, идущий с тростью,
надёжно отличает «чуть довернуть» от «повернуть», но не различает
15 и 25 градусов. Поэтому категорий немного и они широкие.
"""

from __future__ import annotations

import math

from core.types import Maneuver, ManeuverType

#: Изменение курса меньше этого — не манёвр, а изгиб улицы
STRAIGHT_THRESHOLD_DEG = 20.0
#: Между этим и предыдущим — «слегка левее/правее»
SLIGHT_THRESHOLD_DEG = 50.0

#: Теги доступности, которые тащим в манёвр для построения фразы
ACCESSIBILITY_KEYS = (
    "crossing", "crossing_ref", "crossing:markings", "traffic_signals:sound",
    "tactile_paving", "kerb", "handrail", "step_count", "lit", "surface",
)


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Азимут от точки 1 к точке 2, градусы (0 = север, по часовой)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по большому кругу, метры."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def angle_diff(a: float, b: float) -> float:
    """Кратчайшая разность азимутов в диапазоне [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def _edge_bearing(G, u, v, at_start: bool) -> float:
    """Азимут ребра у одного из его концов.

    Берём не хорду между узлами, а первые (или последние) точки геометрии:
    длинное изогнутое ребро может уходить от перекрёстка совсем не туда,
    куда указывает прямая на его дальний конец.
    """
    geom = G[u][v]["geometry"]
    if haversine_m(*geom[0], G.nodes[u]["lat"], G.nodes[u]["lon"]) > \
       haversine_m(*geom[-1], G.nodes[u]["lat"], G.nodes[u]["lon"]):
        geom = list(reversed(geom))
    if at_start:
        p, q = geom[0], geom[min(1, len(geom) - 1)]
    else:
        p, q = geom[max(0, len(geom) - 2)], geom[-1]
    return bearing(p[0], p[1], q[0], q[1])


def _classify(delta: float) -> ManeuverType:
    if abs(delta) < STRAIGHT_THRESHOLD_DEG:
        return ManeuverType.STRAIGHT
    if abs(delta) < SLIGHT_THRESHOLD_DEG:
        return ManeuverType.SLIGHT_RIGHT if delta > 0 else ManeuverType.SLIGHT_LEFT
    return ManeuverType.TURN_RIGHT if delta > 0 else ManeuverType.TURN_LEFT


def _special_type(tags: dict) -> ManeuverType | None:
    """Переход и лестница важнее поворота: о них надо сказать всегда,
    даже если геометрически идём прямо."""
    if tags.get("highway") == "steps":
        incline = str(tags.get("incline", "")).strip()
        if incline.startswith("-") or incline == "down":
            return ManeuverType.STAIRS_DOWN
        return ManeuverType.STAIRS_UP
    if tags.get("footway") == "crossing" or "crossing" in tags:
        return ManeuverType.CROSSING
    return None


def nodes_to_maneuvers(path: list, G) -> list:
    """Путь в графе -> список манёвров."""
    if len(path) < 2:
        return []

    maneuvers: list = []
    n0 = G.nodes[path[0]]
    maneuvers.append(Maneuver(
        type=ManeuverType.DEPART,
        lat=n0["lat"], lon=n0["lon"],
        exit_bearing_deg=_edge_bearing(G, path[0], path[1], at_start=True),
        distance_from_prev_m=0.0,
    ))

    accumulated = 0.0

    for k in range(1, len(path) - 1):
        prev_n, cur_n, next_n = path[k - 1], path[k], path[k + 1]
        e_in = G[prev_n][cur_n]
        e_out = G[cur_n][next_n]
        accumulated += e_in["length_m"]

        b_in = _edge_bearing(G, prev_n, cur_n, at_start=False)
        b_out = _edge_bearing(G, cur_n, next_n, at_start=True)
        delta = angle_diff(b_out, b_in)

        mtype = _special_type(e_out["tags"]) or _classify(delta)
        if mtype == ManeuverType.STRAIGHT:
            continue                       # изгиб улицы — не событие для пользователя

        node = G.nodes[cur_n]
        tags = dict(e_out["tags"])
        tags.update(node.get("tags") or {})     # теги узла (kerb, tactile) важнее

        maneuvers.append(Maneuver(
            type=mtype,
            lat=node["lat"], lon=node["lon"],
            exit_bearing_deg=b_out,
            distance_from_prev_m=accumulated,
            landmark=tags.get("name"),
            accessibility={k_: v for k_, v in tags.items() if k_ in ACCESSIBILITY_KEYS},
        ))
        accumulated = 0.0

    accumulated += G[path[-2]][path[-1]]["length_m"]
    nl = G.nodes[path[-1]]
    maneuvers.append(Maneuver(
        type=ManeuverType.ARRIVE,
        lat=nl["lat"], lon=nl["lon"],
        exit_bearing_deg=_edge_bearing(G, path[-2], path[-1], at_start=False),
        distance_from_prev_m=accumulated,
    ))
    return maneuvers

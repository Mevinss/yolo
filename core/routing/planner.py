# -*- coding: utf-8 -*-
"""
Глобальный маршрутизатор: A* по пешеходному графу с весами доступности.

ДВЕ ФУНКЦИИ С РАЗНЫМИ ТРЕБОВАНИЯМИ
----------------------------------
plan()   вызывается редко (старт и перестроения) — может думать десятки мс.
track()  вызывается каждый кадр, ~20 раз в секунду — обязана быть быстрой.

Поэтому track() ничего не пересчитывает: только проецирует позицию
на заранее подготовленную полилинию (векторизованно, numpy) и ищет
ближайший манёвр впереди.

ПОЧЕМУ НЕ ВНЕШНИЙ СЕРВИС
------------------------
  1. OSRM/GraphHopper/Google не умеют весов доступности — а это ядро вклада;
  2. офлайн обязателен: связь в городе рвётся, и это вопрос безопасности;
  3. воспроизводимость: внешний сервис меняет ответы, и через год
     результаты статьи не повторить.

ДОПУСТИМОСТЬ ЭВРИСТИКИ
----------------------
A* даёт оптимум только при эвристике, не завышающей остаток пути.
Стоимость = длина * множитель, множители бывают меньше единицы
(тактильная плитка = 0.7), поэтому эвристика — расстояние по прямой,
умноженное на МИНИМАЛЬНЫЙ множитель, реально встречающийся в графе.
Брать просто расстояние было бы ошибкой: на участках с плиткой
эвристика завысила бы остаток и A* вернул бы неоптимальный маршрут.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np

from core.config import RoutingConfig
from core.routing.maneuvers import bearing, haversine_m, nodes_to_maneuvers
from core.types import GlobalGuidance, ManeuverType, Pose, Route

M_PER_DEG_LAT = 111_320.0


class Router:
    def __init__(self, config: RoutingConfig, graph, weights):
        self.config = config
        self.graph = graph
        self.weights = weights
        self._min_multiplier = self._compute_min_multiplier()
        # кэш подготовленной полилинии для track()
        self._track_cache: dict = {}

    def _compute_min_multiplier(self) -> float:
        G = getattr(self.graph, "G", None)
        if G is None or G.number_of_edges() == 0:
            return 0.5
        return min(d["multiplier"] for _, _, d in G.edges(data=True))

    # -----------------------------------------------------------------
    # Состав маршрута
    # -----------------------------------------------------------------

    def _profile(self, path: list) -> dict:
        """Разложить маршрут по типам инфраструктуры, метры.

        ПОЧЕМУ НЕ СЧИТАТЬ ПЕРЕХОДЫ ШТУКАМИ
        Маршрут по проезжей части не содержит рёбер crossing вообще:
        идущий по осевой линии нигде формально не «переходит» улицу,
        хотя пересекает поток постоянно. Подсчёт штук делает такой
        маршрут самым безопасным на бумаге — прямо наоборот истине.
        Поэтому основная метрика здесь — доля длины по типам покрытия,
        а переходы считаются только в пределах пешеходной сети.
        """
        G = self.graph.G
        prof: dict = defaultdict(float)
        counts: dict = defaultdict(int)

        for a, b in zip(path, path[1:]):
            d = G[a][b]
            tags = d["tags"]
            length = d["length_m"]
            hw = tags.get("highway", "")
            fw = tags.get("footway", "")

            if fw == "crossing" or "crossing" in tags:
                prof["crossing_m"] += length
                counts["crossings"] += 1
                if tags.get("crossing") in ("traffic_signals", "signals"):
                    counts["crossings_signalled"] += 1
                    if tags.get("traffic_signals:sound") in ("yes", "walk", "always"):
                        counts["crossings_with_audio"] += 1
                elif tags.get("crossing") in ("marked", "zebra", "uncontrolled"):
                    counts["crossings_marked_unsignalled"] += 1
                else:
                    counts["crossings_untagged_or_unmarked"] += 1
            elif hw == "steps":
                prof["steps_m"] += length
                counts["stairs"] += 1
                if tags.get("handrail") not in ("yes", "both", "left", "right"):
                    counts["stairs_no_handrail"] += 1
            elif hw == "footway" and fw == "sidewalk":
                prof["sidewalk_m"] += length
            elif hw in ("footway", "path", "pedestrian", "corridor"):
                prof["footway_m"] += length
            elif hw == "living_street":
                prof["living_street_m"] += length
            elif hw == "cycleway":
                prof["cycleway_m"] += length
            else:
                has_sidewalk = any(
                    k in tags for k in ("sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both")
                )
                if has_sidewalk:
                    prof["roadway_with_sidewalk_m"] += length
                else:
                    # Самый опасный класс: идти по проезжей части,
                    # про которую в данных вообще нет тротуара.
                    prof["roadway_no_sidewalk_m"] += length

        total = sum(prof.values()) or 1.0
        result = {k: round(v, 1) for k, v in prof.items()}
        result.update({k: v for k, v in counts.items()})
        result["pedestrian_share_pct"] = round(
            100.0 * (prof["sidewalk_m"] + prof["footway_m"] + prof["crossing_m"]) / total, 1
        )
        result["roadway_no_sidewalk_share_pct"] = round(
            100.0 * prof["roadway_no_sidewalk_m"] / total, 1
        )
        return result

    # -----------------------------------------------------------------
    # Построение
    # -----------------------------------------------------------------

    def plan(self, start: tuple, goal: tuple) -> Route:
        """A* от (lat, lon) к (lat, lon)."""
        G = self.graph.G
        u, du = self.graph.nearest_node(*start)
        v, dv = self.graph.nearest_node(*goal)

        if u == v:
            raise ValueError("Старт и цель привязались к одному узлу графа")

        def h(a, b):
            na, nb = G.nodes[a], G.nodes[b]
            d = haversine_m(na["lat"], na["lon"], nb["lat"], nb["lon"])
            return d * self._min_multiplier

        try:
            path = nx.astar_path(G, u, v, heuristic=h, weight="cost")
        except nx.NetworkXNoPath:
            raise ValueError("Маршрут не найден: узлы в разных компонентах графа")

        polyline: list = []
        total_len = 0.0
        total_cost = 0.0
        for a, b in zip(path, path[1:]):
            d = G[a][b]
            total_len += d["length_m"]
            total_cost += d["cost"]
            geom = d["geometry"]
            # геометрия ребра хранится в направлении исходной линии OSM,
            # а идти мы можем в обратную сторону — разворачиваем при нужде
            if polyline and haversine_m(*polyline[-1], *geom[0]) > haversine_m(*polyline[-1], *geom[-1]):
                geom = list(reversed(geom))
            polyline.extend(geom if not polyline else geom[1:])

        maneuvers = nodes_to_maneuvers(path, G)

        return Route(
            maneuvers=maneuvers,
            total_distance_m=total_len,
            total_cost=total_cost,
            polyline=polyline,
            profile=self._profile(path),
        )

    def plan_shortest(self, start: tuple, goal: tuple) -> Route:
        """Тот же маршрут, но по чистой длине, без весов доступности.

        Нужен для сравнения в статье: «наш маршрут на X% длиннее,
        но проходит на N нерегулируемых переходов меньше».
        """
        G = self.graph.G
        u, _ = self.graph.nearest_node(*start)
        v, _ = self.graph.nearest_node(*goal)

        def h(a, b):
            na, nb = G.nodes[a], G.nodes[b]
            return haversine_m(na["lat"], na["lon"], nb["lat"], nb["lon"])

        path = nx.astar_path(G, u, v, heuristic=h, weight="length_m")
        total_len = sum(G[a][b]["length_m"] for a, b in zip(path, path[1:]))
        total_cost = sum(G[a][b]["cost"] for a, b in zip(path, path[1:]))
        polyline: list = []
        for a, b in zip(path, path[1:]):
            geom = G[a][b]["geometry"]
            if polyline and haversine_m(*polyline[-1], *geom[0]) > haversine_m(*polyline[-1], *geom[-1]):
                geom = list(reversed(geom))
            polyline.extend(geom if not polyline else geom[1:])
        return Route(
            maneuvers=nodes_to_maneuvers(path, G),
            total_distance_m=total_len,
            total_cost=total_cost,
            polyline=polyline,
            profile=self._profile(path),
        )

    # -----------------------------------------------------------------
    # Отслеживание
    # -----------------------------------------------------------------

    def _prepare(self, route: Route) -> dict:
        """Подготовить полилинию к быстрой проекции. Считается один раз."""
        key = id(route)
        if key in self._track_cache:
            return self._track_cache[key]

        pts = np.asarray(route.polyline, dtype=float)          # (N, 2) lat, lon
        scale_lon = math.cos(math.radians(float(pts[:, 0].mean())))
        xy = np.column_stack([
            pts[:, 0] * M_PER_DEG_LAT,
            pts[:, 1] * M_PER_DEG_LAT * scale_lon,
        ])
        seg_len = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        prepared = {"pts": pts, "xy": xy, "cum": cum, "scale_lon": scale_lon}
        # Положение манёвров вдоль маршрута считается один раз, а не
        # на каждом кадре: track() вызывается двадцать раз в секунду.
        prepared["maneuver_along"] = [
            self._project_along(prepared, m.lat, m.lon) for m in route.maneuvers
        ]
        self._track_cache = {key: prepared}       # держим только текущий маршрут
        return prepared

    def track(self, route: Route, pose: Pose) -> GlobalGuidance:
        """Где мы на маршруте. Вызывается каждый кадр."""
        p = self._prepare(route)
        xy, cum = p["xy"], p["cum"]

        q = np.asarray([pose.lat * M_PER_DEG_LAT,
                        pose.lon * M_PER_DEG_LAT * p["scale_lon"]])

        # векторизованное расстояние до всех сегментов полилинии
        a, b = xy[:-1], xy[1:]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom[denom == 0] = 1e-9
        t = np.clip(np.einsum("ij,ij->i", q - a, ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dists = np.linalg.norm(proj - q, axis=1)

        i = int(np.argmin(dists))
        cross_track = float(dists[i])
        along = float(cum[i] + t[i] * np.linalg.norm(ab[i]))

        # курс текущего сегмента
        lat1, lon1 = route.polyline[i]
        lat2, lon2 = route.polyline[i + 1]
        theta_route = bearing(lat1, lon1, lat2, lon2)

        # Следующий манёвр — первый СТРОГО ВПЕРЕДИ по маршруту.
        #
        # Выбирать ближайший по прямой нельзя: на старте ближайшим всегда
        # оказывается DEPART, на котором пользователь стоит, и система
        # молчит о настоящем следующем событии. Точка отсчёта — положение
        # вдоль полилинии, а не расстояние по воздуху.
        next_man = None
        dist_to_man = float("inf")
        for m, m_along in zip(route.maneuvers, p["maneuver_along"]):
            if m.type is ManeuverType.DEPART:
                continue
            if m_along <= along + 0.5:
                continue
            next_man = m
            dist_to_man = max(m_along - along, 0.0)
            break

        remaining = float(cum[-1] - along)
        arrived = remaining < 8.0

        return GlobalGuidance(
            theta_route_deg=theta_route,
            next_maneuver=next_man,
            distance_to_maneuver_m=dist_to_man if next_man else remaining,
            cross_track_error_m=cross_track,
            off_route=cross_track > self.config.off_route_threshold_m,
            arrived=arrived,
        )

    def _project_along(self, prepared: dict, lat: float, lon: float) -> float:
        """Положение точки вдоль полилинии от начала маршрута, м."""
        xy, cum = prepared["xy"], prepared["cum"]
        q = np.asarray([lat * M_PER_DEG_LAT, lon * M_PER_DEG_LAT * prepared["scale_lon"]])
        d = np.linalg.norm(xy - q, axis=1)
        return float(cum[int(np.argmin(d))])

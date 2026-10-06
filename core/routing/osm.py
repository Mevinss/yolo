# -*- coding: utf-8 -*-
"""
Разбор выгрузки OSM в узлы и рёбра пешеходного графа.

СЖАТИЕ ГЕОМЕТРИИ В ТОПОЛОГИЮ
----------------------------
В OSM линия — это последовательность точек, и большинство из них
описывают изгиб дороги, а не перекрёсток. Делать ребро графа на каждую
пару соседних точек расточительно: получаются сотни тысяч рёбер,
подавляющее большинство которых имеет ровно одного соседа.

Поэтому узлами графа становятся только:
  * точки, принадлежащие двум и более линиям (реальные перекрёстки);
  * концы линий (тупики и границы выгрузки).

Промежуточные точки не исчезают — они сохраняются в геометрии ребра
и используются для расчёта длины и для отклонения от маршрута.
Это стандартный приём, сокращающий граф примерно на порядок
без потери точности.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


#: Теги доступности, которые в OSM принято ставить НА УЗЕЛ, а не на линию.
#:
#: Это не мелочь оформления, а источник тихой ошибки. Звуковой сигнал
#: светофора размечается на узле перехода: в Астане таких узлов 144,
#: а линий с этим тегом — одна. Маршрутизатор, читающий теги только
#: с линии, не увидит ни одного озвученного перехода и будет считать
#: их все одинаково опасными — при этом никакой ошибки не возникнет,
#: система просто молча потеряет самый ценный признак доступности.
#:
#: barrier сознательно НЕ включён: болларды и калитки на тротуаре
#: сделали бы его непроходимым в is_impassable().
NODE_ACCESSIBILITY_TAGS = (
    "tactile_paving",
    "kerb",
    "traffic_signals:sound",
    "traffic_signals:vibration",
    "crossing",
    "crossing_ref",
    "crossing:markings",
    "button_operated",
    "handrail",
)


class OSMLoader:
    def __init__(self, path: str):
        self.path = path
        self._raw: dict | None = None

    # -----------------------------------------------------------------

    def load_raw(self) -> dict:
        if self._raw is None:
            opener = gzip.open if self.path.endswith(".gz") else open
            with opener(self.path, "rt", encoding="utf-8") as fh:
                self._raw = json.load(fh)
        return self._raw

    def snapshot_info(self) -> dict:
        """Дата и bbox выгрузки — идут в раздел Data статьи."""
        import os
        p = os.path.join(os.path.dirname(self.path), "snapshot.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    # -----------------------------------------------------------------

    def load_pedestrian_network(self, weights) -> tuple:
        """-> (nodes, edges)

        nodes: {node_id: {"lat": float, "lon": float, "tags": dict}}
        edges: [{"u", "v", "length_m", "tags", "way_id", "geometry"}]

        weights нужен здесь, чтобы сразу отсечь непроходимые линии
        и не тащить их в граф.
        """
        raw = self.load_raw()
        ways = raw["ways"]
        node_tags = {n["id"]: (n.get("tags") or {}) for n in raw["nodes"]}

        # --- 1. Отобрать проходимые линии ---
        usable = []
        for way in ways:
            tags = way.get("tags") or {}
            if not tags.get("highway"):
                continue
            if weights.is_impassable(tags):
                continue
            node_ids = way.get("nodes") or []
            geom = way.get("geometry") or []
            if len(node_ids) < 2 or len(geom) != len(node_ids):
                # без выровненной геометрии длину не посчитать
                continue
            usable.append((way, node_ids, geom))

        # --- 2. Найти перекрёстки ---
        occurrences: dict = defaultdict(int)
        for _, node_ids, _ in usable:
            for nid in node_ids:
                occurrences[nid] += 1

        junctions = set()
        for _, node_ids, _ in usable:
            junctions.add(node_ids[0])
            junctions.add(node_ids[-1])
            for nid in node_ids[1:-1]:
                if occurrences[nid] >= 2:
                    junctions.add(nid)

        # --- 3. Резать линии по перекрёсткам ---
        nodes: dict = {}
        edges: list = []

        for way, node_ids, geom in usable:
            tags = way.get("tags") or {}
            way_id = way["id"]

            seg_nodes = [node_ids[0]]
            seg_geom = [(geom[0]["lat"], geom[0]["lon"])]
            seg_len = 0.0

            for k in range(1, len(node_ids)):
                prev, cur = geom[k - 1], geom[k]
                seg_len += haversine_m(prev["lat"], prev["lon"], cur["lat"], cur["lon"])
                seg_geom.append((cur["lat"], cur["lon"]))
                seg_nodes.append(node_ids[k])

                if node_ids[k] in junctions:
                    u, v = seg_nodes[0], seg_nodes[-1]
                    if u != v and seg_len > 0:
                        for nid, (la, lo) in ((u, seg_geom[0]), (v, seg_geom[-1])):
                            if nid not in nodes:
                                nodes[nid] = {"lat": la, "lon": lo, "tags": node_tags.get(nid, {})}
                        edges.append({
                            "u": u, "v": v,
                            "length_m": seg_len,
                            "tags": self._merge_node_tags(tags, seg_nodes, node_tags),
                            "way_id": way_id,
                            "geometry": list(seg_geom),
                        })
                    seg_nodes = [node_ids[k]]
                    seg_geom = [(cur["lat"], cur["lon"])]
                    seg_len = 0.0

        return nodes, edges

    # -----------------------------------------------------------------

    @staticmethod
    def _merge_node_tags(way_tags: dict, seg_node_ids: list, node_tags: dict) -> dict:
        """Поднять теги доступности с узлов сегмента на само ребро.

        Тег на линии приоритетнее тега на узле: он описывает участок
        целиком, тогда как узел — одну точку. Но если на линии тега нет,
        а на узле есть — берём узловой, иначе признак теряется.

        Разрешение конфликтов между несколькими узлами сегмента:
        побеждает «худшее» значение там, где это осмысленно
        (kerb=raised важнее kerb=lowered), потому что маршрут проходит
        через ВСЕ узлы сегмента, и трудность одного из них определяет
        трудность всего участка.
        """
        merged = dict(way_tags)
        collected: dict = {}

        for nid in seg_node_ids:
            nt = node_tags.get(nid)
            if not nt:
                continue
            for key in NODE_ACCESSIBILITY_TAGS:
                if key not in nt:
                    continue
                val = nt[key]
                prev = collected.get(key)
                if prev is None:
                    collected[key] = val
                elif key == "kerb":
                    # барьерный бордюр определяет сложность участка
                    order = {"raised": 3, "rolled": 2, "lowered": 1, "flush": 0, "no": 0}
                    if order.get(val, 0) > order.get(prev, 0):
                        collected[key] = val
                elif key in ("tactile_paving", "traffic_signals:sound",
                             "traffic_signals:vibration", "handrail"):
                    # наличие полезного признака хотя бы в одном узле — уже плюс
                    if prev != "yes" and val == "yes":
                        collected[key] = val

        for key, val in collected.items():
            merged.setdefault(key, val)
        return merged

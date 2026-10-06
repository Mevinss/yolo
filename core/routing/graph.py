# -*- coding: utf-8 -*-
"""
Пешеходный граф: узлы, рёбра, стоимости, пространственный индекс.

Граф неориентированный: пешеход ходит в обе стороны. Единственное
исключение — лестницы с тегом incline, но подъём и спуск для незрячего
различаются несущественно по сравнению с самим фактом лестницы,
поэтому направление не различаем.

Стоимость рёбер считается ОДИН РАЗ при сборке и кэшируется вместе
с графом. Пересчитывать её на каждый вызов A* незачем: теги не меняются.
Меняются только веса — и тогда кэш инвалидируется по хэшу конфига.

ПРОСТРАНСТВЕННЫЙ ИНДЕКС
-----------------------
nearest_node вызывается при каждой постановке цели и при каждом
перестроении маршрута. Линейный перебор по сотням тысяч узлов
недопустим, поэтому строится KD-дерево. Координаты переводятся
в локальную метрическую проекцию: KD-дерево работает с евклидовым
расстоянием, а градусы широты и долготы в Астане имеют разный масштаб
(на широте 51 градус долготы примерно в 1.6 раза короче градуса широты).
"""

from __future__ import annotations

import math
import pickle
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

#: Опорная широта Астаны для локальной проекции
REF_LAT = 51.16
M_PER_DEG_LAT = 111_320.0


class PedestrianGraph:
    def __init__(self, nodes: dict, edges: list, weights):
        self.nodes = nodes
        self.edges = edges
        self.weights = weights

        self.G: Optional[nx.Graph] = None
        self._kdtree: Optional[cKDTree] = None
        self._node_ids: Optional[list] = None

    # -----------------------------------------------------------------

    def build(self) -> "PedestrianGraph":
        """Собрать граф и предрассчитать стоимости рёбер."""
        G = nx.Graph()

        for nid, data in self.nodes.items():
            G.add_node(nid, lat=data["lat"], lon=data["lon"], tags=data.get("tags", {}))

        for e in self.edges:
            mult = self.weights.edge_multiplier(e["tags"])
            cost = e["length_m"] * mult

            # Между теми же узлами может быть несколько линий — оставляем
            # самую дешёвую: пешеход выберет лучший из параллельных путей.
            if G.has_edge(e["u"], e["v"]) and G[e["u"]][e["v"]]["cost"] <= cost:
                continue

            G.add_edge(
                e["u"], e["v"],
                length_m=e["length_m"],
                multiplier=mult,
                cost=cost,
                tags=e["tags"],
                way_id=e["way_id"],
                geometry=e["geometry"],
            )

        self.G = G
        self._build_index()
        return self

    def _build_index(self) -> None:
        ids, coords = [], []
        scale_lon = math.cos(math.radians(REF_LAT))
        for nid in self.G.nodes:
            d = self.G.nodes[nid]
            ids.append(nid)
            coords.append((
                d["lat"] * M_PER_DEG_LAT,
                d["lon"] * M_PER_DEG_LAT * scale_lon,
            ))
        self._node_ids = ids
        self._kdtree = cKDTree(np.asarray(coords, dtype=float))

    # -----------------------------------------------------------------

    def nearest_node(self, lat: float, lon: float, k: int = 1):
        """Ближайший узел графа к точке. Возвращает (node_id, расстояние_м)."""
        if self._kdtree is None:
            raise RuntimeError("Граф не собран: вызовите build()")
        scale_lon = math.cos(math.radians(REF_LAT))
        q = np.asarray([lat * M_PER_DEG_LAT, lon * M_PER_DEG_LAT * scale_lon])
        dist, idx = self._kdtree.query(q, k=k)
        if k == 1:
            return self._node_ids[int(idx)], float(dist)
        return [(self._node_ids[int(i)], float(d)) for d, i in zip(dist, idx)]

    def largest_component(self) -> "PedestrianGraph":
        """Оставить крупнейшую связную компоненту.

        Выгрузка по bbox обрезает линии на границе, порождая висячие
        островки. Маршрут в такой островок не найдётся, и A* потратит
        время впустую, поэтому чистим сразу.
        """
        if self.G is None:
            raise RuntimeError("Граф не собран")
        comps = list(nx.connected_components(self.G))
        if not comps:
            return self
        biggest = max(comps, key=len)
        self.G = self.G.subgraph(biggest).copy()
        self._build_index()
        return self

    def stats(self) -> dict:
        if self.G is None:
            return {}
        lengths = [d["length_m"] for _, _, d in self.G.edges(data=True)]
        mults = [d["multiplier"] for _, _, d in self.G.edges(data=True)]
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "total_km": round(sum(lengths) / 1000, 1),
            "mean_edge_m": round(float(np.mean(lengths)), 1) if lengths else 0,
            "mean_multiplier": round(float(np.mean(mults)), 3) if mults else 0,
            "components": nx.number_connected_components(self.G),
        }

    # -----------------------------------------------------------------

    def save_cache(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"G": self.G, "weights_repr": repr(self.weights.config)}, fh, protocol=4)

    @classmethod
    def load_cache(cls, path: str, weights) -> "PedestrianGraph":
        """Загрузить кэш. Если веса изменились — кэш недействителен,
        потому что стоимости рёбер посчитаны по старым."""
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("weights_repr") != repr(weights.config):
            raise ValueError("Кэш собран с другими весами — пересоберите граф")
        obj = cls({}, [], weights)
        obj.G = blob["G"]
        obj._build_index()
        return obj

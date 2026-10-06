# -*- coding: utf-8 -*-
"""
Демонстрация глобального уровня: построение маршрута и сравнение с кратчайшим.

Даёт готовый результат для статьи: чем маршрут, учитывающий доступность,
отличается от кратчайшего. Формулировка вида «на 12 % длиннее, но проходит
на 3 нерегулируемых перехода и 2 лестницы без перил меньше» — это
количественный ответ на вопрос, зачем вообще нужны веса доступности.

Запуск:
    python tools/demo_route.py
    python tools/demo_route.py --from 51.1283 71.4305 --to 51.1325 71.4044
    python tools/demo_route.py --random 20        # 20 случайных пар, сводная статистика
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import RoutingConfig                       # noqa: E402
from core.routing.graph import PedestrianGraph              # noqa: E402
from core.routing.maneuvers import haversine_m              # noqa: E402
from core.routing.osm import OSMLoader                      # noqa: E402
from core.routing.planner import Router                     # noqa: E402
from core.routing.weights import AccessibilityWeights       # noqa: E402
from core.types import ManeuverType                         # noqa: E402

NETWORK = os.path.join("data", "osm", "astana_pedestrian.json.gz")
CACHE = os.path.join("data", "osm", "graph_cache.pkl")

# Ориентиры Астаны для демонстрации
LANDMARKS = {
    "bayterek": (51.1283, 71.4305),
    "khan_shatyr": (51.1325, 71.4044),
    "enu": (51.0906, 71.3986),
    "mega": (51.1408, 71.4166),
}

MANEUVER_RU = {
    ManeuverType.DEPART: "старт",
    ManeuverType.STRAIGHT: "прямо",
    ManeuverType.TURN_LEFT: "поворот налево",
    ManeuverType.TURN_RIGHT: "поворот направо",
    ManeuverType.SLIGHT_LEFT: "левее",
    ManeuverType.SLIGHT_RIGHT: "правее",
    ManeuverType.CROSSING: "ПЕРЕХОД",
    ManeuverType.STAIRS_UP: "ЛЕСТНИЦА вверх",
    ManeuverType.STAIRS_DOWN: "ЛЕСТНИЦА вниз",
    ManeuverType.ARRIVE: "прибытие",
}


def build_router(rebuild: bool = False):
    weights = AccessibilityWeights(RoutingConfig())

    if os.path.exists(CACHE) and not rebuild:
        try:
            t = time.perf_counter()
            graph = PedestrianGraph.load_cache(CACHE, weights)
            print(f"  граф из кэша за {time.perf_counter() - t:.1f} с")
            return Router(RoutingConfig(), graph, weights), graph
        except (ValueError, EOFError) as ex:
            print(f"  кэш непригоден ({ex}), пересобираю")

    t = time.perf_counter()
    loader = OSMLoader(NETWORK)
    nodes, edges = loader.load_pedestrian_network(weights)
    print(f"  разбор OSM: {len(nodes):,} узлов, {len(edges):,} рёбер за {time.perf_counter() - t:.1f} с")

    t = time.perf_counter()
    graph = PedestrianGraph(nodes, edges, weights).build().largest_component()
    print(f"  сборка графа за {time.perf_counter() - t:.1f} с")
    print("  " + json.dumps(graph.stats(), ensure_ascii=False))

    graph.save_cache(CACHE)
    return Router(RoutingConfig(), graph, weights), graph


# Строки отчёта: (ключ в Route.profile, подпись, единица)
#
# ВНИМАНИЕ К ИНТЕРПРЕТАЦИИ
# Число переходов НЕ является мерой безопасности маршрута. Путь по осевой
# линии проезжей части не содержит рёбер crossing вообще и потому выглядит
# «безопаснее» тротуарного, хотя пешеход там пересекает поток непрерывно.
# Ведущая метрика — доля длины по пешеходной инфраструктуре.
PROFILE_ROWS = [
    ("pedestrian_share_pct",         "ДОЛЯ ПО ПЕШЕХОДНОЙ ИНФРАСТР., %", "%"),
    ("roadway_no_sidewalk_share_pct", "ДОЛЯ ПО ПРОЕЗЖЕЙ БЕЗ ТРОТУАРА, %", "%"),
    ("sidewalk_m",                   "  тротуары, м",                   "m"),
    ("footway_m",                    "  пешеходные дорожки, м",         "m"),
    ("crossing_m",                   "  переходы, м",                   "m"),
    ("roadway_no_sidewalk_m",        "  проезжая часть без тротуара, м", "m"),
    ("roadway_with_sidewalk_m",      "  проезжая часть с тротуаром, м",  "m"),
    ("steps_m",                      "  лестницы, м",                   "m"),
    ("crossings",                    "переходов, шт (в пеш. сети)",     "n"),
    ("crossings_signalled",          "  из них регулируемых",           "n"),
    ("crossings_with_audio",         "  из них со звуковым сигналом",    "n"),
    ("crossings_untagged_or_unmarked", "  без тега / нерегулируемых",    "n"),
    ("stairs",                       "лестниц, шт",                     "n"),
    ("stairs_no_handrail",           "  из них без перил",              "n"),
]


def compare(router, a: tuple, b: tuple, verbose: bool = True) -> dict | None:
    try:
        t = time.perf_counter()
        acc = router.plan(a, b)
        t_acc = (time.perf_counter() - t) * 1000
        short = router.plan_shortest(a, b)
    except ValueError as ex:
        if verbose:
            print(f"  не удалось: {ex}")
        return None

    detour = 100.0 * (acc.total_distance_m - short.total_distance_m) / max(short.total_distance_m, 1)

    if verbose:
        print(f"\n  {'':<38}{'доступный':>13}{'кратчайший':>13}")
        print(f"  {'-' * 64}")
        print(f"  {'длина, м':<38}{acc.total_distance_m:>13,.0f}{short.total_distance_m:>13,.0f}")
        print(f"  {'стоимость (взвешенная)':<38}{acc.total_cost:>13,.0f}{short.total_cost:>13,.0f}")
        print(f"  {'-' * 64}")
        for key, label, unit in PROFILE_ROWS:
            va = acc.profile.get(key, 0)
            vs = short.profile.get(key, 0)
            if va == 0 and vs == 0:
                continue
            fmt = "{:>13,.1f}" if unit != "n" else "{:>13,}"
            print(f"  {label:<38}" + fmt.format(va) + fmt.format(vs))
        print(f"  {'-' * 64}")
        print(f"  крюк: {detour:+.1f} %   |   A* за {t_acc:.0f} мс")

        print("\n  --- МАНЁВРЫ ДОСТУПНОГО МАРШРУТА ---")
        for m in acc.maneuvers[:20]:
            name = MANEUVER_RU.get(m.type, m.type.value)
            extra = ""
            if m.accessibility:
                extra = "  [" + ", ".join(f"{k}={v}" for k, v in list(m.accessibility.items())[:3]) + "]"
            print(f"    {m.distance_from_prev_m:>6.0f} м  {name:<16}{extra}")
        if len(acc.maneuvers) > 20:
            print(f"    ... ещё {len(acc.maneuvers) - 20}")

    return {
        "len_acc": acc.total_distance_m, "len_short": short.total_distance_m,
        "detour_pct": detour, "ms": t_acc,
        "prof_acc": acc.profile, "prof_short": short.profile,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", nargs=2, type=float)
    ap.add_argument("--to", dest="dst", nargs=2, type=float)
    ap.add_argument("--random", type=int, default=0, help="N случайных пар для сводной статистики")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(NETWORK):
        print(f"Нет {NETWORK}. Сначала: python tools/fetch_osm.py")
        return 1

    print("Сборка маршрутизатора...")
    router, graph = build_router(args.rebuild)

    if args.random:
        node_ids = list(graph.G.nodes)
        random.seed(42)                      # воспроизводимость
        results = []
        print(f"\nСлучайные пары: {args.random}")
        for i in range(args.random):
            u, v = random.sample(node_ids, 2)
            nu, nv = graph.G.nodes[u], graph.G.nodes[v]
            if haversine_m(nu["lat"], nu["lon"], nv["lat"], nv["lon"]) < 500:
                continue
            r = compare(router, (nu["lat"], nu["lon"]), (nv["lat"], nv["lon"]), verbose=False)
            if r:
                results.append(r)

        if results:
            n = len(results)
            km_acc = sum(r["len_acc"] for r in results) / 1000
            km_short = sum(r["len_short"] for r in results) / 1000
            print(f"\n  {'=' * 66}")
            print(f"  СВОДКА ПО {n} МАРШРУТАМ  ({km_acc:.1f} км против {km_short:.1f} км)")
            print(f"  {'=' * 66}")
            print(f"  Средний крюк ради доступности: {sum(r['detour_pct'] for r in results) / n:+.1f} %")
            print(f"  Медианное время A*: {sorted(r['ms'] for r in results)[n // 2]:.0f} мс")

            print(f"\n  {'':<40}{'доступный':>12}{'кратчайший':>13}")
            print(f"  {'-' * 66}")
            # Доли считаем по суммарной длине всех маршрутов, а не как
            # среднее долей: иначе короткие маршруты весят как длинные.
            tot_a = sum(sum(v for k, v in r["prof_acc"].items() if k.endswith("_m")) for r in results)
            tot_s = sum(sum(v for k, v in r["prof_short"].items() if k.endswith("_m")) for r in results)
            for key, label, unit in PROFILE_ROWS:
                if not key.endswith("_m"):
                    continue
                a = sum(r["prof_acc"].get(key, 0) for r in results)
                s = sum(r["prof_short"].get(key, 0) for r in results)
                if a == 0 and s == 0:
                    continue
                print(f"  {label + ', доля':<40}{100 * a / tot_a:>11.1f}%{100 * s / tot_s:>12.1f}%")

            print(f"  {'-' * 66}")
            for key, label, unit in PROFILE_ROWS:
                if unit != "n":
                    continue
                a = sum(r["prof_acc"].get(key, 0) for r in results) / km_acc
                s = sum(r["prof_short"].get(key, 0) for r in results) / km_short
                print(f"  {label + ' / км':<40}{a:>12.2f}{s:>13.2f}")
            print(f"  {'=' * 66}")

            # Сохраняем как воспроизводимый артефакт: seed зафиксирован,
            # значит цифры повторяются на том же снимке OSM.
            os.makedirs(os.path.join("docs", "article"), exist_ok=True)
            out = {
                "n_routes": n,
                "seed": 42,
                "total_km_accessible": round(km_acc, 1),
                "total_km_shortest": round(km_short, 1),
                "mean_detour_pct": round(sum(r["detour_pct"] for r in results) / n, 1),
                "median_astar_ms": round(sorted(r["ms"] for r in results)[n // 2], 1),
                "share_pct": {
                    key: {
                        "accessible": round(100 * sum(r["prof_acc"].get(key, 0) for r in results) / tot_a, 1),
                        "shortest": round(100 * sum(r["prof_short"].get(key, 0) for r in results) / tot_s, 1),
                    }
                    for key, _, unit in PROFILE_ROWS if key.endswith("_m")
                },
                "per_km": {
                    key: {
                        "accessible": round(sum(r["prof_acc"].get(key, 0) for r in results) / km_acc, 3),
                        "shortest": round(sum(r["prof_short"].get(key, 0) for r in results) / km_short, 3),
                    }
                    for key, _, unit in PROFILE_ROWS if unit == "n"
                },
            }
            path = os.path.join("docs", "article", "routing_comparison_astana.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
            print(f"\n  -> {path}")
        return 0

    a = tuple(args.src) if args.src else LANDMARKS["bayterek"]
    b = tuple(args.dst) if args.dst else LANDMARKS["khan_shatyr"]

    for label, pt in (("старт", a), ("цель", b)):
        nid, dist = graph.nearest_node(*pt)
        print(f"  {label}: {pt} -> узел {nid}, привязка {dist:.0f} м")

    compare(router, a, b)
    return 0


if __name__ == "__main__":
    sys.exit(main())

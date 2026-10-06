# -*- coding: utf-8 -*-
"""
Аудит покрытия пешеходной сети тегами доступности. РЕЗУЛЬТАТ ДЛЯ СТАТЬИ.

ЗАЧЕМ ЭТО В РАБОТЕ
------------------
Accessible routing невозможна без данных о доступности. Прежде чем
предлагать алгоритм, взвешивающий переходы по наличию звукового сигнала,
надо показать, сколько таких данных вообще существует в целевом городе.

Если в Астане размечена малая доля переходов — это не недостаток метода,
а измеренный факт, из которого следуют два вывода:

  1. прямой перенос западных решений (BlindSquare, Soundscape) в Астану
     невозможен не из-за алгоритмов, а из-за отсутствия данных;
  2. раз карта молчит, информацию о доступности приходится добывать
     камерой в реальном времени — что и обосновывает двухуровневую
     архитектуру данной работы.

Таким образом аудит — не вспомогательная справка, а логическое
обоснование самой идеи слияния.

Запуск:
    python tools/audit_osm_coverage.py
    python tools/audit_osm_coverage.py --network data/osm/vienna.json.gz --label Vienna
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import RoutingConfig                      # noqa: E402
from core.routing.weights import (                          # noqa: E402
    AccessibilityWeights, PEDESTRIAN_HIGHWAYS, ROAD_HIGHWAYS,
)

DEFAULT_NETWORK = os.path.join("data", "osm", "astana_pedestrian.json.gz")
OUT_DIR = os.path.join("docs", "article")


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def way_length_m(way: dict) -> float:
    geom = way.get("geometry") or []
    if len(geom) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(geom, geom[1:]):
        total += haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
    return total


# ---------------------------------------------------------------------------
# Классификация
# ---------------------------------------------------------------------------

def classify(tags: dict) -> str:
    """Отнести объект к классу для отчёта."""
    hw = tags.get("highway", "")
    footway = tags.get("footway", "")

    if hw == "steps":
        return "steps"
    if hw == "footway" and (footway == "crossing" or "crossing" in tags):
        return "crossing"
    if hw == "footway" and footway == "sidewalk":
        return "sidewalk"
    if hw == "footway":
        return "footway"
    if hw in ("path", "corridor"):
        return "path"
    if hw == "pedestrian":
        return "pedestrian_zone"
    if hw == "living_street":
        return "living_street"
    if hw == "cycleway":
        return "cycleway"
    if hw in ROAD_HIGHWAYS:
        return "road"
    return "other"


# ---------------------------------------------------------------------------
# Аудит
# ---------------------------------------------------------------------------

def audit(network: dict, label: str) -> dict:
    weights = AccessibilityWeights(RoutingConfig())

    ways = network["ways"]
    nodes = network["nodes"]

    length_by_class: dict = defaultdict(float)
    count_by_class: dict = defaultdict(int)
    # attribute -> [сколько объектов имеет тег, сколько всего таких объектов]
    attr_stats: dict = defaultdict(lambda: [0, 0])
    # то же, но взвешенное по длине — километры важнее, чем число линий
    attr_len: dict = defaultdict(lambda: [0.0, 0.0])
    multipliers: list = []

    for way in ways:
        tags = way.get("tags") or {}
        if not tags.get("highway"):
            continue
        cls = classify(tags)
        length = way_length_m(way)

        length_by_class[cls] += length
        count_by_class[cls] += 1

        if not weights.is_impassable(tags):
            multipliers.append((weights.edge_multiplier(tags), length))

        for attr, present in weights.coverage_report(tags).items():
            key = f"{cls}.{attr}"
            attr_stats[key][1] += 1
            attr_len[key][1] += length
            if present:
                attr_stats[key][0] += 1
                attr_len[key][0] += length

    # --- распределение ЗНАЧЕНИЙ ключевых тегов ---
    # Наличие тега и его значение — разные вещи. traffic_signals:sound=no
    # означает «звука нет», и засчитывать это как покрытие было бы подлогом.
    VALUE_TAGS = (
        "tactile_paving", "kerb", "traffic_signals:sound", "handrail",
        "lit", "sidewalk", "surface", "wheelchair", "smoothness",
    )
    value_dist: dict = defaultdict(lambda: defaultdict(int))
    for way in ways:
        tags = way.get("tags") or {}
        if not tags.get("highway"):
            continue
        for key in VALUE_TAGS:
            if key in tags:
                value_dist[key][str(tags[key])] += 1

    # --- точечные объекты ---
    node_stats: dict = defaultdict(lambda: [0, 0])
    crossing_kinds: dict = defaultdict(int)
    kerb_kinds: dict = defaultdict(int)
    #: положительные значения на переходах — реально доступная инфраструктура
    crossing_positive: dict = defaultdict(int)

    for node in nodes:
        tags = node.get("tags") or {}
        if tags.get("highway") == "crossing":
            node_stats["crossing_nodes"][1] += 1
            node_stats["crossing_nodes"][0] += 1

            kind = tags.get("crossing") or tags.get("crossing_ref") or "(нет тега)"
            crossing_kinds[kind] += 1

            for attr, cond in [
                ("tactile_paving", tags.get("tactile_paving") is not None),
                ("kerb", tags.get("kerb") is not None),
                ("audio_signal", tags.get("traffic_signals:sound") is not None),
                ("markings", tags.get("crossing:markings") is not None),
            ]:
                node_stats[f"crossing_node.{attr}"][1] += 1
                if cond:
                    node_stats[f"crossing_node.{attr}"][0] += 1

            # реально доступная инфраструктура, а не просто наличие тега
            if tags.get("tactile_paving") == "yes":
                crossing_positive["tactile_paving=yes"] += 1
            if tags.get("traffic_signals:sound") in ("yes", "walk", "always"):
                crossing_positive["audio_signal=yes"] += 1
            if tags.get("kerb") in ("lowered", "flush"):
                crossing_positive["kerb lowered/flush"] += 1
            for key in VALUE_TAGS:
                if key in tags:
                    value_dist[f"node.{key}"][str(tags[key])] += 1

        if "kerb" in tags:
            kerb_kinds[tags["kerb"]] += 1

    total_mult = sum(m * ln for m, ln in multipliers)
    total_len = sum(ln for _, ln in multipliers)

    return {
        "label": label,
        "totals": {
            "ways": len(ways),
            "nodes": len(nodes),
            "network_km": round(sum(length_by_class.values()) / 1000, 1),
            "pedestrian_km": round(
                sum(v for k, v in length_by_class.items()
                    if k in ("footway", "sidewalk", "path", "pedestrian_zone", "steps", "crossing"))
                / 1000, 1
            ),
            "mean_multiplier": round(total_mult / total_len, 3) if total_len else None,
        },
        "by_class": {
            cls: {"count": count_by_class[cls], "km": round(length_by_class[cls] / 1000, 1)}
            for cls in sorted(length_by_class, key=lambda c: -length_by_class[c])
        },
        "attributes": {
            k: {
                "with_tag": v[0],
                "total": v[1],
                "pct": round(100.0 * v[0] / v[1], 1) if v[1] else 0.0,
                "km_with_tag": round(attr_len[k][0] / 1000, 1),
                "km_total": round(attr_len[k][1] / 1000, 1),
                "pct_by_km": round(100.0 * attr_len[k][0] / attr_len[k][1], 1) if attr_len[k][1] else 0.0,
            }
            for k, v in sorted(attr_stats.items())
        },
        "crossing_nodes": {
            k: {"with_tag": v[0], "total": v[1],
                "pct": round(100.0 * v[0] / v[1], 1) if v[1] else 0.0}
            for k, v in sorted(node_stats.items())
        },
        "crossing_kinds": dict(sorted(crossing_kinds.items(), key=lambda kv: -kv[1])),
        "kerb_kinds": dict(sorted(kerb_kinds.items(), key=lambda kv: -kv[1])),
        "crossing_positive": {
            k: {"count": v,
                "pct_of_crossings": round(100.0 * v / max(node_stats["crossing_nodes"][1], 1), 2)}
            for k, v in sorted(crossing_positive.items())
        },
        "value_distributions": {
            k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:8])
            for k, v in sorted(value_dist.items())
        },
        "n_crossing_nodes": node_stats["crossing_nodes"][1],
    }


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def print_report(rep: dict) -> None:
    t = rep["totals"]
    print("=" * 74)
    print(f"  ПОКРЫТИЕ ТЕГАМИ ДОСТУПНОСТИ — {rep['label']}")
    print("=" * 74)
    print(f"  Линий в сети:            {t['ways']:>10,}")
    print(f"  Точечных объектов:       {t['nodes']:>10,}")
    print(f"  Длина сети:              {t['network_km']:>10,.1f} км")
    print(f"  Из них пешеходных:       {t['pedestrian_km']:>10,.1f} км")
    print(f"  Средний множитель:       {t['mean_multiplier']:>10}   (1.0 = идеальный тротуар)")

    print("\n  --- СОСТАВ СЕТИ ---")
    print(f"  {'класс':<18}{'линий':>10}{'км':>12}")
    for cls, v in rep["by_class"].items():
        print(f"  {cls:<18}{v['count']:>10,}{v['km']:>12,.1f}")

    print("\n  --- ПОКРЫТИЕ АТРИБУТАМИ (по объектам) ---")
    print(f"  {'атрибут':<34}{'есть':>9}{'всего':>10}{'%':>8}")
    for k, v in rep["attributes"].items():
        if v["total"] < 10:
            continue
        flag = "  <-- пробел" if v["pct"] < 5 else ""
        print(f"  {k:<34}{v['with_tag']:>9,}{v['total']:>10,}{v['pct']:>7.1f}%{flag}")

    if rep["crossing_nodes"]:
        print("\n  --- ПЕРЕХОДЫ (точечные объекты) ---")
        print(f"  {'атрибут':<34}{'есть':>9}{'всего':>10}{'%':>8}")
        for k, v in rep["crossing_nodes"].items():
            print(f"  {k:<34}{v['with_tag']:>9,}{v['total']:>10,}{v['pct']:>7.1f}%")

    if rep["crossing_kinds"]:
        print("\n  --- ТИПЫ ПЕРЕХОДОВ ---")
        total = sum(rep["crossing_kinds"].values())
        for k, n in list(rep["crossing_kinds"].items())[:12]:
            print(f"  {k:<34}{n:>9,}{100.0 * n / total:>17.1f}%")

    if rep["kerb_kinds"]:
        print("\n  --- ТИПЫ БОРДЮРОВ ---")
        for k, n in list(rep["kerb_kinds"].items())[:8]:
            print(f"  {k:<34}{n:>9,}")

    if rep.get("crossing_positive"):
        print("\n  --- РЕАЛЬНО ДОСТУПНАЯ ИНФРАСТРУКТУРА НА ПЕРЕХОДАХ ---")
        print(f"  (из {rep['n_crossing_nodes']:,} переходов; наличие тега != наличие объекта)")
        for k, v in rep["crossing_positive"].items():
            print(f"  {k:<34}{v['count']:>9,}{v['pct_of_crossings']:>16.2f}%")

    if rep.get("value_distributions"):
        print("\n  --- ЗНАЧЕНИЯ КЛЮЧЕВЫХ ТЕГОВ ---")
        for tag in ("tactile_paving", "node.tactile_paving", "node.traffic_signals:sound",
                    "node.kerb", "handrail", "sidewalk", "lit"):
            dist = rep["value_distributions"].get(tag)
            if not dist:
                continue
            total = sum(dist.values())
            vals = ", ".join(f"{k}={n} ({100.0 * n / total:.0f}%)" for k, n in list(dist.items())[:5])
            print(f"  {tag:<30} всего {total:>6,}:  {vals}")
    print("=" * 74)


def save(rep: dict, label: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    slug = label.lower().replace(" ", "_")

    json_path = os.path.join(OUT_DIR, f"osm_coverage_{slug}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)

    csv_path = os.path.join(OUT_DIR, f"osm_coverage_{slug}.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["attribute", "with_tag", "total", "pct", "km_with_tag", "km_total", "pct_by_km"])
        for k, v in rep["attributes"].items():
            w.writerow([k, v["with_tag"], v["total"], v["pct"],
                        v["km_with_tag"], v["km_total"], v["pct_by_km"]])
        for k, v in rep["crossing_nodes"].items():
            w.writerow([k, v["with_tag"], v["total"], v["pct"], "", "", ""])

    print(f"\n  -> {json_path}")
    print(f"  -> {csv_path}   (готовая таблица для статьи)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Аудит покрытия OSM тегами доступности")
    ap.add_argument("--network", default=DEFAULT_NETWORK)
    ap.add_argument("--label", default="Astana")
    args = ap.parse_args()

    if not os.path.exists(args.network):
        print(f"Нет файла {args.network}. Сначала: python tools/fetch_osm.py")
        return 1

    with gzip.open(args.network, "rt", encoding="utf-8") as fh:
        network = json.load(fh)

    rep = audit(network, args.label)
    print_report(rep)
    save(rep, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())

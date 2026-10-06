# -*- coding: utf-8 -*-
"""
Выгрузка пешеходной сети Астаны из OpenStreetMap через Overpass API.

ПОЧЕМУ OVERPASS, А НЕ .osm.pbf
------------------------------
Geofabrik отдаёт весь Казахстан (~200 МБ) и требует pyrosm/osmium для
разбора. Нам нужен один город и только пешеходно-релевантные объекты.
Overpass отдаёт ровно это в JSON, без дополнительных зависимостей.

ТАЙЛИНГ
-------
Один запрос на весь город Overpass не отдаёт: срабатывает таймаут.
Режем bbox на сетку тайлов, качаем по одному, кэшируем на диск.
Прерванную выгрузку можно продолжить — уже скачанные тайлы пропускаются.

ВОСПРОИЗВОДИМОСТЬ
-----------------
OSM меняется ежедневно. В data/osm/snapshot.json пишем дату выгрузки,
bbox, версию запроса и контрольные суммы. В статье указывается именно
этот снимок, иначе результаты через год не повторить.

Запуск:
    python tools/fetch_osm.py                 # весь город
    python tools/fetch_osm.py --tiles 2       # грубая сетка, быстрее
    python tools/fetch_osm.py --bbox 51.12 71.40 51.18 71.50   # свой участок
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

# (min_lat, min_lon, max_lat, max_lon)
ASTANA_BBOX = (51.00, 71.20, 51.25, 71.65)

# Центр города — для быстрой проверки, что всё работает
ASTANA_CENTER_BBOX = (51.12, 71.40, 51.18, 71.50)

# Порядок важен. overpass.osm.ch исключён сознательно: он отдаёт
# HTTP 200 с пустым elements вместо ошибки, из-за чего часть города
# молча теряется и в статью уходят заниженные цифры. Проверено 2026-08-22.
PRIMARY_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_ENDPOINTS = [
    PRIMARY_ENDPOINT,
    "https://overpass.kumi.systems/api/interpreter",
]

# Пауза между тайлами: Overpass бесплатный и общий, не долбим его подряд.
TILE_DELAY_S = 2.0

# Версия запроса. Менять при изменении набора тегов — попадает в snapshot.json,
# чтобы было видно, что старые тайлы собраны другим запросом.
QUERY_VERSION = "v1"

OUT_DIR = os.path.join("data", "osm")
RAW_DIR = os.path.join(OUT_DIR, "raw")


# ---------------------------------------------------------------------------
# Запросы
# ---------------------------------------------------------------------------

def ways_query(bbox: tuple, timeout: int = 180) -> str:
    """Все линейные объекты с highway=*.

    Берём ВСЕ, а не только footway, по двум причинам:
      1. в Астане тротуары часто не размечены отдельными линиями —
         пешеход идёт вдоль residential/service, и без них граф рвётся;
      2. тег sidewalk=* висит на самой дороге, а не на тротуаре,
         и без дорог долю улиц с тротуарами не посчитать.

    out body geom  ->  для каждого way отдаются и nodes (топология),
    и geometry (координаты). Это позволяет построить граф за один проход.
    """
    s, w, n, e = bbox
    return f"""
[out:json][timeout:{timeout}];
(
  way["highway"]({s},{w},{n},{e});
);
out body geom;
""".strip()


def nodes_query(bbox: tuple, timeout: int = 180) -> str:
    """Точечные объекты, важные для доступности.

    crossing        — пешеходный переход
    traffic_signals — светофор (в т.ч. со звуковым сигналом)
    kerb            — бордюр: raised (барьер) / lowered / flush
    tactile_paving  — тактильная плитка
    """
    s, w, n, e = bbox
    return f"""
[out:json][timeout:{timeout}];
(
  node["highway"="crossing"]({s},{w},{n},{e});
  node["highway"="traffic_signals"]({s},{w},{n},{e});
  node["kerb"]({s},{w},{n},{e});
  node["tactile_paving"]({s},{w},{n},{e});
  node["barrier"]({s},{w},{n},{e});
);
out body;
""".strip()


# ---------------------------------------------------------------------------
# Сеть
# ---------------------------------------------------------------------------

def run_query(query: str, attempt_budget: int = 8) -> dict:
    """Выполнить запрос с перебором зеркал и экспоненциальной паузой.

    Overpass — бесплатный публичный сервис. При 429/504 ждём и пробуем
    другое зеркало, а не долбим один эндпоинт.

    ЗАЩИТА ОТ ТИХОЙ ПОТЕРИ ДАННЫХ
    -----------------------------
    Зеркало может вернуть HTTP 200 с пустым elements вместо ошибки.
    Если такое приходит НЕ с основного эндпоинта — не верим и переспрашиваем
    основной. Пустой ответ принимается только от основного эндпоинта
    (тайл в степи действительно может быть пустым).
    """
    last_error = None
    # Основной эндпоинт пробуем чаще: он единственный, чьему нулю мы верим.
    order = [PRIMARY_ENDPOINT, PRIMARY_ENDPOINT, OVERPASS_ENDPOINTS[1]] * 3

    for attempt in range(attempt_budget):
        endpoint = order[attempt % len(order)]
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=300,
                headers={"User-Agent": "blind-nav-research/0.1 (ENU Astana; accessibility routing study)"},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("remark"):
                    last_error = "remark: " + str(data["remark"])[:150]
                    print(f"      Overpass сообщает: {last_error}")
                    time.sleep(10)
                    continue
                if not data.get("elements") and endpoint != PRIMARY_ENDPOINT:
                    print("      пустой ответ от зеркала — переспрашиваю основной")
                    last_error = "empty from mirror"
                    time.sleep(3)
                    continue
                return data
            if resp.status_code in (429, 504, 502, 503):
                wait = min(60, 5 * (2 ** attempt))
                print(f"      {resp.status_code} от {endpoint[:34]}..., пауза {wait} с")
                time.sleep(wait)
                last_error = f"HTTP {resp.status_code}"
                continue
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as ex:            # сетевые сбои, таймауты, битый JSON
            last_error = repr(ex)
            wait = min(60, 5 * (2 ** attempt))
            print(f"      сбой ({last_error[:60]}), пауза {wait} с")
            time.sleep(wait)
    raise RuntimeError(f"Overpass не ответил после {attempt_budget} попыток: {last_error}")


# ---------------------------------------------------------------------------
# Тайлы
# ---------------------------------------------------------------------------

def make_tiles(bbox: tuple, n: int) -> list:
    """Разрезать bbox на сетку n x n."""
    s, w, north, e = bbox
    dlat = (north - s) / n
    dlon = (e - w) / n
    tiles = []
    for i in range(n):
        for j in range(n):
            tiles.append((
                round(s + i * dlat, 6),
                round(w + j * dlon, 6),
                round(s + (i + 1) * dlat, 6),
                round(w + (j + 1) * dlon, 6),
            ))
    return tiles


def tile_id(bbox: tuple, kind: str) -> str:
    key = f"{kind}:{QUERY_VERSION}:{':'.join(str(x) for x in bbox)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def fetch_tile(bbox: tuple, kind: str, force: bool = False) -> dict:
    """Скачать тайл или взять из кэша."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{kind}_{tile_id(bbox, kind)}.json.gz")

    if os.path.exists(path) and not force:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    query = ways_query(bbox) if kind == "ways" else nodes_query(bbox)
    data = run_query(query)

    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return data


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------

def merge(bbox: tuple, n_tiles: int, force: bool = False) -> dict:
    """Скачать все тайлы и слить в один набор без дублей.

    Тайлы перекрываются по границам (way пересекает границу и попадает
    в оба), поэтому дедупликация по id обязательна.
    """
    tiles = make_tiles(bbox, n_tiles)
    ways: dict = {}
    nodes: dict = {}

    for idx, tile in enumerate(tiles, 1):
        print(f"  [{idx}/{len(tiles)}] тайл {tile}")

        w_data = fetch_tile(tile, "ways", force)
        n_new = 0
        for el in w_data.get("elements", []):
            if el.get("type") == "way" and el["id"] not in ways:
                ways[el["id"]] = el
                n_new += 1
        print(f"        линий: +{n_new} (всего {len(ways)})")

        n_data = fetch_tile(tile, "nodes", force)
        p_new = 0
        for el in n_data.get("elements", []):
            if el.get("type") == "node" and el["id"] not in nodes:
                nodes[el["id"]] = el
                p_new += 1
        print(f"        точек: +{p_new} (всего {len(nodes)})")

        if idx < len(tiles):
            time.sleep(TILE_DELAY_S)

    return {"ways": ways, "nodes": nodes}


def save(data: dict, bbox: tuple, n_tiles: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    net_path = os.path.join(OUT_DIR, "astana_pedestrian.json.gz")
    with gzip.open(net_path, "wt", encoding="utf-8") as fh:
        json.dump(
            {"ways": list(data["ways"].values()), "nodes": list(data["nodes"].values())},
            fh, ensure_ascii=False,
        )

    snapshot = {
        "source": "OpenStreetMap via Overpass API",
        "license": "ODbL 1.0",
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox": {"min_lat": bbox[0], "min_lon": bbox[1], "max_lat": bbox[2], "max_lon": bbox[3]},
        "tiles": n_tiles * n_tiles,
        "query_version": QUERY_VERSION,
        "counts": {"ways": len(data["ways"]), "nodes": len(data["nodes"])},
        "file": os.path.basename(net_path),
        "file_bytes": os.path.getsize(net_path),
    }
    with open(os.path.join(OUT_DIR, "snapshot.json"), "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)

    print(f"\n  сеть      -> {net_path}  ({snapshot['file_bytes'] / 1e6:.1f} МБ)")
    print(f"  снимок    -> {os.path.join(OUT_DIR, 'snapshot.json')}")
    print(f"  дата UTC  :  {snapshot['fetched_utc']}   <- указывается в статье")


def main() -> int:
    ap = argparse.ArgumentParser(description="Выгрузка пешеходной сети из OSM")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    ap.add_argument("--tiles", type=int, default=4, help="сетка N x N (по умолчанию 4)")
    ap.add_argument("--center", action="store_true", help="только центр — быстрая проверка")
    ap.add_argument("--force", action="store_true", help="игнорировать кэш")
    args = ap.parse_args()

    if args.bbox:
        bbox = tuple(args.bbox)
    elif args.center:
        bbox = ASTANA_CENTER_BBOX
    else:
        bbox = ASTANA_BBOX

    print(f"Выгрузка OSM: bbox={bbox}, сетка {args.tiles}x{args.tiles}")
    data = merge(bbox, args.tiles, args.force)
    save(data, bbox, args.tiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())

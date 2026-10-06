# -*- coding: utf-8 -*-
"""
Загрузка офлайновых голосов Piper.

КАЗАХСКИЙ ГОЛОС — ОТДЕЛЬНЫЙ РЕЗУЛЬТАТ
-------------------------------------
kk_KZ-issai-high обучен в ISSAI (Институт интеллектуальных систем
и искусственного интеллекта, Назарбаев Университет). Это означает,
что казахская речь синтезируется офлайн, на устройстве, моделью
казахстанской разработки — без обращения к облачным сервисам,
у которых поддержка казахского ограничена.

В работе Ж. Каржаубай (раздел 5.4) отсутствие офлайнового казахского
синтеза названо препятствием для внедрения. Здесь оно снимается.

Три казахских голоса различаются размером и качеством:
    kk_KZ-issai-high    128 МБ   лучшее качество
    kk_KZ-iseke-x_low    28 МБ   быстрее, хуже
    kk_KZ-raya-x_low     28 МБ

Задержка синтеза входит в сквозную задержку системы, поэтому выбор
между ними — предмет замера (см. tools/bench_tts.py), а не вкуса.

Запуск:
    python tools/fetch_voices.py                 # набор по умолчанию
    python tools/fetch_voices.py --all-kazakh    # все казахские, для сравнения
    python tools/fetch_voices.py --voice ru_RU-denis-medium
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

CATALOG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
OUT_DIR = os.path.join("data", "models", "tts")

DEFAULT_SET = [
    "kk_KZ-issai-high",      # казахский, ISSAI — основной
    "ru_RU-irina-medium",    # русский
]

ALL_KAZAKH = ["kk_KZ-issai-high", "kk_KZ-iseke-x_low", "kk_KZ-raya-x_low"]


def load_catalog() -> dict:
    r = requests.get(CATALOG_URL, timeout=120)
    r.raise_for_status()
    return r.json()


def download(url: str, dest: str) -> int:
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r      {done / 1e6:6.1f} / {total / 1e6:.1f} МБ  {pct:5.1f}%",
                          end="", flush=True)
    print()
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def fetch_voice(name: str, catalog: dict) -> bool:
    entry = catalog.get(name)
    if entry is None:
        print(f"  {name}: нет в каталоге")
        return False

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  {name}  (качество {entry.get('quality')})")

    for rel_path in entry.get("files", {}):
        if not (rel_path.endswith(".onnx") or rel_path.endswith(".onnx.json")):
            continue
        dest = os.path.join(OUT_DIR, os.path.basename(rel_path))
        if os.path.exists(dest):
            print(f"      уже есть: {os.path.basename(rel_path)}")
            continue
        print(f"      {os.path.basename(rel_path)}")
        download(BASE_URL + rel_path, dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Загрузка голосов Piper")
    ap.add_argument("--voice", action="append", default=[])
    ap.add_argument("--all-kazakh", action="store_true",
                    help="все казахские голоса — для сравнения качества и задержки")
    ap.add_argument("--list", action="store_true", help="показать доступные kk и ru")
    args = ap.parse_args()

    print("Каталог голосов Piper...")
    catalog = load_catalog()

    if args.list:
        for code in ("kk", "ru", "en_US"):
            print(f"\n{code}:")
            for k, v in catalog.items():
                if k.startswith(code):
                    onnx = [f for f in v.get("files", {}) if f.endswith(".onnx")]
                    size = v["files"][onnx[0]]["size_bytes"] / 1e6 if onnx else 0
                    print(f"  {k:<32} {v.get('quality'):<8} {size:6.0f} МБ")
        return 0

    wanted = list(args.voice) or (ALL_KAZAKH + ["ru_RU-irina-medium"]
                                  if args.all_kazakh else DEFAULT_SET)
    print(f"К загрузке: {', '.join(wanted)}\n")
    for name in wanted:
        fetch_voice(name, catalog)

    print(f"\nГолоса в {OUT_DIR}:")
    if os.path.isdir(OUT_DIR):
        for f in sorted(os.listdir(OUT_DIR)):
            p = os.path.join(OUT_DIR, f)
            print(f"  {f:<40} {os.path.getsize(p) / 1e6:7.1f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())

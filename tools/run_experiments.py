# -*- coding: utf-8 -*-
"""
Запуск всех экспериментов статьи: baseline, ablation, кривая компромисса.

ТРИ ГРУППЫ ЭКСПЕРИМЕНТОВ
------------------------
1. BASELINE — те же записи через конфигурации, воспроизводящие
   парадигмы конкурентов:

     route_only   w_risk = 0    маршрутные приложения (BlindSquare,
                                Lazarillo, Google Maps): ведут к цели,
                                не видя препятствий
     vision_only  маршрут не строится   приложения распознавания
                                (Lookout, Seeing AI, проект-предшественник):
                                предупреждают, но не ведут
     full         обе части     наша система

   Мы НЕ тестируем чужие приложения: разные карты, разные условия,
   невоспроизводимо. Мы воспроизводим их принцип в своей системе
   на своих записях. Такое сравнение честно и повторяемо.

2. ABLATION — отключение по одному слагаемому (ABLATION_PROFILES).
   Показывает вклад каждой части.

3. КРИВАЯ КОМПРОМИССА — перебор отношения w_risk / w_route.
   Центральный график статьи: по одной оси доля дошедших маршрутов,
   по другой пропущенные опасности. Точка выбирается осознанно,
   а не подбирается задним числом.

СИНТЕТИЧЕСКИЕ ЗАПИСИ ИСКЛЮЧАЮТСЯ
--------------------------------
Записи с "synthetic": true в meta.json пригодны только для проверки
кода. Они не попадают в итоговые таблицы, и это проверяется явно,
а не оставляется на внимательность.

Запуск:
    python tools/run_experiments.py --walks data/walks
    python tools/run_experiments.py --walks data/walks --only baseline
    python tools/run_experiments.py --walks data/walks --allow-synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ABLATION_PROFILES, Config                  # noqa: E402
from core.types import Lang                                        # noqa: E402
from eval.metrics import evaluate                                  # noqa: E402
from eval.replay import apply_overrides, replay                    # noqa: E402

OUT_DIR = os.path.join("docs", "article")

#: Baseline-конфигурации: парадигмы конкурентов, воспроизведённые
#: в нашей системе на наших записях.
BASELINES = {
    "full": {},
    "route_only": {"fusion.w_risk": 0.0},
    "vision_only": {"fusion.w_route": 0.0},
}

#: Точки кривой компромисса: отношение веса риска к весу маршрута
TRADEOFF_RATIOS = [0.25, 0.5, 1.0, 1.67, 2.5, 5.0, 10.0]


def find_walks(root: str, allow_synthetic: bool) -> list:
    walks, skipped = [], []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if not os.path.exists(os.path.join(d, "track.jsonl")):
            continue
        meta_path = os.path.join(d, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        if meta.get("synthetic") and not allow_synthetic:
            skipped.append(name)
            continue
        walks.append(d)

    if skipped:
        print(f"  исключены синтетические записи: {', '.join(skipped)}")
        print("  (пригодны только для проверки кода; --allow-synthetic чтобы включить)")
    return walks


def base_config(args) -> Config:
    cfg = Config()
    cfg.perception.model_path = args.model
    cfg.perception.device = args.device
    cfg.guidance.lang = Lang(args.lang)
    return cfg


def run_group(walks: list, name: str, variants: dict, args) -> dict:
    """Прогнать набор конфигураций по всем записям и собрать метрики."""
    results = {}
    for variant, overrides in variants.items():
        cfg = apply_overrides(base_config(args), overrides)
        cfg.profile_name = variant
        print(f"\n  --- {variant} ---")

        per_walk = []
        for walk in walks:
            out = os.path.join(walk, f"ticks_{variant}.jsonl.gz")
            if os.path.exists(out) and not args.force:
                print(f"    {os.path.basename(walk)}: уже прогнано")
            else:
                replay(walk, cfg, out_path=out)
            # metrics читает ticks.jsonl(.gz); подставляем нужный прогон
            import shutil
            shutil.copy2(out, os.path.join(walk, "ticks.jsonl.gz"))
            per_walk.append(evaluate(walk))

        results[variant] = aggregate(per_walk)
        results[variant]["overrides"] = overrides
    return results


def aggregate(per_walk: list) -> dict:
    """Свод по записям.

    Складываются исходные счётчики, а не средние по прогулкам:
    иначе короткая прогулка весит столько же, сколько длинная.
    """
    if not per_walk:
        return {}
    hz_total = sum(w["safety"]["hazards_total"] or 0 for w in per_walk)
    hz_found = sum(round((w["safety"]["hazard_recall"] or 0)
                         * (w["safety"]["hazards_total"] or 0)) for w in per_walk)
    dur = sum(w["load"].get("duration_s", 0) for w in per_walk)
    utts = sum(w["load"].get("utterances_total", 0) for w in per_walk)
    lat = [w["performance"].get("latency_p50_ms") for w in per_walk
           if w["performance"].get("latency_p50_ms")]

    return {
        "walks": len(per_walk),
        "duration_s": round(dur, 1),
        "hazard_recall": round(hz_found / hz_total, 4) if hz_total else None,
        "missed_critical": sum(w["safety"]["missed_critical"] for w in per_walk),
        "arrived_rate": round(sum(1 for w in per_walk if w["navigation"]["arrived"])
                              / len(per_walk), 4),
        "cross_track_rmse_m": _mean([w["navigation"]["cross_track_rmse_m"] for w in per_walk]),
        "replan_total": sum(w["navigation"]["replan_count"] for w in per_walk),
        "utterances_per_min": round(utts / (dur / 60.0), 2) if dur else None,
        "contradiction_rate_per_min": _mean(
            [w["load"].get("contradiction_rate_per_min") for w in per_walk]),
        "speech_duty_cycle": _mean([w["load"].get("speech_duty_cycle") for w in per_walk]),
        "latency_p50_ms": _mean(lat),
    }


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def tradeoff(walks: list, args) -> dict:
    """Кривая компромисса: перебор отношения w_risk / w_route.

    Вес риска держится равным единице, меняется вес маршрута.
    Так меняется ровно одно отношение, а абсолютный масштаб стоимости
    остаётся сопоставимым между точками.
    """
    variants = {}
    for ratio in TRADEOFF_RATIOS:
        variants[f"ratio_{ratio:g}"] = {"fusion.w_risk": 1.0,
                                        "fusion.w_route": round(1.0 / ratio, 4)}
    return run_group(walks, "tradeoff", variants, args)


def print_table(title: str, results: dict, columns: list) -> None:
    print(f"\n{'=' * 96}")
    print(f"  {title}")
    print("=" * 96)
    head = f"  {'конфигурация':<20}"
    for c, label, _ in columns:
        head += f"{label:>13}"
    print(head)
    print("  " + "-" * 92)
    for name, m in results.items():
        row = f"  {name:<20}"
        for c, _, fmt in columns:
            v = m.get(c)
            row += f"{'—':>13}" if v is None else f"{format(v, fmt):>13}"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", default=os.path.join("data", "walks"))
    ap.add_argument("--model", default="data/models/yolov8m.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--only", choices=["baseline", "ablation", "tradeoff"])
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--force", action="store_true", help="перепрогнать, игнорируя кэш")
    args = ap.parse_args()

    print(f"Поиск записей в {args.walks}...")
    walks = find_walks(args.walks, args.allow_synthetic)
    if not walks:
        print("\nНет пригодных записей.")
        print("Как записать: docs/RECORDING_PROTOCOL.md")
        print("Проверить машинерию на синтетике:")
        print("  python tools/make_synthetic_walk.py")
        print("  python tools/run_experiments.py --allow-synthetic")
        return 1
    print(f"  записей: {len(walks)}")

    report = {"walks": [os.path.basename(w) for w in walks],
              "model": args.model, "device": args.device}
    t0 = time.perf_counter()

    if args.only in (None, "baseline"):
        report["baseline"] = run_group(walks, "baseline", BASELINES, args)
        print_table("BASELINE: парадигмы конкурентов на наших записях",
                    report["baseline"], [
                        ("hazard_recall", "полнота", ".3f"),
                        ("missed_critical", "проп. крит.", "d"),
                        ("arrived_rate", "дошли", ".2f"),
                        ("cross_track_rmse_m", "откл. RMSE", ".2f"),
                        ("utterances_per_min", "реплик/мин", ".1f"),
                    ])

    if args.only in (None, "ablation"):
        report["ablation"] = run_group(walks, "ablation", ABLATION_PROFILES, args)
        print_table("ABLATION: вклад каждого слагаемого",
                    report["ablation"], [
                        ("hazard_recall", "полнота", ".3f"),
                        ("missed_critical", "проп. крит.", "d"),
                        ("arrived_rate", "дошли", ".2f"),
                        ("contradiction_rate_per_min", "противореч.", ".2f"),
                        ("utterances_per_min", "реплик/мин", ".1f"),
                    ])

    if args.only in (None, "tradeoff"):
        report["tradeoff"] = tradeoff(walks, args)
        print_table("КОМПРОМИСС w_risk / w_route",
                    report["tradeoff"], [
                        ("hazard_recall", "полнота", ".3f"),
                        ("missed_critical", "проп. крит.", "d"),
                        ("arrived_rate", "дошли", ".2f"),
                        ("cross_track_rmse_m", "откл. RMSE", ".2f"),
                    ])

    report["elapsed_min"] = round((time.perf_counter() - t0) / 60, 1)
    report["synthetic_included"] = bool(args.allow_synthetic)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_synthetic" if args.allow_synthetic else ""
    out = os.path.join(OUT_DIR, f"experiments{suffix}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {out}   ({report['elapsed_min']} мин)")
    if args.allow_synthetic:
        print("\n  ВНИМАНИЕ: включены синтетические записи. "
              "Эти цифры непригодны для статьи.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

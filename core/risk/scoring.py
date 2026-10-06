# -*- coding: utf-8 -*-
"""
Профили объектов: приоритет опасности, названия, физические размеры.

ЧЕТЫРЕ КАТЕГОРИИ ВМЕСТО ДВУХ
----------------------------
В проекте-предшественнике объекты делились на опасные (DANGERS)
и прочие (GENERAL_LABELS). Здесь категорий четыре, и различие между
ними определяет не громкость предупреждения, а само поведение системы:

  danger    может травмировать: машина, яма, лестница, люк.
            Полный вклад в поле риска, право на экстренное сообщение.

  obstacle  можно врезаться: человек, скамейка, столб, стул.
            Вклад в поле риска, обычное предупреждение.

  landmark  полезен для ориентации, но НЕ мешает движению:
            светофор, дверь, знак, часы.
            ВКЛАД В ПОЛЕ РИСКА НУЛЕВОЙ.

  neutral   не относится к делу: еда, мелкие предметы, животные зоопарка.
            Упоминается только по запросу.

ПОЧЕМУ LANDMARK НЕ ДАЁТ РИСКА
-----------------------------
Светофор висит на высоте два с половиной метра, дверь — это проход,
а не преграда. Если считать их препятствиями, локальный планировщик
начнёт обводить пользователя вокруг всего, что видит, и на перекрёстке
со светофорами свободного коридора не останется вовсе.

При этом молчать о них нельзя: незрячий подтверждает своё положение
именно по ориентирам. «Справа дверь магазина» ценнее, чем молчание,
даже когда никакой опасности нет. Поэтому ориентиры проходят мимо
поля риска прямо в guidance.

Это разделение — то, чего не было у предшественника, где любой
распознанный объект либо предупреждал, либо игнорировался.
"""

from __future__ import annotations

import csv
import os
from typing import Optional

#: Категории
DANGER = "danger"
OBSTACLE = "obstacle"
LANDMARK = "landmark"
NEUTRAL = "neutral"

#: Только эти категории наполняют поле риска
RISK_CATEGORIES = frozenset({DANGER, OBSTACLE})

DEFAULT_PROFILES_PATH = os.path.join("data", "object_profiles.csv")


class ObjectProfiles:
    """Загруженные профили объектов.

    Отдельный класс, а не словарь модуля: профили — часть конфигурации
    прогона, и в ablation-эксперименте может понадобиться подменить их
    (например, прогнать с приоритетами предшественника, чтобы измерить
    вклад именно новой разметки категорий).
    """

    def __init__(self, entries: dict, default_priority: float = 0.3,
                 indoor: bool = False):
        self.entries = entries
        self.default_priority = default_priority
        # Один и тот же предмет ведёт себя по-разному внутри и снаружи.
        # Нож на улице не встречается, а дома это главная опасность;
        # машина в комнате — почти всегда ложное срабатывание, и держать
        # ей уличный приоритет значит заглушать настоящие домашние угрозы.
        self.indoor = indoor

    # -----------------------------------------------------------------

    @classmethod
    def load(cls, path: str = DEFAULT_PROFILES_PATH,
             default_priority: float = 0.3,
             indoor: bool = False) -> "ObjectProfiles":
        entries: dict = {}
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                label = (row.get("english_name") or "").strip()
                if not label:
                    continue
                entries[label] = {
                    "ru": (row.get("russian_name") or label).strip(),
                    "kk": (row.get("kazakh_name") or "").strip(),
                    "en": label,
                    "category": (row.get("category") or NEUTRAL).strip().lower(),
                    "priority": _as_float(row.get("priority"), 0.3),
                    "indoor_category": (row.get("indoor_category")
                                        or row.get("category") or NEUTRAL).strip().lower(),
                    "indoor_priority": _as_float(row.get("indoor_priority"),
                                                 _as_float(row.get("priority"), 0.3)),
                    "height_m": _as_float(row.get("height_m"), 0.0),
                    "ground_compatible": str(row.get("ground_compatible", "0")).strip() in ("1", "true", "yes"),
                }
        if not entries:
            raise ValueError(f"Профили объектов пусты: {path}")
        return cls(entries, default_priority, indoor=indoor)

    # -----------------------------------------------------------------

    def category(self, label: str) -> str:
        e = self.entries.get(label)
        if not e:
            return NEUTRAL
        return e["indoor_category"] if self.indoor else e["category"]

    def priority(self, label: str) -> float:
        e = self.entries.get(label)
        if not e:
            return self.default_priority
        return float(e["indoor_priority"] if self.indoor else e["priority"])

    def find_by_name(self, query: str) -> list:
        """Поиск класса по названию на любом из языков.

        Человек говорит «кружка», а не «cup». Совпадение по вхождению,
        а не по равенству: «зубная щётка» и «щётка» должны находить одно
        и то же, иначе поиск требует помнить точную формулировку.
        """
        q = (query or "").strip().lower()
        if not q:
            return []
        exact, partial = [], []
        for label, e in self.entries.items():
            names = [str(e.get(k, "")).lower() for k in ("ru", "kk", "en")]
            if q in names:
                exact.append(label)
            elif any(q in n or n in q for n in names if n):
                partial.append(label)
        return exact + partial

    def height_m(self, label: str) -> Optional[float]:
        """Физическая высота класса — для оценки расстояния по высоте bbox.

        None означает, что оценка по bbox для этого класса неприменима:
        у объектов переменного размера она даёт грубые ошибки.
        """
        e = self.entries.get(label)
        if not e or not e["height_m"]:
            return None
        return float(e["height_m"])

    def ground_compatible(self, label: str) -> bool:
        """Стоит ли объект на земле — тогда применима ground-plane оценка."""
        e = self.entries.get(label)
        return bool(e["ground_compatible"]) if e else False

    def name(self, label: str, lang: str = "ru") -> str:
        e = self.entries.get(label)
        if not e:
            return label
        return e.get(lang) or e.get("ru") or label

    # -----------------------------------------------------------------

    def contributes_to_risk(self, label: str) -> bool:
        """Наполняет ли объект поле риска.

        Ориентиры не наполняют: светофор на высоте 2.5 м не мешает идти,
        а если считать его препятствием, на перекрёстке со светофорами
        свободного коридора не останется.
        """
        return self.category(label) in RISK_CATEGORIES

    def is_landmark(self, label: str) -> bool:
        return self.category(label) == LANDMARK

    def is_danger(self, label: str) -> bool:
        return self.category(label) == DANGER

    # -----------------------------------------------------------------

    def risk_weight(self, label: str, confidence: float,
                    distance_m: Optional[float], decay_m: float) -> float:
        """Вклад одного объекта в ячейку поля риска.

            r = priority * confidence * exp(-distance / decay_m)

        Возвращает 0 для ориентиров и нейтральных объектов.

        Отсутствие оценки расстояния трактуется консервативно:
        объект считается находящимся на границе полосы реакции,
        а не бесконечно далёким. Неизвестное расстояние — повод
        насторожиться, а не повод игнорировать.
        """
        if not self.contributes_to_risk(label):
            return 0.0
        import math
        d = distance_m if distance_m is not None else decay_m
        return self.priority(label) * max(0.0, min(confidence, 1.0)) * math.exp(-d / max(decay_m, 1e-6))

    def stats(self) -> dict:
        counts: dict = {}
        for e in self.entries.values():
            counts[e["category"]] = counts.get(e["category"], 0) + 1
        return {"total": len(self.entries), "by_category": counts}


def _as_float(value, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Совместимость с прежним интерфейсом
# ---------------------------------------------------------------------------

def load_object_profiles(csv_path: str = DEFAULT_PROFILES_PATH) -> ObjectProfiles:
    return ObjectProfiles.load(csv_path)


def class_priority(label: str, profiles: ObjectProfiles, default: float = 0.3) -> float:
    return profiles.priority(label)

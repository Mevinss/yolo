# -*- coding: utf-8 -*-
"""
Поиск предмета: «найди кружку».

ЗАЧЕМ ЭТО ОТДЕЛЬНЫЙ СЛОЙ
------------------------
Поиск не влияет на то, куда безопасно идти. Он меняет только то,
что система ГОВОРИТ. Смешивать его с полем риска нельзя: тогда
искомая кружка начала бы притягивать человека к столу, о который
он ударится.

Поэтому безопасность остаётся за слиянием, а поиск добавляет
собственную реплику — и уступает ей дорогу, когда впереди опасность.

ПАМЯТЬ О ПРЕДМЕТЕ
-----------------
Кружка попадает в кадр на мгновение и пропадает, стоит повернуть
голову. Без памяти система молчала бы ровно тогда, когда человек
отвернулся, чтобы к ней подойти.

Поэтому последнее положение запоминается в МИРОВЫХ координатах:
угол складывается с курсом взгляда. Тогда, повернувшись, человек
слышит «кружка справа» — и это правда относительно его нового
положения, а не устаревшая копия прежнего кадра.

Память живёт ограниченное время: предмет могли унести, и уверенно
указывать на пустое место хуже, чем признать, что его не видно.

ПОЧЕМУ НЕ ПРОСТО «НАЙДЕНО / НЕ НАЙДЕНО»
---------------------------------------
Между этими состояниями есть третье, самое частое: предмет виден,
но человек стоит к нему боком. Для него нужна не констатация,
а указание поворота, и различать эти случаи обязан сам слой поиска —
иначе решение просочится в текст фраз и размажется по языкам.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Сколько живёт память о предмете, ушедшем из кадра, с.
#: Дольше — и система указывает на место, где предмета уже нет.
MEMORY_TTL_S = 25.0

#: Считаем, что предмет прямо перед человеком, если он в этом секторе
CENTERED_DEG = 12.0

#: Ближе этого — предмет достижим рукой, поиск закончен, м
REACHED_M = 0.7

#: Совсем рядом, но ещё не в руке: стоит сказать точнее, м
CLOSE_M = 1.5


@dataclass
class TargetState:
    """Что известно об искомом предмете прямо сейчас."""

    label: str
    query: str
    #: угол относительно ВЗГЛЯДА: минус слева, плюс справа
    bearing_deg: float
    distance_m: Optional[float]
    #: виден в текущем кадре или взят из памяти
    visible: bool
    #: сколько секунд назад видели в последний раз
    age_s: float
    #: предмет в пределах вытянутой руки
    reached: bool
    #: человек смотрит прямо на него
    centered: bool


class TargetTracker:
    def __init__(self, profiles, memory_ttl_s: float = MEMORY_TTL_S):
        self.profiles = profiles
        self.memory_ttl_s = memory_ttl_s
        self.label: Optional[str] = None
        self.query: str = ""
        #: последнее положение в МИРОВЫХ координатах
        self._world_bearing: Optional[float] = None
        self._distance_m: Optional[float] = None
        self._last_seen_ts: float = 0.0

    # -----------------------------------------------------------------

    def set_target(self, query: str) -> Optional[str]:
        """Задать предмет по названию. Возвращает найденный класс или None.

        Название приходит от человека («кружка»), а не в терминах
        модели («cup»), поэтому сопоставление идёт по всем языкам.
        """
        matches = self.profiles.find_by_name(query)
        if not matches:
            return None
        self.label = matches[0]
        self.query = query.strip()
        self._world_bearing = None
        self._distance_m = None
        self._last_seen_ts = 0.0
        return self.label

    def clear(self) -> None:
        self.label = None
        self.query = ""
        self._world_bearing = None

    @property
    def active(self) -> bool:
        return self.label is not None

    # -----------------------------------------------------------------

    def update(self, detections: list, heading_deg: float,
               ts: float) -> Optional[TargetState]:
        """Обновить состояние по кадру. None, если поиск не задан."""
        if self.label is None:
            return None

        seen = [d for d in detections if d.label == self.label]
        if seen:
            # Ближайший: если кружек несколько, вести к дальней незачем
            best = min(
                seen,
                key=lambda d: d.distance_m if d.distance_m is not None else 1e6,
            )
            self._world_bearing = (heading_deg + best.bearing_deg) % 360.0
            self._distance_m = best.distance_m
            self._last_seen_ts = ts
            return self._state(best.bearing_deg, best.distance_m,
                               visible=True, age_s=0.0)

        if self._world_bearing is None:
            return None

        age = ts - self._last_seen_ts
        if age > self.memory_ttl_s:
            # Предмет могли унести. Указывать на пустое место
            # увереннее, чем признать незнание, — вредно.
            self._world_bearing = None
            return None

        # Из мировых координат обратно в угол относительно взгляда:
        # человек повернулся, и подсказка должна повернуться вместе с ним.
        rel = ((self._world_bearing - heading_deg + 180.0) % 360.0) - 180.0
        return self._state(rel, self._distance_m, visible=False, age_s=age)

    # -----------------------------------------------------------------

    def _state(self, bearing_deg: float, distance_m: Optional[float],
               visible: bool, age_s: float) -> TargetState:
        centered = abs(bearing_deg) <= CENTERED_DEG
        reached = (visible and centered and distance_m is not None
                   and distance_m <= REACHED_M)
        return TargetState(
            label=self.label, query=self.query,
            bearing_deg=float(bearing_deg), distance_m=distance_m,
            visible=visible, age_s=float(age_s),
            reached=reached, centered=centered,
        )

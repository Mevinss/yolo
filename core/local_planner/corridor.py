# -*- coding: utf-8 -*-
"""
Локальный планировщик: поле риска -> свободный коридор.

ЗАДАЧА
------
Найти направления, в которых человек может безопасно сделать следующие
несколько шагов, и оценить, насколько каждое из них «широкое».

Ширина коридора не менее важна, чем его наличие. Проход в 15 градусов
формально свободен, но незрячий не удержит такую точность курса —
система должна в этом случае замедлить пользователя, а не вести его
в игольное ушко.

АЛГОРИТМ
--------
1. Свернуть поле риска по дальности в профиль риска по углу:

       R(i) = sum_j  w_band(j) * cells[i][j]

   Веса полос убывают со временем до достижения: препятствие,
   до которого полсекунды, важнее того, до которого семь.

2. Интерполировать профиль на мелкую угловую сетку.

3. Пометить углы с R > block_threshold как заблокированные.

4. Найти максимальные непрерывные свободные интервалы.

5. Отбросить интервалы уже min_corridor_deg.

6. Если не осталось ни одного — blocked = True (команда «стоп»).

7. Отдать наружу ВЕСЬ профиль R(theta) и список коридоров-кандидатов.

ВАЖНОЕ РАЗГРАНИЧЕНИЕ
--------------------
Этот слой отвечает на вопрос «куда МОЖНО», а не «куда НАДО».
Смешивать их здесь — распространённая ошибка, из-за которой система
начинает уводить пользователя с маршрута при каждом препятствии.

ПОЧЕМУ НАРУЖУ ИДЁТ ПРОФИЛЬ, А НЕ ОДИН УГОЛ
------------------------------------------
Слияние минимизирует cost(theta) и потому требует R(theta) для любого
theta. Передача одного рекомендованного угла заставила бы слияние
интерполировать между ним и маршрутным курсом, а промежуточное
направление может оказаться прямо в препятствии:

    маршрут прямо (0), забор занимает от -15 до +15,
    свободны коридоры при -45 (широкий) и +25 (узкий).
    Отдав только -45, мы получим компромисс около -25 — внутрь забора.

Выбор между коридорами делает слияние, потому что он зависит
от маршрута, о котором этот слой ничего не знает.

ЗАЧЕМ СЕТКА МЕЛЬЧЕ СЕКТОРОВ
---------------------------
Сектор шириной 6.7 градуса — это 35 см на трёх метрах. Округление
выбора направления до сектора теряет точность, которая у нас есть:
границы препятствий известны из углового размера боксов, а не только
из номера сектора.
"""

from __future__ import annotations

import numpy as np

from core.config import LocalPlannerConfig
from core.types import Corridor, LocalGuidance, RiskField

#: Шаг мелкой угловой сетки, градусы
PROFILE_STEP_DEG = 1.0

#: Скорость пешехода, м/с. Пересчитывает дальность во время до достижения.
WALKING_SPEED_MPS = 1.2

#: Постоянная затухания веса полос по времени, с.
#: Порядок реакции с последующим действием: то, до чего больше
#: нескольких секунд, влияет на выбор направления слабо.
BAND_TIME_CONSTANT_S = 2.5


class CorridorPlanner:
    def __init__(self, config: LocalPlannerConfig):
        self.config = config
        self._prev_theta: float | None = None

    # -----------------------------------------------------------------

    def plan(self, risk: RiskField) -> LocalGuidance:
        """Поле риска -> профиль, коридоры, рекомендация."""
        angles, profile = self._collapse_over_range(risk)
        corridors = self._find_free_intervals(profile, angles, risk)

        clearance = self._clearance_ahead(risk, angles, profile)

        if not corridors:
            # Свободного прохода нет вовсе. Сглаживание здесь НЕ применяется:
            # команда «стоп» не должна размазываться по времени.
            return LocalGuidance(
                risk_profile=profile,
                profile_angles_deg=angles,
                corridors=[],
                theta_free_deg=self._prev_theta or 0.0,
                corridor_width_deg=0.0,
                clearance_m=clearance,
                blocking=self._blocking_indices(risk),
                blocked=True,
            )

        widest = max(corridors, key=lambda c: c.width_deg)
        theta = self._smooth(widest.center_deg)

        return LocalGuidance(
            risk_profile=profile,
            profile_angles_deg=angles,
            corridors=sorted(corridors, key=lambda c: -c.width_deg),
            theta_free_deg=theta,
            corridor_width_deg=widest.width_deg,
            clearance_m=clearance,
            blocking=self._blocking_indices(risk),
            blocked=False,
        )

    # -----------------------------------------------------------------

    def _band_weights(self, band_edges: np.ndarray) -> np.ndarray:
        """Веса полос дальности: ближнее важнее дальнего.

        Вес задаётся не расстоянием, а временем до достижения полосы
        при скорости пешехода: препятствие в двух метрах на бегу
        и в двух метрах при остановке — разные ситуации, и время
        описывает их лучше, чем метры.
        """
        centers = (band_edges[:-1] + band_edges[1:]) / 2.0
        times = centers / WALKING_SPEED_MPS
        return np.exp(-times / BAND_TIME_CONSTANT_S)

    def _collapse_over_range(self, risk: RiskField) -> tuple:
        """Свёртка поля по дальности -> профиль риска на мелкой сетке."""
        weights = self._band_weights(risk.band_edges_m)
        per_sector = risk.cells @ weights                     # (n_sectors,)

        # Нормировка по МАКСИМАЛЬНОМУ весу, то есть по весу ближней полосы.
        #
        # Деление на сумму весов было бы ошибкой: яма в метре занимает
        # только ближнюю полосу и дала бы примерно 0.5 при пороге
        # блокировки 0.7, то есть не считалась бы препятствием вовсе.
        # При нормировке по максимуму занятая ближняя полоса даёт 1.0,
        # а занятая дальняя — около 0.06, что и отражает смысл величины:
        # опасность направления определяется ближайшей преградой.
        per_sector = per_sector / float(weights.max())

        edges = risk.sector_edges_deg
        centers = (edges[:-1] + edges[1:]) / 2.0

        lo, hi = float(edges[0]), float(edges[-1])
        n = max(int((hi - lo) / PROFILE_STEP_DEG) + 1, 2)
        angles = np.linspace(lo, hi, n)

        # Ступенчатую сетку секторов интерполируем линейно: границы
        # препятствий в действительности не совпадают с границами секторов.
        profile = np.interp(angles, centers, per_sector,
                            left=per_sector[0], right=per_sector[-1])
        return angles, np.clip(profile, 0.0, 1.0)

    def _find_free_intervals(self, profile: np.ndarray, angles: np.ndarray,
                             risk: RiskField) -> list:
        """Максимальные непрерывные свободные интервалы -> list[Corridor]."""
        free = profile <= self.config.block_threshold
        corridors: list = []

        i = 0
        n = len(free)
        while i < n:
            if not free[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and free[j + 1]:
                j += 1

            left, right = float(angles[i]), float(angles[j])
            width = right - left
            if width >= self.config.min_corridor_deg:
                corridors.append(Corridor(
                    center_deg=(left + right) / 2.0,
                    left_deg=left,
                    right_deg=right,
                    clearance_m=self._clearance_in(risk, left, right),
                    mean_risk=float(profile[i:j + 1].mean()),
                ))
            i = j + 1

        return corridors

    # -----------------------------------------------------------------

    def _clearance_in(self, risk: RiskField, left_deg: float, right_deg: float) -> float:
        """Расстояние до ближайшего препятствия в угловом секторе.

        Берётся ближняя полоса дальности, в которой хоть один сектор
        внутри интервала занят. Если занятых нет — коридор чист
        на всю дальность поля.
        """
        edges = risk.sector_edges_deg
        centers = (edges[:-1] + edges[1:]) / 2.0
        mask = (centers >= left_deg) & (centers <= right_deg)
        if not mask.any():
            return float("inf")

        sub = risk.cells[mask, :]
        for j in range(risk.n_bands):
            if float(sub[:, j].max()) > self.config.block_threshold:
                return float(risk.band_edges_m[j])
        return float(risk.band_edges_m[-1])

    def _clearance_ahead(self, risk: RiskField, angles: np.ndarray,
                         profile: np.ndarray) -> float:
        """Свободное расстояние прямо по курсу — вход для экстренной остановки."""
        half = self.config.min_corridor_deg / 2.0
        return self._clearance_in(risk, -half, half)

    @staticmethod
    def _blocking_indices(risk: RiskField) -> list:
        """Индексы детекций, создавших риск. Нужны для объяснимой фразы."""
        seen: list = []
        for idxs in risk.contributors.values():
            for i in idxs:
                if i not in seen:
                    seen.append(i)
        return seen

    # -----------------------------------------------------------------

    def _smooth(self, theta: float) -> float:
        """Экспоненциальное сглаживание по времени.

        Без него подсказка дребезжит: «левее — правее — левее» на соседних
        кадрах. Для незрячего это хуже, чем отсутствие подсказки: он
        не успевает отработать команду до того, как она отменяется.
        """
        a = self.config.smoothing_alpha
        if self._prev_theta is None:
            self._prev_theta = theta
        else:
            self._prev_theta = a * theta + (1 - a) * self._prev_theta
        return self._prev_theta

    def reset(self) -> None:
        self._prev_theta = None

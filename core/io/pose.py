# -*- coding: utf-8 -*-
"""
Позиция и курс пользователя.

ТРИ ИСТОЧНИКА, ТРИ ПРОБЛЕМЫ
---------------------------
  GPS      — точность 5-15 м в городе, хуже между высотными домами.
             Для маршрута хватает, для «возьмите левее» — нет.
  Компас   — даёт абсолютный курс, но телефон в руке качается
             на десятки градусов, а магнитометр врёт рядом с металлом.
  Оптический поток — даёт относительный поворот между кадрами точнее
             компаса, но накапливает дрейф и не знает, где север.

Комплементарный фильтр: компас как медленная абсолютная опора,
поток как быстрая относительная поправка.

ОТКУДА БЕРЁТСЯ heading_sigma_deg
--------------------------------
Не из паспортных характеристик датчика, а из РАСХОЖДЕНИЯ двух
независимых источников. Если компас и оптический поток дают близкий
поворот, курс достоверен; если расходятся — достоверность падает,
и неважно, кто именно ошибается.

Это честнее фиксированной константы: телефон, лежащий в кармане,
и телефон, направленный вперёд на вытянутой руке, дают разную
точность, и система должна это замечать сама.

Величина критична: слияние переводит углы камеры в азимуты через
heading_deg, и ошибка в 20 градусов означает, что человека уводят
не в тот проход. Поэтому при большой sigma слияние снижает вес
маршрутного слагаемого — см. FusionPolicy.

ИСТОЧНИК
--------
Оценка поворота по оптическому потоку перенесена из проекта
Ж. Каржаубай (blind_nav.py::update_heading_from_gray, cv2.phaseCorrelate).
Добавлены GPS, компас, их слияние и оценка достоверности.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.config import IOConfig
from core.types import Pose, PoseSource

#: Максимальный поворот между соседними кадрами, градусы.
#: Больше — это рывок или ошибка корреляции, а не поворот головы.
MAX_YAW_STEP_DEG = 8.0

#: Минимальная надёжность фазовой корреляции, ниже — поток не использовать
MIN_FLOW_RESPONSE = 0.02

#: Границы оценки достоверности курса, градусы
HEADING_SIGMA_MIN = 5.0
HEADING_SIGMA_MAX = 60.0

#: Насколько быстро sigma реагирует на расхождение источников
SIGMA_ALPHA = 0.25


def normalize_deg(a: float) -> float:
    return a % 360.0


def angle_diff(a: float, b: float) -> float:
    """Кратчайшая разность азимутов в диапазоне [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


class PoseEstimator:
    def __init__(self, config: IOConfig):
        self.config = config
        self._heading: Optional[float] = None
        self._heading_sigma: float = HEADING_SIGMA_MAX
        self._prev_gray: Optional[np.ndarray] = None
        self._last_lat: Optional[float] = None
        self._last_lon: Optional[float] = None
        self._last_ts: Optional[float] = None

    # -----------------------------------------------------------------

    def update(
        self,
        lat: float,
        lon: float,
        compass_deg: Optional[float],
        accuracy_m: float,
        ts: float,
        gray: Optional[np.ndarray] = None,
    ) -> Pose:
        """Собрать Pose из показаний телефона и текущего кадра.

        gray — кадр в градациях серого для оценки поворота по потоку.
        Может отсутствовать: тогда курс держится только на компасе,
        и sigma это отражает.
        """
        flow_delta = self._yaw_from_flow(gray) if gray is not None else None

        if self._heading is None:
            self._heading = compass_deg if compass_deg is not None else 0.0
            self._heading_sigma = HEADING_SIGMA_MAX if compass_deg is None else 20.0
        else:
            predicted = self._heading + (flow_delta or 0.0)
            if compass_deg is None:
                # Только поток: курс относительный, дрейф накапливается
                self._heading = normalize_deg(predicted)
                self._heading_sigma = min(HEADING_SIGMA_MAX, self._heading_sigma + 1.0)
            else:
                disagreement = abs(angle_diff(compass_deg, predicted))
                a = self.config.heading_smoothing_alpha
                self._heading = normalize_deg(
                    predicted + a * angle_diff(compass_deg, predicted)
                )
                # Расхождение источников И ЕСТЬ мера недостоверности
                target = min(HEADING_SIGMA_MAX, max(HEADING_SIGMA_MIN, disagreement * 1.5))
                self._heading_sigma += SIGMA_ALPHA * (target - self._heading_sigma)

        speed = self._speed(lat, lon, ts)
        self._last_lat, self._last_lon, self._last_ts = lat, lon, ts

        return Pose(
            lat=lat, lon=lon,
            heading_deg=self._heading,
            ts=ts,
            accuracy_m=accuracy_m,
            heading_sigma_deg=round(self._heading_sigma, 2),
            speed_mps=speed,
            source=PoseSource.FUSED if flow_delta is not None else PoseSource.GPS,
        )

    # -----------------------------------------------------------------

    def _yaw_from_flow(self, gray: np.ndarray) -> Optional[float]:
        """Относительный поворот между кадрами по фазовой корреляции.

        Перенос из yolo-main/blind_nav.py::update_heading_from_gray.
        Горизонтальный сдвиг кадра пересчитывается в поворот через
        поле зрения: сдвиг на всю ширину = поворот на hfov.
        """
        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape or gray.shape[1] == 0:
            return None

        try:
            import cv2
            (shift_x, _), response = cv2.phaseCorrelate(
                np.float32(prev), np.float32(gray)
            )
        except Exception:
            return None

        if response < MIN_FLOW_RESPONSE:
            return None

        yaw = -float(shift_x) / float(gray.shape[1]) * float(self.config.camera_hfov_deg)
        return max(-MAX_YAW_STEP_DEG, min(MAX_YAW_STEP_DEG, yaw))

    def _speed(self, lat: float, lon: float, ts: float) -> float:
        if self._last_ts is None or ts <= self._last_ts:
            return 0.0
        d = _haversine_m(self._last_lat, self._last_lon, lat, lon)
        v = d / (ts - self._last_ts)
        # Скачки GPS дают неправдоподобные скорости; пешеход не бегает
        # быстрее пяти метров в секунду, и такие выбросы надо гасить,
        # иначе они попадут в оценку времени до контакта.
        return v if v < 5.0 else 0.0

    @property
    def heading_sigma_deg(self) -> float:
        return self._heading_sigma

    def reset(self) -> None:
        self._heading = None
        self._heading_sigma = HEADING_SIGMA_MAX
        self._prev_gray = None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

# -*- coding: utf-8 -*-
"""
Оценка расстояния до объектов и её НЕОПРЕДЕЛЁННОСТЬ.

Три независимых метода, каждый со своей областью применимости:

  1. BBOX_HEIGHT — по известной физической высоте класса.
     Работает 0.5-3 м. Ломается на объектах переменного размера:
     автобус в 10 м и человек в 2 м дают похожий бокс.
     (Признано ограничением в работе Ж. Каржаубай, раздел 5.3.)

  2. GROUND_PLANE — по точке касания земли, pinhole-модель.
     Точнее для объектов на земле (яма, бордюр, ступень).
     Требует известных высоты камеры и наклона.

  3. MONO_DEPTH — MiDaS / Depth Anything.
     Даёт относительную глубину; привязка к метрам — в depth.py.

ЧТО ЗДЕСЬ НОВОГО ПО СРАВНЕНИЮ С ИСХОДНЫМ ПРОЕКТОМ
--------------------------------------------------
Возвращается не только оценка, но и sigma — её разброс.

Исходный проект озвучивал расстояние всегда, независимо от достоверности.
Для навигационного аида это опаснее молчания: услышав «три метра»
там, где пять, человек перестаёт доверять системе — причём именно
тогда, когда доверие нужнее всего. Зная sigma, guidance произносит
метры только там, где им можно верить, а иначе говорит «близко».

КАК СЧИТАЕТСЯ SIGMA
-------------------
Для bbox-метода — аналитически. d = H * f / h_px, поэтому

    sigma_d / d = sqrt( (sigma_H / H)^2 + (sigma_px / h_px)^2 )

где sigma_H — разброс физической высоты внутри класса (люди разного
роста, машины разных типов), а sigma_px — шум границ бокса.

Для ground-plane — численно. Формула сильно нелинейна по углу наклона,
и главный источник ошибки не арифметика, а то, что телефон в руке
качается: наклон известен с точностью около пяти градусов. Поэтому
расстояние считается при theta, theta+sigma и theta-sigma, а половина
размаха берётся за sigma. Такой способ честнее аналитической
линеаризации: у горизонта производная обращается в бесконечность,
и линейная оценка там просто неверна.

Слияние методов — взвешивание по обратной дисперсии.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

from core.types import CameraIntrinsics, DistanceSource

#: Относительный разброс физической высоты внутри класса.
#: Люди 1.5-1.9 м, машины от седана до внедорожника — около 10 процентов.
REL_HEIGHT_SIGMA = 0.10

#: Шум границ бокса, пиксели. Детектор не даёт края точнее.
BBOX_NOISE_PX = 4.0

#: Разброс наклона камеры, градусы. Телефон в руке не удерживается ровнее.
#: Главный источник ошибки ground-plane метода.
PITCH_SIGMA_DEG = 5.0

#: Разброс высоты камеры над землёй, м (рост, положение руки).
CAMERA_HEIGHT_SIGMA_M = 0.10

#: Дальше этого любая монокулярная оценка не имеет смысла
MAX_USABLE_DISTANCE_M = 25.0


# ---------------------------------------------------------------------------
# Метод 1: по высоте bounding box
# ---------------------------------------------------------------------------

def estimate_from_bbox_height(
    bbox_height_px: float,
    real_height_m: Optional[float],
    intrinsics: CameraIntrinsics,
) -> Optional[tuple]:
    """-> (distance_m, sigma_m) или None, если метод неприменим."""
    if not real_height_m or bbox_height_px <= 1.0 or intrinsics.focal_px <= 0:
        return None

    d = (real_height_m * intrinsics.focal_px) / bbox_height_px
    if d <= 0 or d > MAX_USABLE_DISTANCE_M:
        return None

    rel = math.sqrt(REL_HEIGHT_SIGMA ** 2 + (BBOX_NOISE_PX / bbox_height_px) ** 2)
    return d, d * rel


# ---------------------------------------------------------------------------
# Метод 2: по точке касания земли
# ---------------------------------------------------------------------------

def _ground_distance(
    bbox_bottom_y_px: float,
    intrinsics: CameraIntrinsics,
    camera_height_m: float,
    pitch_deg: float,
) -> Optional[float]:
    """Чистая pinhole-геометрия без оценки погрешности."""
    if intrinsics.height <= 0 or intrinsics.focal_px <= 0 or camera_height_m <= 0:
        return None

    cy = intrinsics.height / 2.0
    theta = math.radians(pitch_deg)
    denom = (float(bbox_bottom_y_px) - cy) * math.cos(theta) + intrinsics.focal_px * math.sin(theta)
    if denom <= 1e-6:
        # Точка выше линии горизонта: объект не касается земли в кадре
        return None

    d = (camera_height_m * intrinsics.focal_px) / denom
    return d if d > 0 else None


def estimate_from_ground_plane(
    bbox_bottom_y_px: float,
    intrinsics: CameraIntrinsics,
) -> Optional[tuple]:
    """-> (distance_m, sigma_m) или None.

    Погрешность считается численно: наклон и высота камеры варьируются
    в пределах своих sigma, а половина размаха результата берётся
    за неопределённость. У горизонта эта величина резко растёт — и это
    правильное поведение, потому что там метод действительно неприменим.
    """
    base = _ground_distance(
        bbox_bottom_y_px, intrinsics,
        intrinsics.height_above_ground_m, intrinsics.pitch_deg,
    )
    if base is None or base > MAX_USABLE_DISTANCE_M:
        return None

    candidates = []
    for dp in (-PITCH_SIGMA_DEG, PITCH_SIGMA_DEG):
        for dh in (-CAMERA_HEIGHT_SIGMA_M, CAMERA_HEIGHT_SIGMA_M):
            d = _ground_distance(
                bbox_bottom_y_px, intrinsics,
                intrinsics.height_above_ground_m + dh,
                intrinsics.pitch_deg + dp,
            )
            if d is not None and d <= MAX_USABLE_DISTANCE_M * 2:
                candidates.append(d)

    if not candidates:
        # Возмущение выводит точку за горизонт — метод на грани
        # применимости, честнее объявить оценку недостоверной
        return base, base

    sigma = (max(candidates) - min(candidates)) / 2.0
    return base, max(sigma, 0.05)


# ---------------------------------------------------------------------------
# Слияние
# ---------------------------------------------------------------------------

def fuse(estimates: list) -> Optional[tuple]:
    """estimates: [(d, sigma, source), ...] -> (d, sigma, DistanceSource).

    Взвешивание по обратной дисперсии:

        d = sum(d_k / s_k^2) / sum(1 / s_k^2)
        s = sqrt(1 / sum(1 / s_k^2))

    Метод с меньшей неопределённостью получает больший вес автоматически,
    без ручных коэффициентов на класс объекта — в отличие от исходного
    проекта, где веса 0.8/0.2 и 0.6/0.4 были заданы списком классов.
    """
    valid = [(d, s, src) for d, s, src in estimates
             if d is not None and s is not None and s > 0]
    if not valid:
        return None
    if len(valid) == 1:
        d, s, src = valid[0]
        return d, s, src

    inv_var_sum = sum(1.0 / (s * s) for _, s, _ in valid)
    d = sum(dd / (s * s) for dd, s, _ in valid) / inv_var_sum
    s = math.sqrt(1.0 / inv_var_sum)
    return d, s, DistanceSource.FUSED


# ---------------------------------------------------------------------------
# Сглаживание по времени
# ---------------------------------------------------------------------------

class DistanceSmoother:
    """Медианное сглаживание истории по треку.

    Ключ — track_id, а не строка из класса и координат, как было
    в исходном проекте. Разница существенна: при строковом ключе
    два человека рядом смешивали свои истории, а один человек,
    сместившийся в кадре, начинал историю заново.
    """

    def __init__(self, max_history: int = 5, ttl_s: float = 2.0):
        self.max_history = max_history
        self.ttl_s = ttl_s
        self._hist: dict = {}      # key -> [(ts, d), ...]

    def update(self, key, distance_m: Optional[float], ts: float) -> Optional[float]:
        if distance_m is None:
            return None
        hist = self._hist.setdefault(key, [])
        hist.append((ts, float(distance_m)))
        # выбросить устаревшее: объект мог уйти, а его история — остаться
        cutoff = ts - self.ttl_s
        hist[:] = [(t, d) for t, d in hist if t >= cutoff][-self.max_history:]
        return statistics.median(d for _, d in hist)

    def velocity(self, key, ts: float) -> Optional[float]:
        """Скорость сближения по истории, м/с. Положительная = приближается.

        Считается по крайним точкам окна, а не по соседним кадрам:
        разность двух последовательных измерений тонет в шуме оценки
        расстояния, который сравним с перемещением за 50 миллисекунд.
        """
        hist = self._hist.get(key)
        if not hist or len(hist) < 3:
            return None
        (t0, d0), (t1, d1) = hist[0], hist[-1]
        dt = t1 - t0
        if dt < 0.15:
            return None
        return (d0 - d1) / dt

    def forget(self, key) -> None:
        self._hist.pop(key, None)

    def prune(self, ts: float) -> None:
        cutoff = ts - self.ttl_s
        for key in [k for k, v in self._hist.items() if not v or v[-1][0] < cutoff]:
            self._hist.pop(key, None)


# ---------------------------------------------------------------------------
# Калибровка фокусного расстояния
# ---------------------------------------------------------------------------

def estimate_focal_px(bbox_height_px: float, real_height_m: float,
                      distance_m: float) -> Optional[float]:
    """Обратная задача: известны высота объекта и расстояние -> focal_px."""
    if bbox_height_px <= 0 or distance_m <= 0 or real_height_m <= 0:
        return None
    return (bbox_height_px * distance_m) / real_height_m

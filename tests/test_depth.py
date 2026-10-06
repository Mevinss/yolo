# -*- coding: utf-8 -*-
"""
Тесты модуля глубины: конвенция знака и опасности без класса.

ПОЧЕМУ ЭТО КРИТИЧНО
-------------------
Метод отклонений от плоскости земли верен, только если карта — обратная
глубина (больше значение = ближе). При обратной конвенции «провал»
и «преграда» меняются местами, и система начинает предупреждать
ровно наоборот: обходить ровный тротуар и вести в открытый люк.

При этом НИЧЕГО НЕ ПАДАЕТ. Исключений нет, числа правдоподобны,
опасности находятся — просто не те. Такую ошибку невозможно заметить
по логам, поэтому она закрывается тестом, а не наблюдательностью.

Модель здесь не используется: проверяется геометрия, а не сеть.
Синтетические карты строятся по той же формуле перспективы, которой
пользуется сам метод, но с ЗАВЕДОМО известным ответом.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PerceptionConfig                            # noqa: E402
from core.perception.depth import (                                  # noqa: E402
    RESIDUAL_SIGMA_THRESHOLD, DepthEstimator,
)
from core.types import CameraIntrinsics                              # noqa: E402

H, W = 480, 640
INTR = CameraIntrinsics(focal_px=700.0, width=W, height=H, hfov_deg=60.0,
                        height_above_ground_m=1.35, pitch_deg=10.0)


def metric_ground_plane(h: int = H, w: int = W,
                        intr: CameraIntrinsics = INTR) -> np.ndarray:
    """Метрическая глубина ровной земли: d(v) = h*f / ((v-cy)cos+f sin).

    Выше линии горизонта — большое конечное значение (небо).
    """
    cy = h / 2.0
    theta = math.radians(intr.pitch_deg)
    v = np.arange(h, dtype=float)[:, None]
    denom = (v - cy) * math.cos(theta) + intr.focal_px * math.sin(theta)
    d = np.where(denom > 1e-6, intr.height_above_ground_m * intr.focal_px / np.maximum(denom, 1e-6), 1e3)
    return np.repeat(np.clip(d, 0.1, 1e3), w, axis=1)


def inverse_ground_plane(**kw) -> np.ndarray:
    """Обратная глубина той же плоскости — то, что выдаёт MiDaS/DA-V2."""
    return 1.0 / metric_ground_plane(**kw)


def make(config_model: str = "depth_anything_v2_s") -> DepthEstimator:
    cfg = PerceptionConfig()
    cfg.depth_model = config_model
    return DepthEstimator(cfg)


# ---------------------------------------------------------------------------
# Конвенция знака
# ---------------------------------------------------------------------------

class TestConvention:
    def test_inverse_depth_recognised(self):
        """У обратной глубины нижние строки ярче: земля близко."""
        assert DepthEstimator._detect_convention(inverse_ground_plane()) is True

    def test_metric_depth_recognised(self):
        """У метрической глубины наоборот: внизу малые значения."""
        assert DepthEstimator._detect_convention(metric_ground_plane()) is False

    def test_detection_survives_noise(self):
        rng = np.random.default_rng(0)
        m = inverse_ground_plane()
        noisy = m + rng.normal(0, float(np.std(m)) * 0.15, m.shape)
        assert DepthEstimator._detect_convention(noisy) is True

    def test_detection_survives_objects_at_bottom(self):
        """Ноги и предметы внизу кадра не должны переворачивать вывод —
        поэтому берутся медианы, а не средние."""
        m = inverse_ground_plane()
        m[int(H * 0.9):, :120] = float(np.median(m[:int(H * 0.6)]))
        assert DepthEstimator._detect_convention(m) is True


# ---------------------------------------------------------------------------
# Опасности без класса
# ---------------------------------------------------------------------------

class TestGeometricHazards:
    @staticmethod
    def with_anomaly(kind: str, x0: int = 280, x1: int = 380,
                     v0: int = 380, v1: int = 450,
                     strength_frac: float = 0.20, seed: int = 0) -> np.ndarray:
        """Плоскость с локальным отклонением.

        kind='drop' — участок ДАЛЬШЕ плоскости (яма, люк, ступень вниз):
        в обратной глубине это меньшее значение.
        kind='rise' — участок БЛИЖЕ плоскости (преграда).

        Сила задаётся долей от размаха значений поверхности, а не долей
        разброса остатков: у идеальной плоскости разброс равен нулю,
        и такая мера теряет смысл. Добавляется мягкий шум — реальные
        карты глубины гладкими не бывают.
        """
        rng = np.random.default_rng(seed)
        m = inverse_ground_plane()
        band = m[int(H * (1 - 0.45)):, :]
        lo, hi = np.percentile(band, [5, 95])
        span = float(hi - lo)
        m = m + rng.normal(0.0, span * 0.004, m.shape)
        delta = strength_frac * span
        m[v0:v1, x0:x1] += (-delta if kind == "drop" else delta)
        return m

    def test_flat_ground_gives_nothing(self):
        """Ровный тротуар не должен порождать предупреждений."""
        est = make()
        assert est.geometric_hazards(inverse_ground_plane(), INTR, n_sectors=9) == []

    def test_smooth_ground_with_noise_gives_nothing(self):
        """Гладкая поверхность с шумом модели — по-прежнему ничего.

        Регрессия на найденный дефект: без пола робастного разброса
        деление на почти нулевую величину превращало шум в поток
        предупреждений на ровном месте.
        """
        rng = np.random.default_rng(3)
        m = inverse_ground_plane()
        lo, hi = np.percentile(m[int(H * 0.55):, :], [5, 95])
        m = m + rng.normal(0.0, float(hi - lo) * 0.004, m.shape)
        assert make().geometric_hazards(m, INTR, n_sectors=9) == []

    def test_drop_detected_as_drop(self):
        est = make()
        hz = est.geometric_hazards(self.with_anomaly("drop"), INTR, n_sectors=9)
        assert hz, "провал не обнаружен"
        assert any(h.label == "drop" for h in hz)
        assert not any(h.label == "rise" for h in hz), "провал принят за преграду"

    def test_rise_detected_as_rise(self):
        est = make()
        hz = est.geometric_hazards(self.with_anomaly("rise"), INTR, n_sectors=9)
        assert hz, "преграда не обнаружена"
        assert any(h.label == "rise" for h in hz)
        assert not any(h.label == "drop" for h in hz), "преграда принята за провал"

    def test_bearing_points_at_anomaly(self):
        """Угол должен указывать туда, где аномалия, а не в центр кадра."""
        est = make()
        left = est.geometric_hazards(self.with_anomaly("drop", x0=40, x1=140),
                                     INTR, n_sectors=9)
        right = est.geometric_hazards(self.with_anomaly("drop", x0=500, x1=600),
                                      INTR, n_sectors=9)
        assert left and right
        assert min(h.bearing_deg for h in left) < -10.0
        assert max(h.bearing_deg for h in right) > 10.0

    def test_hazards_carry_distance(self):
        est = make()
        hz = est.geometric_hazards(self.with_anomaly("drop"), INTR, n_sectors=9)
        assert all(h.distance_m is None or h.distance_m > 0 for h in hz)
        # Геометрия даёт положение уверенно, а дальность грубо —
        # sigma обязана это отражать, иначе guidance произнесёт
        # метры, которым нельзя верить.
        assert all(h.distance_sigma_m is None or h.distance_sigma_m > 0.2 for h in hz)

    def test_weak_anomaly_ignored(self):
        """Шум ниже порога не должен порождать предупреждений: система,
        предупреждающая обо всём, опаснее предупреждающей о главном."""
        est = make()
        weak = self.with_anomaly("drop", strength_frac=0.01)
        assert est.geometric_hazards(weak, INTR, n_sectors=9) == []

    def test_none_map_is_safe(self):
        assert make().geometric_hazards(None, INTR, n_sectors=9) == []


# ---------------------------------------------------------------------------
# Приведение к единой конвенции
# ---------------------------------------------------------------------------

class TestCanonicalisation:
    """РЕГРЕССИЯ на главный риск модуля.

    Метрическая и обратная карты одной и той же сцены обязаны давать
    ОДИНАКОВЫЕ опасности. Если приведение к единой конвенции сломается,
    метки поменяются местами — и система поведёт человека в яму,
    обходя ровное место.
    """

    @staticmethod
    def canonicalise(est: DepthEstimator, raw: np.ndarray) -> np.ndarray:
        """Повторяет то, что делает estimate() после инференса."""
        higher_is_closer = DepthEstimator._detect_convention(raw)
        if not higher_is_closer:
            return 1.0 / (np.abs(raw) + 1e-6)
        return raw

    def test_metric_and_inverse_agree(self):
        est = make()
        inverse = TestGeometricHazards.with_anomaly("drop")
        # та же сцена, но выраженная метрической глубиной
        metric = 1.0 / np.maximum(inverse, 1e-9)

        hz_inv = est.geometric_hazards(self.canonicalise(est, inverse),
                                       INTR, n_sectors=9)
        hz_met = est.geometric_hazards(self.canonicalise(est, metric),
                                       INTR, n_sectors=9)

        assert hz_inv and hz_met, "аномалия потеряна при одном из представлений"
        assert {h.label for h in hz_inv} == {h.label for h in hz_met}, (
            "метки поменялись местами: провал и преграда перепутаны"
        )

    def test_metric_input_still_finds_drop_as_drop(self):
        """Самая опасная перестановка: провал, принятый за преграду."""
        est = make()
        metric = 1.0 / np.maximum(TestGeometricHazards.with_anomaly("drop"), 1e-9)
        hz = est.geometric_hazards(self.canonicalise(est, metric), INTR, n_sectors=9)
        assert any(h.label == "drop" for h in hz)
        assert not any(h.label == "rise" for h in hz)

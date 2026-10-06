# -*- coding: utf-8 -*-
"""
Тесты восприятия: профили, расстояния, курс, самодиагностика.

Проверяются не значения формул, а СВОЙСТВА, на которых держится
безопасность: что ориентир не попадает в поле риска, что неопределённость
растёт с расстоянием, что расхождение датчиков снижает доверие к курсу.
Формулы можно менять; эти свойства — нет.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import IOConfig                                  # noqa: E402
from core.io.health import (                                       # noqa: E402
    GPS_DEGRADED_M, GPS_LOST_M, HEADING_UNRELIABLE_DEG, HealthMonitor,
)
from core.io.pose import PoseEstimator, angle_diff                 # noqa: E402
from core.perception.distance import (                             # noqa: E402
    estimate_from_bbox_height, estimate_from_ground_plane, fuse, DistanceSmoother,
)
from core.risk.scoring import ObjectProfiles                       # noqa: E402
from core.types import CameraIntrinsics, DistanceSource, HealthFlag, Pose  # noqa: E402

PROFILES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "object_profiles.csv",
)

INTR = CameraIntrinsics(focal_px=700.0, width=1280, height=720, hfov_deg=60.0,
                        height_above_ground_m=1.35, pitch_deg=10.0)


# ---------------------------------------------------------------------------
# Профили объектов
# ---------------------------------------------------------------------------

class TestProfiles:
    @pytest.fixture
    def p(self):
        return ObjectProfiles.load(PROFILES_PATH)

    def test_landmark_does_not_contribute_to_risk(self, p):
        """Светофор на высоте 2.5 м не препятствие. Если считать его
        таковым, на перекрёстке не останется свободного коридора."""
        assert p.is_landmark("traffic light")
        assert not p.contributes_to_risk("traffic light")
        assert p.risk_weight("traffic light", 0.9, 3.0, 3.0) == 0.0

    def test_danger_and_obstacle_do_contribute(self, p):
        assert p.contributes_to_risk("car")
        assert p.contributes_to_risk("person")
        assert p.risk_weight("car", 0.9, 3.0, 3.0) > 0

    def test_neutral_does_not_contribute(self, p):
        assert not p.contributes_to_risk("banana")

    def test_geometric_hazards_present(self, p):
        """drop и rise порождаются глубиной, а не детектором, но обязаны
        быть в профилях — иначе получат приоритет по умолчанию
        и не попадут в поле риска."""
        assert p.is_danger("drop")
        assert p.contributes_to_risk("drop")
        assert p.contributes_to_risk("rise")

    def test_risk_falls_with_distance(self, p):
        near = p.risk_weight("car", 0.9, 1.0, 3.0)
        far = p.risk_weight("car", 0.9, 10.0, 3.0)
        assert near > far

    def test_unknown_distance_is_not_treated_as_far(self, p):
        """Неизвестное расстояние — повод насторожиться, а не игнорировать."""
        unknown = p.risk_weight("car", 0.9, None, 3.0)
        far = p.risk_weight("car", 0.9, 15.0, 3.0)
        assert unknown > far

    def test_kazakh_names_present(self, p):
        for label in ("person", "car", "pothole", "stairs", "traffic light"):
            assert p.name(label, "kk"), f"нет казахского названия для {label}"
            assert p.name(label, "kk") != label

    def test_all_coco_classes_covered(self, p):
        """Пропущенный класс COCO получит приоритет по умолчанию молча."""
        must_have = ["person", "bicycle", "car", "motorcycle", "bus", "truck",
                     "traffic light", "stop sign", "bench", "dog", "chair",
                     "backpack", "suitcase", "potted plant"]
        missing = [c for c in must_have if c not in p.entries]
        assert not missing, f"нет профилей: {missing}"


# ---------------------------------------------------------------------------
# Оценка расстояния
# ---------------------------------------------------------------------------

class TestDistance:
    def test_bbox_matches_pinhole(self):
        """d = H * f / h_px — проверка на подставленных числах."""
        d, sigma = estimate_from_bbox_height(bbox_height_px=350.0,
                                             real_height_m=1.68, intrinsics=INTR)
        assert d == pytest.approx(1.68 * 700.0 / 350.0, rel=1e-6)
        assert sigma > 0

    def test_bbox_sigma_grows_with_distance(self):
        """Дальний объект оценивается хуже — это и должна показывать sigma."""
        near = estimate_from_bbox_height(400.0, 1.68, INTR)
        far = estimate_from_bbox_height(60.0, 1.68, INTR)
        assert far[0] > near[0]
        assert far[1] > near[1]

    def test_bbox_relative_sigma_never_below_class_spread(self):
        """Разброс роста людей никуда не девается даже при огромном боксе."""
        d, sigma = estimate_from_bbox_height(700.0, 1.68, INTR)
        assert sigma / d >= 0.09

    def test_bbox_none_without_known_height(self):
        assert estimate_from_bbox_height(300.0, None, INTR) is None

    def test_ground_plane_closer_object_is_lower_in_frame(self):
        """Чем ниже точка касания в кадре, тем ближе объект."""
        near = estimate_from_ground_plane(700.0, INTR)
        far = estimate_from_ground_plane(420.0, INTR)
        assert near[0] < far[0]

    def test_ground_plane_sigma_explodes_near_horizon(self):
        """У горизонта метод неприменим, и sigma обязана это показать,
        а не выдать правдоподобное число."""
        near = estimate_from_ground_plane(700.0, INTR)
        horizon = estimate_from_ground_plane(378.0, INTR)
        assert horizon is None or horizon[1] / horizon[0] > near[1] / near[0]

    def test_ground_plane_above_horizon_is_none(self):
        assert estimate_from_ground_plane(10.0, INTR) is None

    def test_fusion_prefers_more_certain(self):
        """Взвешивание по обратной дисперсии: точная оценка тянет сильнее."""
        d, s, src = fuse([
            (5.0, 2.0, DistanceSource.BBOX_HEIGHT),
            (3.0, 0.2, DistanceSource.GROUND_PLANE),
        ])
        assert d == pytest.approx(3.0, abs=0.1)
        assert src == DistanceSource.FUSED

    def test_fusion_sigma_below_each_input(self):
        """Два независимых измерения точнее любого одного."""
        _, s, _ = fuse([
            (3.0, 0.5, DistanceSource.BBOX_HEIGHT),
            (3.2, 0.5, DistanceSource.GROUND_PLANE),
        ])
        assert s < 0.5

    def test_fusion_single_keeps_source(self):
        d, s, src = fuse([(4.0, 0.3, DistanceSource.BBOX_HEIGHT)])
        assert src == DistanceSource.BBOX_HEIGHT

    def test_fusion_empty(self):
        assert fuse([]) is None
        assert fuse([(None, None, DistanceSource.UNKNOWN)]) is None


class TestSmoother:
    def test_median_suppresses_outlier(self):
        s = DistanceSmoother()
        for i, d in enumerate([3.0, 3.1, 9.9, 3.05, 3.0]):
            out = s.update(7, d, 100.0 + i * 0.05)
        assert out == pytest.approx(3.05, abs=0.1)

    def test_velocity_positive_when_approaching(self):
        s = DistanceSmoother()
        for i, d in enumerate([5.0, 4.5, 4.0, 3.5]):
            s.update(7, d, 100.0 + i * 0.25)
        v = s.velocity(7, 100.75)
        assert v is not None and v > 0

    def test_velocity_negative_when_receding(self):
        s = DistanceSmoother()
        for i, d in enumerate([3.0, 3.5, 4.0, 4.5]):
            s.update(7, d, 100.0 + i * 0.25)
        assert s.velocity(7, 100.75) < 0

    def test_tracks_do_not_mix(self):
        """Два человека рядом не должны смешивать истории — ради этого
        ключом служит track_id, а не позиция в кадре."""
        s = DistanceSmoother()
        for i in range(4):
            s.update(1, 2.0, 100.0 + i * 0.1)
            s.update(2, 8.0, 100.0 + i * 0.1)
        assert s.update(1, 2.0, 100.5) == pytest.approx(2.0)
        assert s.update(2, 8.0, 100.5) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Курс
# ---------------------------------------------------------------------------

class TestPose:
    @staticmethod
    def gray(shift: int = 0) -> np.ndarray:
        img = np.zeros((120, 160), dtype=np.uint8)
        img[:, 40 + shift:60 + shift] = 255
        return img

    def test_agreement_lowers_sigma(self):
        """Согласие компаса и потока = курс достоверен."""
        est = PoseEstimator(IOConfig())
        est.update(51.1, 71.4, compass_deg=90.0, accuracy_m=5.0, ts=100.0)
        for i in range(1, 15):
            est.update(51.1, 71.4, compass_deg=90.0, accuracy_m=5.0, ts=100.0 + i * 0.05)
        assert est.heading_sigma_deg < 15.0

    def test_disagreement_raises_sigma(self):
        """Компас скачет — доверие к курсу падает, и неважно, кто виноват."""
        est = PoseEstimator(IOConfig())
        est.update(51.1, 71.4, compass_deg=90.0, accuracy_m=5.0, ts=100.0)
        for i in range(1, 15):
            jump = 90.0 + (60.0 if i % 2 else -60.0)
            est.update(51.1, 71.4, compass_deg=jump, accuracy_m=5.0, ts=100.0 + i * 0.05)
        assert est.heading_sigma_deg > 30.0

    def test_no_compass_sigma_grows(self):
        """Только оптический поток — курс относительный, дрейф копится."""
        est = PoseEstimator(IOConfig())
        est.update(51.1, 71.4, compass_deg=90.0, accuracy_m=5.0, ts=100.0)
        start = est.heading_sigma_deg
        for i in range(1, 20):
            est.update(51.1, 71.4, compass_deg=None, accuracy_m=5.0, ts=100.0 + i * 0.05)
        assert est.heading_sigma_deg > start

    def test_heading_stays_in_range(self):
        est = PoseEstimator(IOConfig())
        for i in range(30):
            p = est.update(51.1, 71.4, compass_deg=(i * 37) % 360,
                           accuracy_m=5.0, ts=100.0 + i * 0.05)
            assert 0.0 <= p.heading_deg < 360.0

    def test_wrap_around_north(self):
        """Переход через север не должен давать поворот на 350 градусов."""
        assert abs(angle_diff(10.0, 350.0)) == pytest.approx(20.0)

    def test_gps_jump_does_not_become_speed(self):
        """Скачок GPS не должен превращаться в скорость пешехода:
        она уходит в оценку времени до контакта."""
        est = PoseEstimator(IOConfig())
        est.update(51.1000, 71.4000, 90.0, 5.0, 100.0)
        p = est.update(51.1050, 71.4050, 90.0, 5.0, 100.1)   # ~600 м за 0.1 с
        assert p.speed_mps == 0.0


# ---------------------------------------------------------------------------
# Самодиагностика
# ---------------------------------------------------------------------------

class TestHealth:
    @staticmethod
    def pose(accuracy_m=6.0, heading_sigma=12.0, ts=100.0) -> Pose:
        return Pose(lat=51.1, lon=71.4, heading_deg=90.0, ts=ts,
                    accuracy_m=accuracy_m, heading_sigma_deg=heading_sigma)

    def test_healthy_is_full_confidence(self):
        h = HealthMonitor().assess(self.pose(), 100.0, mean_intensity=120)
        assert not h.degraded
        assert h.confidence == pytest.approx(1.0)
        assert h.detail is None

    def test_gps_thresholds(self):
        m = HealthMonitor()
        assert HealthFlag.GPS_DEGRADED in m.assess(
            self.pose(accuracy_m=GPS_DEGRADED_M + 1), 100.0, mean_intensity=120).flags
        assert HealthFlag.GPS_LOST in m.assess(
            self.pose(accuracy_m=GPS_LOST_M + 1), 101.0, mean_intensity=120).flags

    def test_unreliable_heading_detected(self):
        h = HealthMonitor().assess(
            self.pose(heading_sigma=HEADING_UNRELIABLE_DEG + 1), 100.0, mean_intensity=120)
        assert HealthFlag.HEADING_UNRELIABLE in h.flags

    def test_low_light_from_frame_intensity(self):
        h = HealthMonitor().assess(self.pose(), 100.0, mean_intensity=20)
        assert HealthFlag.LOW_LIGHT in h.flags

    def test_lux_sensor_takes_priority_over_frame(self):
        """Датчик света точнее оценки по кадру: тёмный асфальт в яркий
        день даёт низкую среднюю яркость, но света достаточно."""
        h = HealthMonitor().assess(self.pose(), 100.0, lux=500.0, mean_intensity=20)
        assert HealthFlag.LOW_LIGHT not in h.flags

    def test_confidence_compounds(self):
        """Два отказа хуже суммы вкладов: перемножение, а не вычитание."""
        m = HealthMonitor()
        one = m.assess(self.pose(), 100.0, mean_intensity=20).confidence
        two = HealthMonitor().assess(
            self.pose(heading_sigma=50.0), 100.0, mean_intensity=20).confidence
        assert two < one

    def test_announces_once_then_reminds(self):
        m = HealthMonitor(reminder_interval_s=30.0)
        h = m.assess(self.pose(), 100.0, mean_intensity=20)
        assert m.should_announce(h, 100.0) is True
        assert m.should_announce(h, 105.0) is False
        assert m.should_announce(h, 131.0) is True

    def test_recovery_resets_announcement(self):
        m = HealthMonitor()
        bad = m.assess(self.pose(), 100.0, mean_intensity=20)
        m.should_announce(bad, 100.0)
        good = m.assess(self.pose(), 101.0, mean_intensity=120)
        assert m.should_announce(good, 101.0) is False
        assert good.degraded_since_ts is None

    def test_detail_names_worst_not_all(self):
        """Перечислять четыре неисправности бессмысленно: действие
        от этого не меняется, а речь занята."""
        h = HealthMonitor().assess(
            self.pose(accuracy_m=60.0, heading_sigma=50.0), 100.0,
            mean_intensity=20, depth_available=False)
        assert h.detail is not None
        assert "," not in h.detail

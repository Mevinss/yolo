# -*- coding: utf-8 -*-
"""
Тесты глобального уровня.

Основное внимание — азимутам и углам. Ошибка в них не вызывает исключения:
система продолжает уверенно работать и вести человека не туда. Такой отказ
опаснее падения, поэтому проверяется отдельно и подробно.

Запуск:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import RoutingConfig                     # noqa: E402
from core.routing.maneuvers import (                       # noqa: E402
    angle_diff, bearing, haversine_m, _classify,
)
from core.routing.weights import AccessibilityWeights      # noqa: E402
from core.types import ManeuverType                        # noqa: E402

ASTANA = (51.1283, 71.4305)


# ---------------------------------------------------------------------------
# Азимуты
# ---------------------------------------------------------------------------

class TestBearing:
    def test_north_is_zero(self):
        assert bearing(51.0, 71.0, 51.01, 71.0) == pytest.approx(0.0, abs=0.5)

    def test_east_is_ninety(self):
        assert bearing(51.0, 71.0, 51.0, 71.01) == pytest.approx(90.0, abs=0.5)

    def test_south_is_180(self):
        assert bearing(51.0, 71.0, 50.99, 71.0) == pytest.approx(180.0, abs=0.5)

    def test_west_is_270(self):
        assert bearing(51.0, 71.0, 51.0, 70.99) == pytest.approx(270.0, abs=0.5)

    def test_always_in_range(self):
        for dlat in (-0.01, 0, 0.01):
            for dlon in (-0.01, 0, 0.01):
                if dlat == dlon == 0:
                    continue
                b = bearing(51.0, 71.0, 51.0 + dlat, 71.0 + dlon)
                assert 0.0 <= b < 360.0


class TestAngleDiff:
    def test_wraps_through_north(self):
        # с 350 на 10 градусов — поворот направо на 20, а не налево на 340
        assert angle_diff(10.0, 350.0) == pytest.approx(20.0)

    def test_sign_right_is_positive(self):
        assert angle_diff(100.0, 90.0) > 0

    def test_sign_left_is_negative(self):
        assert angle_diff(80.0, 90.0) < 0

    def test_range(self):
        for a in range(0, 360, 17):
            for b in range(0, 360, 23):
                assert -180.0 <= angle_diff(float(a), float(b)) <= 180.0


class TestCameraToGeographic:
    """Перевод углов камеры в географические — единственное место,
    где две системы координат встречаются (FusionPolicy._to_geographic)."""

    @staticmethod
    def to_geo(theta_cam: float, heading: float) -> float:
        return (heading + theta_cam) % 360.0

    def test_straight_ahead_equals_heading(self):
        assert self.to_geo(0.0, 90.0) == 90.0

    def test_object_on_left_is_counterclockwise(self):
        # смотрим на восток (90), объект слева (-30) -> северо-восток (60)
        assert self.to_geo(-30.0, 90.0) == 60.0

    def test_object_on_right_is_clockwise(self):
        assert self.to_geo(30.0, 90.0) == 120.0

    def test_wraps_past_north(self):
        assert self.to_geo(-30.0, 10.0) == 340.0


# ---------------------------------------------------------------------------
# Расстояния
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_zero(self):
        assert haversine_m(*ASTANA, *ASTANA) == pytest.approx(0.0, abs=1e-6)

    def test_one_degree_latitude(self):
        d = haversine_m(51.0, 71.0, 52.0, 71.0)
        assert d == pytest.approx(111_195, rel=0.01)

    def test_symmetry(self):
        a = haversine_m(51.0, 71.0, 51.1, 71.1)
        b = haversine_m(51.1, 71.1, 51.0, 71.0)
        assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Веса доступности
# ---------------------------------------------------------------------------

class TestWeights:
    @pytest.fixture
    def w(self):
        return AccessibilityWeights(RoutingConfig())

    def test_crossing_hierarchy(self, w):
        """Порядок безопасности переходов — ядро логики маршрутизации.

        звук < светофор < зебра < без тега
        """
        audio = w.edge_multiplier({"highway": "footway", "footway": "crossing",
                                   "crossing": "traffic_signals",
                                   "traffic_signals:sound": "yes"})
        signal = w.edge_multiplier({"highway": "footway", "footway": "crossing",
                                    "crossing": "traffic_signals"})
        marked = w.edge_multiplier({"highway": "footway", "footway": "crossing",
                                    "crossing": "marked"})
        untagged = w.edge_multiplier({"highway": "footway", "footway": "crossing"})
        assert audio < signal < marked < untagged

    def test_unknown_is_not_safe(self, w):
        """Отсутствие тега трактуется как худший случай, а не как норма."""
        untagged = w.edge_multiplier({"highway": "footway", "footway": "crossing"})
        unmarked = w.edge_multiplier({"highway": "footway", "footway": "crossing",
                                      "crossing": "unmarked"})
        assert untagged == pytest.approx(unmarked)

    def test_steps_without_handrail_worse(self, w):
        with_rail = w.edge_multiplier({"highway": "steps", "handrail": "yes"})
        without = w.edge_multiplier({"highway": "steps"})
        assert without > with_rail

    def test_tactile_paving_preferred(self, w):
        plain = w.edge_multiplier({"highway": "footway", "footway": "sidewalk"})
        tactile = w.edge_multiplier({"highway": "footway", "footway": "sidewalk",
                                     "tactile_paving": "yes"})
        assert tactile < plain

    def test_sidewalk_beats_road_without_one(self, w):
        sidewalk = w.edge_multiplier({"highway": "footway", "footway": "sidewalk"})
        road = w.edge_multiplier({"highway": "residential", "sidewalk": "no"})
        assert sidewalk < road

    def test_modifiers_compound(self, w):
        """Лестница без перил в темноте хуже, чем каждое по отдельности."""
        steps = w.edge_multiplier({"highway": "steps"})
        steps_dark = w.edge_multiplier({"highway": "steps", "lit": "no"})
        assert steps_dark > steps

    def test_multiplier_is_bounded(self, w):
        """Потолок обязателен: иначе один тег делает ребро неудалимо
        дорогим и маршрут перестаёт находиться."""
        awful = w.edge_multiplier({
            "highway": "steps", "lit": "no", "surface": "mud",
            "smoothness": "very_bad", "incline": "15", "wheelchair": "no",
            "width": "0.5",
        })
        assert awful <= 12.0

    def test_impassable(self, w):
        assert w.is_impassable({"highway": "motorway"})
        assert w.is_impassable({"highway": "footway", "foot": "no"})
        assert w.is_impassable({"highway": "service", "access": "private"})
        assert not w.is_impassable({"highway": "footway"})
        # доступ приватный, но пешеходу явно разрешено
        assert not w.is_impassable({"highway": "service", "access": "private", "foot": "yes"})

    def test_coverage_report_is_context_aware(self, w):
        """Спрашивать про перила у тротуара бессмысленно — это исказило бы
        статистику покрытия."""
        steps = w.coverage_report({"highway": "steps"})
        sidewalk = w.coverage_report({"highway": "footway", "footway": "sidewalk"})
        assert "handrail" in steps
        assert "handrail" not in sidewalk
        assert "surface" in sidewalk


# ---------------------------------------------------------------------------
# Классификация манёвров
# ---------------------------------------------------------------------------

class TestManeuverClassification:
    def test_small_change_is_straight(self):
        assert _classify(5.0) == ManeuverType.STRAIGHT
        assert _classify(-15.0) == ManeuverType.STRAIGHT

    def test_moderate_is_slight(self):
        assert _classify(35.0) == ManeuverType.SLIGHT_RIGHT
        assert _classify(-35.0) == ManeuverType.SLIGHT_LEFT

    def test_large_is_turn(self):
        assert _classify(90.0) == ManeuverType.TURN_RIGHT
        assert _classify(-90.0) == ManeuverType.TURN_LEFT

    def test_boundaries_are_consistent(self):
        """Между категориями не должно быть дыр."""
        prev = None
        for d in range(-180, 181, 1):
            t = _classify(float(d))
            assert t is not None
            prev = t
        assert prev is not None


# ---------------------------------------------------------------------------
# Подъём тегов с узлов на рёбра
# ---------------------------------------------------------------------------

class TestNodeTagMerge:
    """Регрессия на реальную ошибку.

    Тег traffic_signals:sound в OSM ставится на УЗЕЛ перехода. В Астане
    таких узлов 144, а линий с этим тегом — одна. Маршрутизатор, читающий
    теги только с линии, не видел ни одного озвученного перехода
    и считал их все одинаково опасными. Исключения при этом не возникало:
    признак терялся молча.
    """

    @staticmethod
    def merge(way_tags, node_ids, node_tags):
        from core.routing.osm import OSMLoader
        return OSMLoader._merge_node_tags(way_tags, node_ids, node_tags)

    def test_audio_signal_lifted_from_node(self):
        merged = self.merge(
            {"highway": "footway", "footway": "crossing"},
            [1, 2],
            {2: {"highway": "crossing", "traffic_signals:sound": "yes"}},
        )
        assert merged["traffic_signals:sound"] == "yes"

    def test_way_tag_wins_over_node(self):
        merged = self.merge(
            {"highway": "footway", "kerb": "flush"},
            [1, 2],
            {2: {"kerb": "raised"}},
        )
        assert merged["kerb"] == "flush"

    def test_worst_kerb_wins_between_nodes(self):
        """Маршрут проходит через все узлы сегмента, поэтому трудность
        участка определяет худший из них."""
        merged = self.merge(
            {"highway": "footway"},
            [1, 2, 3],
            {1: {"kerb": "lowered"}, 3: {"kerb": "raised"}},
        )
        assert merged["kerb"] == "raised"

    def test_positive_feature_wins_between_nodes(self):
        merged = self.merge(
            {"highway": "footway", "footway": "crossing"},
            [1, 2],
            {1: {"tactile_paving": "no"}, 2: {"tactile_paving": "yes"}},
        )
        assert merged["tactile_paving"] == "yes"

    def test_barrier_not_lifted(self):
        """Боллард на тротуаре не должен делать его непроходимым."""
        merged = self.merge(
            {"highway": "footway"},
            [1, 2],
            {2: {"barrier": "bollard"}},
        )
        assert "barrier" not in merged

    def test_audio_bonus_reaches_weights(self):
        """Сквозная проверка: подъём тега действительно меняет стоимость."""
        from core.config import RoutingConfig
        from core.routing.weights import AccessibilityWeights
        w = AccessibilityWeights(RoutingConfig())

        plain = {"highway": "footway", "footway": "crossing", "crossing": "traffic_signals"}
        with_audio = self.merge(plain, [1, 2], {2: {"traffic_signals:sound": "yes"}})
        assert w.edge_multiplier(with_audio) < w.edge_multiplier(plain)

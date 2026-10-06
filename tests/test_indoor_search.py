# -*- coding: utf-8 -*-
"""
Тесты домашнего режима и поиска предмета.

ДВА СВОЙСТВА, КОТОРЫЕ НЕЛЬЗЯ ПОТЕРЯТЬ
-------------------------------------
1. Дома опасно ДРУГОЕ. Нож и ножницы на улице не встречаются и потому
   числятся безобидными; в комнате это главная угроза. Машина наоборот:
   в помещении её распознавание почти всегда ложное, и уличный приоритет
   заглушал бы настоящие домашние опасности.

2. Поиск не влияет на безопасность. Искомая кружка не должна притягивать
   человека к столу, о который он ударится: поиск добавляет реплику,
   но не меняет выбор направления.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config, GuidanceConfig                     # noqa: E402
from core.guidance.phrasing import Phrasebook                      # noqa: E402
from core.risk.scoring import ObjectProfiles                       # noqa: E402
from core.search.target import MEMORY_TTL_S, TargetTracker          # noqa: E402
from core.types import (                                            # noqa: E402
    Action, Detection, DistanceSource, FusionResult, Lang, SystemHealth, Urgency,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = os.path.join(ROOT, "data", "object_profiles.csv")


def det(label, bearing_deg=0.0, distance_m=1.0, conf=0.9, track_id=1):
    return Detection(label=label, confidence=conf, bbox=(0, 0, 50, 50),
                     bearing_deg=bearing_deg, distance_m=distance_m,
                     distance_source=DistanceSource.FUSED, distance_sigma_m=0.2,
                     track_id=track_id)


# ---------------------------------------------------------------------------
# Домашние категории
# ---------------------------------------------------------------------------

class TestIndoorProfiles:
    @pytest.fixture
    def outside(self):
        return ObjectProfiles.load(PROFILES, indoor=False)

    @pytest.fixture
    def inside(self):
        return ObjectProfiles.load(PROFILES, indoor=True)

    def test_knife_is_harmless_outside_dangerous_inside(self, outside, inside):
        """Главный смысл разделения."""
        assert not outside.contributes_to_risk("knife")
        assert inside.is_danger("knife")
        assert inside.priority("knife") > 0.8

    def test_scissors_likewise(self, outside, inside):
        assert not outside.contributes_to_risk("scissors")
        assert inside.is_danger("scissors")

    def test_oven_is_danger_inside(self, inside):
        """Горячее опаснее просто громоздкого."""
        assert inside.is_danger("oven")

    def test_car_is_demoted_inside(self, outside, inside):
        """Машина в комнате — почти всегда ложное срабатывание.
        Уличный приоритет заглушал бы настоящие домашние угрозы."""
        assert outside.is_danger("car")
        assert inside.priority("car") < 0.2
        assert not inside.contributes_to_risk("car")

    def test_furniture_matters_more_inside(self, outside, inside):
        """Обойти стул в комнате негде — в отличие от улицы."""
        for label in ("chair", "dining table", "couch"):
            assert inside.priority(label) > outside.priority(label)

    def test_stairs_dangerous_everywhere(self, outside, inside):
        assert outside.is_danger("stairs")
        assert inside.is_danger("stairs")

    def test_door_is_landmark_inside(self, inside):
        """Дверь — главный ориентир в помещении, но не преграда."""
        assert inside.is_landmark("door")
        assert not inside.contributes_to_risk("door")

    def test_names_resolve_across_languages(self, inside):
        assert inside.find_by_name("кружка") == ["cup"]
        assert inside.find_by_name("cup") == ["cup"]
        assert "toilet" in inside.find_by_name("унитаз")

    def test_unknown_name_gives_nothing(self, inside):
        assert inside.find_by_name("телепорт") == []


class TestIndoorConfig:
    def test_indoor_shortens_distances(self):
        """Комната — три-четыре метра. Предупреждать за пять значит
        говорить о том, что стоит в другом конце помещения."""
        out = Config()
        ind = out.as_indoor()
        assert ind.guidance.hazard_range_danger_m < out.guidance.hazard_range_danger_m
        assert ind.guidance.clear_requires_empty_m < out.guidance.clear_requires_empty_m
        assert ind.risk.max_range_m < out.risk.max_range_m

    def test_indoor_widens_cone(self):
        """Дома чаще поворачиваешься, чем идёшь прямо."""
        assert Config().as_indoor().guidance.hazard_cone_deg > Config().guidance.hazard_cone_deg

    def test_original_config_untouched(self):
        """as_indoor возвращает копию: один процесс может обслуживать
        оба режима, и утечка настроек между ними была бы незаметной."""
        out = Config()
        before = out.guidance.hazard_range_danger_m
        out.as_indoor()
        assert out.guidance.hazard_range_danger_m == before
        assert out.indoor is False


# ---------------------------------------------------------------------------
# Поиск предмета
# ---------------------------------------------------------------------------

class TestTargetTracker:
    @pytest.fixture
    def tracker(self):
        return TargetTracker(ObjectProfiles.load(PROFILES, indoor=True))

    def test_set_target_by_russian_name(self, tracker):
        assert tracker.set_target("кружка") == "cup"
        assert tracker.active

    def test_unknown_target_rejected(self, tracker):
        assert tracker.set_target("телепорт") is None
        assert not tracker.active

    def test_visible_target_reports_side(self, tracker):
        tracker.set_target("кружка")
        st = tracker.update([det("cup", bearing_deg=25.0, distance_m=2.0)], 0.0, 100.0)
        assert st.visible and st.bearing_deg == pytest.approx(25.0)
        assert not st.centered

    def test_reached_when_close_and_centered(self, tracker):
        tracker.set_target("кружка")
        st = tracker.update([det("cup", bearing_deg=3.0, distance_m=0.5)], 0.0, 100.0)
        assert st.reached

    def test_nearest_of_several_is_chosen(self, tracker):
        """Вести к дальней кружке, когда рядом стоит ближняя, незачем."""
        tracker.set_target("кружка")
        st = tracker.update([det("cup", 30.0, 4.0, track_id=1),
                             det("cup", -10.0, 1.2, track_id=2)], 0.0, 100.0)
        assert st.distance_m == pytest.approx(1.2)

    def test_memory_survives_object_leaving_frame(self, tracker):
        """Кружка пропадает, стоит повернуть голову. Без памяти система
        замолчала бы ровно тогда, когда человек к ней поворачивается."""
        tracker.set_target("кружка")
        tracker.update([det("cup", bearing_deg=40.0, distance_m=2.0)], 0.0, 100.0)
        st = tracker.update([], 0.0, 101.0)
        assert st is not None and not st.visible
        assert st.bearing_deg == pytest.approx(40.0, abs=1.0)

    def test_memory_rotates_with_the_person(self, tracker):
        """Повернувшись на 40 градусов вправо, человек должен услышать,
        что предмет теперь прямо — а не всё ещё справа."""
        tracker.set_target("кружка")
        tracker.update([det("cup", bearing_deg=40.0, distance_m=2.0)], 0.0, 100.0)
        st = tracker.update([], 40.0, 101.0)
        assert abs(st.bearing_deg) < 5.0

    def test_memory_expires(self, tracker):
        """Предмет могли унести. Указывать на пустое место увереннее,
        чем признать незнание, — вредно."""
        tracker.set_target("кружка")
        tracker.update([det("cup", 20.0, 2.0)], 0.0, 100.0)
        assert tracker.update([], 0.0, 100.0 + MEMORY_TTL_S + 1) is None

    def test_no_target_no_state(self, tracker):
        assert tracker.update([det("cup", 0.0, 1.0)], 0.0, 100.0) is None


# ---------------------------------------------------------------------------
# Фразы поиска
# ---------------------------------------------------------------------------

class TestTargetPhrases:
    @pytest.fixture
    def book(self):
        cfg = Config().as_indoor().guidance
        cfg.lang = Lang.RU
        return Phrasebook(cfg, ObjectProfiles.load(PROFILES, indoor=True))

    @staticmethod
    def fusion(action=Action.STRAIGHT):
        return FusionResult(action=action, theta_star_deg=0.0,
                            urgency=Urgency.INFO, deviation_from_route_deg=0.0)

    @staticmethod
    def local():
        import numpy as np
        from core.types import Corridor, LocalGuidance
        angles = np.arange(-30.0, 30.5, 1.0)
        return LocalGuidance(risk_profile=np.zeros_like(angles),
                             profile_angles_deg=angles,
                             corridors=[Corridor(0.0, -30.0, 30.0, 5.0, 0.0)],
                             theta_free_deg=0.0, corridor_width_deg=60.0,
                             clearance_m=5.0)

    def say(self, book, tracker, detections, heading=0.0, ts=100.0):
        target = tracker.update(detections, heading, ts)
        return book.compose(self.fusion(), None, self.local(),
                            SystemHealth(), detections, ts, target)

    @pytest.fixture
    def tracker(self):
        t = TargetTracker(ObjectProfiles.load(PROFILES, indoor=True))
        t.set_target("кружка")
        return t

    def test_says_side_and_distance(self, book, tracker):
        u = self.say(book, tracker, [det("cup", bearing_deg=30.0, distance_m=2.0)])
        assert "кружка" in u.text.lower()
        assert "справа" in u.text.lower()

    def test_says_reached_when_in_hand_range(self, book, tracker):
        u = self.say(book, tracker, [det("cup", bearing_deg=2.0, distance_m=0.4)])
        assert "перед вами" in u.text.lower()

    def test_says_turn_around_when_behind(self, book, tracker):
        tracker.update([det("cup", bearing_deg=40.0, distance_m=2.0)], 0.0, 100.0)
        u = self.say(book, tracker, [], heading=200.0, ts=101.0)
        assert "развернитесь" in u.text.lower()

    def test_memory_phrase_tells_where_to_turn(self, book, tracker):
        tracker.update([det("cup", bearing_deg=35.0, distance_m=2.0)], 0.0, 100.0)
        u = self.say(book, tracker, [], heading=0.0, ts=101.0)
        assert "направо" in u.text.lower()

    def test_hazard_wins_over_target(self, book, tracker):
        """Довести до кружки важно, но не ценой удара о стол по дороге."""
        objs = [det("cup", bearing_deg=5.0, distance_m=2.0, track_id=1),
                det("knife", bearing_deg=0.0, distance_m=0.5, track_id=2)]
        u = self.say(book, tracker, objs)
        assert "нож" in u.text.lower(), f"опасность уступила поиску: {u.text}"

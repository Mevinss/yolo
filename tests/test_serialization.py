# -*- coding: utf-8 -*-
"""
Круговая проверка сериализации Tick.

ЗАЧЕМ ЭТО ВАЖНЕЕ ОБЫЧНОГО
-------------------------
Запись прогулки создаётся один раз: ту же улицу в тех же условиях
второй раз не пройти. Если сериализация теряет поле, обнаружится это
на шаге 6 при подсчёте метрик — когда переснимать уже поздно.

Поэтому проверяется не «работает ли запись», а РАВЕНСТВО объекта
до и после круга. Любое новое поле контракта обязано ломать эти тесты,
пока не будет добавлено в преобразователи.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.serialization import (                                # noqa: E402
    FORMAT_VERSION, TickWriter, detection_from_dict, detection_to_dict,
    local_from_dict, local_to_dict, read_ticks, risk_from_dict, risk_to_dict,
    route_from_dict, route_to_dict, tick_from_dict, tick_to_dict,
)
from core.types import (                                        # noqa: E402
    Action, Corridor, CrossingPhase, Detection, DistanceSource, FusionResult,
    GlobalGuidance, HealthFlag, Lang, LocalGuidance, Maneuver, ManeuverType,
    Pose, PoseSource, RiskField, Route, SystemHealth, Tick, Urgency, Utterance,
)


# ---------------------------------------------------------------------------
# Фикстуры: заполняем ВСЕ поля, включая необязательные
# ---------------------------------------------------------------------------

def make_detection(i: int = 0) -> Detection:
    return Detection(
        label="person", confidence=0.87, bbox=(10, 20, 110, 320),
        bearing_deg=-12.5, distance_m=3.42,
        distance_source=DistanceSource.FUSED, distance_sigma_m=0.31,
        track_id=7 + i, radial_velocity_mps=1.15, ts=100.25,
    )


def make_risk() -> RiskField:
    return RiskField(
        cells=np.array([[0.1, 0.2, 0.0, 0.0], [0.9, 0.5, 0.1, 0.0]]),
        sector_edges_deg=np.array([-30.0, 0.0, 30.0]),
        band_edges_m=np.array([0.0, 1.5, 3.0, 6.0, 12.0]),
        ts=100.25,
        contributors={"1,0": [0, 1]},
    )


def make_local() -> LocalGuidance:
    return LocalGuidance(
        risk_profile=np.array([0.0, 0.4, 0.95, 0.3, 0.0]),
        profile_angles_deg=np.array([-30.0, -15.0, 0.0, 15.0, 30.0]),
        corridors=[Corridor(center_deg=22.0, left_deg=14.0, right_deg=30.0,
                            clearance_m=5.5, mean_risk=0.05)],
        theta_free_deg=22.0, corridor_width_deg=16.0, clearance_m=5.5,
        blocking=[0], blocked=False,
    )


def make_route() -> Route:
    return Route(
        maneuvers=[Maneuver(
            type=ManeuverType.CROSSING, lat=51.1283456, lon=71.4305123,
            exit_bearing_deg=92.5, distance_from_prev_m=41.0,
            landmark="Кенесары", accessibility={"crossing": "traffic_signals"},
        )],
        total_distance_m=2150.4, total_cost=2746.1,
        polyline=[(51.1283456, 71.4305123), (51.1290000, 71.4310000)],
        profile={"sidewalk_m": 900.0, "pedestrian_share_pct": 84.7},
    )


def make_tick(seq: int = 42) -> Tick:
    return Tick(
        seq=seq, ts=100.25,
        pose=Pose(lat=51.1283456, lon=71.4305123, heading_deg=87.5, ts=100.2,
                  accuracy_m=8.0, heading_sigma_deg=22.0, speed_mps=1.2,
                  source=PoseSource.FUSED),
        health=SystemHealth(
            flags=[HealthFlag.LOW_LIGHT, HealthFlag.GPS_DEGRADED],
            confidence=0.51, degraded_since_ts=95.0, detail="темно",
        ),
        detections=[make_detection(0), make_detection(1)],
        risk=make_risk(),
        global_guidance=GlobalGuidance(
            theta_route_deg=90.0,
            next_maneuver=make_route().maneuvers[0],
            distance_to_maneuver_m=41.0, cross_track_error_m=2.3,
            off_route=False, arrived=False,
        ),
        local_guidance=make_local(),
        fusion=FusionResult(
            action=Action.BEAR_RIGHT, theta_star_deg=101.0,
            urgency=Urgency.CAUTION, deviation_from_route_deg=11.0,
            reason="яма впереди", cause_indices=[0], replan_needed=False,
            crossing_phase=CrossingPhase.APPROACHING,
            cost_breakdown={"risk": 0.41, "route": 0.06, "smooth": 0.02},
        ),
        utterance=Utterance(
            text="Возьмите правее — впереди яма", lang=Lang.RU,
            urgency=Urgency.CAUTION, ts=100.25, ttl_s=3.0,
            dedup_key="bear_right:pothole:7", haptic="short-double",
        ),
        timings_ms={"perception_ms": 31.2, "total_ms": 58.9},
    )


# ---------------------------------------------------------------------------
# Круговые проверки по частям
# ---------------------------------------------------------------------------

class TestRoundTripParts:
    def test_detection(self):
        d = make_detection()
        back = detection_from_dict(detection_to_dict(d))
        assert back == d

    def test_detection_with_none_fields(self):
        d = Detection(label="pothole", confidence=0.5, bbox=(0, 0, 10, 10),
                      bearing_deg=0.0)
        assert detection_from_dict(detection_to_dict(d)) == d

    def test_risk_field(self):
        r = make_risk()
        back = risk_from_dict(risk_to_dict(r))
        assert np.allclose(back.cells, r.cells)
        assert np.allclose(back.sector_edges_deg, r.sector_edges_deg)
        assert np.allclose(back.band_edges_m, r.band_edges_m)
        assert back.contributors == r.contributors

    def test_local_guidance(self):
        l = make_local()
        back = local_from_dict(local_to_dict(l))
        assert np.allclose(back.risk_profile, l.risk_profile)
        assert np.allclose(back.profile_angles_deg, l.profile_angles_deg)
        assert back.corridors == l.corridors
        assert back.blocked == l.blocked

    def test_infinite_clearance_survives(self):
        """inf не входит в стандарт JSON и требует отдельной обработки."""
        l = make_local()
        l.clearance_m = float("inf")
        back = local_from_dict(local_to_dict(l))
        assert back.clearance_m == float("inf")

    def test_route(self):
        r = make_route()
        back = route_from_dict(route_to_dict(r))
        assert back.maneuvers == r.maneuvers
        assert back.profile == r.profile
        assert back.polyline == r.polyline

    def test_none_sections(self):
        assert risk_to_dict(None) is None
        assert local_to_dict(None) is None
        assert risk_from_dict(None) is None


# ---------------------------------------------------------------------------
# Круговая проверка Tick целиком
# ---------------------------------------------------------------------------

class TestRoundTripTick:
    def test_scalars_survive(self):
        t = make_tick()
        back = tick_from_dict(tick_to_dict(t))
        assert back.seq == t.seq
        assert back.ts == t.ts
        assert back.pose == t.pose
        assert back.detections == t.detections
        assert back.global_guidance == t.global_guidance
        assert back.fusion == t.fusion
        assert back.utterance == t.utterance
        assert back.timings_ms == t.timings_ms

    def test_enums_survive_as_enums(self):
        """После круга должны вернуться Enum, а не строки: сравнения
        вида action == Action.STOP иначе молча дают False."""
        back = tick_from_dict(tick_to_dict(make_tick()))
        assert isinstance(back.fusion.action, Action)
        assert isinstance(back.fusion.urgency, Urgency)
        assert isinstance(back.fusion.crossing_phase, CrossingPhase)
        assert isinstance(back.pose.source, PoseSource)
        assert isinstance(back.detections[0].distance_source, DistanceSource)
        assert isinstance(back.utterance.lang, Lang)
        assert all(isinstance(f, HealthFlag) for f in back.health.flags)

    def test_health_survives(self):
        t = make_tick()
        back = tick_from_dict(tick_to_dict(t))
        assert back.health == t.health
        assert back.health.degraded is True

    def test_heading_sigma_survives(self):
        """Неопределённость курса — то, что ломает слияние; потерять её
        в логе значит не суметь объяснить неудачные прогоны."""
        back = tick_from_dict(tick_to_dict(make_tick()))
        assert back.pose.heading_sigma_deg == 22.0

    def test_track_id_and_velocity_survive(self):
        back = tick_from_dict(tick_to_dict(make_tick()))
        assert back.detections[0].track_id == 7
        assert back.detections[0].radial_velocity_mps == pytest.approx(1.15)

    def test_time_to_contact_recomputes(self):
        """Производная величина не хранится, а выводится — и должна
        совпадать до и после круга."""
        t = make_tick()
        before = t.detections[0].time_to_contact_s
        after = tick_from_dict(tick_to_dict(t)).detections[0].time_to_contact_s
        assert before == pytest.approx(after)
        assert after == pytest.approx(3.42 / 1.15, rel=1e-3)

    def test_minimal_tick(self):
        """Tick без восприятия и маршрута — то, что пишется до старта."""
        t = Tick(
            seq=0, ts=0.0,
            pose=Pose(lat=51.0, lon=71.0, heading_deg=0.0, ts=0.0),
            health=SystemHealth(), detections=[],
        )
        back = tick_from_dict(tick_to_dict(t))
        assert back.risk is None
        assert back.fusion is None
        assert back.detections == []


# ---------------------------------------------------------------------------
# Файлы
# ---------------------------------------------------------------------------

class TestJsonl:
    def test_write_read_roundtrip(self):
        ticks = [make_tick(i) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "walk.jsonl")
            with TickWriter(path, config={"profile_name": "test"},
                            route=make_route(), meta={"weather": "clear"}) as w:
                for t in ticks:
                    w.write(t)

            header, stream = read_ticks(path)
            got = list(stream)

        assert header["_format_version"] == FORMAT_VERSION
        assert header["meta"]["weather"] == "clear"
        assert header["route"]["total_distance_m"] == pytest.approx(2150.4)
        assert len(got) == 5
        assert [t.seq for t in got] == [0, 1, 2, 3, 4]
        assert got[0].fusion.action == Action.BEAR_RIGHT

    def test_gzip_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "walk.jsonl.gz")
            with TickWriter(path, config={}) as w:
                w.write(make_tick())
            header, stream = read_ticks(path)
            assert len(list(stream)) == 1

    def test_version_mismatch_refuses(self):
        """Читать запись другого формата нельзя: метрики статьи,
        посчитанные по неверно разобранному логу, выглядят правдоподобно."""
        import json
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"_format_version": "0.1", "config": {}}) + "\n")
            with pytest.raises(ValueError, match="Формат записи"):
                read_ticks(path)

    def test_missing_field_fails_loudly(self):
        """Старая запись без нового поля должна падать, а не читаться
        с молчаливым значением по умолчанию."""
        d = tick_to_dict(make_tick())
        del d["pose"]["heading_sigma_deg"]
        with pytest.raises(KeyError):
            tick_from_dict(d)

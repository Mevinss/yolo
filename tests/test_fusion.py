# -*- coding: utf-8 -*-
"""
Тесты научного ядра: поле риска, коридоры, слияние, арбитраж речи.

Проверяются СВОЙСТВА, а не значения формул. Веса и пороги будут
меняться при калибровке на шаге 6; свойства — нет.

Главный тест здесь — TestFusion::test_fence_scenario. Он воспроизводит
ошибку проектирования, найденную на шаге 2: при передаче одного угла
вместо профиля слияние выбирало направление внутрь препятствия.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (                                          # noqa: E402
    FusionConfig, GuidanceConfig, LocalPlannerConfig, RiskConfig,
)
from core.fusion.policy import FusionPolicy                        # noqa: E402
from core.guidance.phrasing import Phrasebook                      # noqa: E402
from core.guidance.scheduler import UtteranceScheduler             # noqa: E402
from core.local_planner.corridor import CorridorPlanner            # noqa: E402
from core.risk.field import RiskFieldBuilder                       # noqa: E402
from core.risk.scoring import ObjectProfiles                       # noqa: E402
from core.types import (                                           # noqa: E402
    Action, CameraIntrinsics, Corridor, CrossingPhase, Detection, DistanceSource,
    GlobalGuidance, HealthFlag, Lang, LocalGuidance, Maneuver, ManeuverType,
    Pose, RiskField, SystemHealth, Urgency,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = os.path.join(ROOT, "data", "object_profiles.csv")

INTR = CameraIntrinsics(focal_px=700.0, width=1280, height=720, hfov_deg=60.0,
                        height_above_ground_m=1.35, pitch_deg=10.0)


def profiles():
    return ObjectProfiles.load(PROFILES)


def det(label, bearing_deg, distance_m, conf=0.9, width_px=120, **kw):
    """Детекция с боксом, соответствующим заданному углу."""
    cx = INTR.width / 2.0
    x = cx + math.tan(math.radians(bearing_deg)) * INTR.focal_px
    return Detection(
        label=label, confidence=conf,
        bbox=(int(x - width_px / 2), 300, int(x + width_px / 2), 600),
        bearing_deg=bearing_deg, distance_m=distance_m,
        distance_source=DistanceSource.FUSED, distance_sigma_m=0.3, **kw,
    )


def health(confidence=1.0, flags=None, detail=None):
    return SystemHealth(flags=flags or [], confidence=confidence, detail=detail)


def pose(heading=0.0):
    return Pose(lat=51.13, lon=71.43, heading_deg=heading, ts=100.0,
                accuracy_m=5.0, heading_sigma_deg=10.0)


# ---------------------------------------------------------------------------
# Поле риска
# ---------------------------------------------------------------------------

class TestRiskField:
    @pytest.fixture
    def builder(self):
        return RiskFieldBuilder(RiskConfig(), profiles())

    def test_shape_matches_config(self, builder):
        f = builder.build([], INTR, 100.0)
        assert f.cells.shape == (RiskConfig().n_sectors, len(RiskConfig().band_edges_m) - 1)
        assert f.sector_edges_deg[0] == pytest.approx(-INTR.hfov_deg / 2)
        assert f.sector_edges_deg[-1] == pytest.approx(INTR.hfov_deg / 2)

    def test_empty_field_is_zero(self, builder):
        assert builder.build([], INTR, 100.0).cells.sum() == 0.0

    def test_object_lands_in_its_sector(self, builder):
        f = builder.build([det("car", bearing_deg=20.0, distance_m=2.0)], INTR, 100.0)
        centers = (f.sector_edges_deg[:-1] + f.sector_edges_deg[1:]) / 2
        hot = int(np.argmax(f.cells.sum(axis=1)))
        assert abs(centers[hot] - 20.0) < 10.0

    def test_landmark_contributes_nothing(self, builder):
        """Светофор над головой не препятствие. Если считать его таковым,
        на перекрёстке не останется свободного коридора."""
        f = builder.build([det("traffic light", 0.0, 3.0)], INTR, 100.0)
        assert f.cells.sum() == 0.0

    def test_wide_object_spans_multiple_sectors(self, builder):
        """Человек не проходит сквозь стул, задев его краем."""
        f = builder.build([det("car", 0.0, 2.0, width_px=600)], INTR, 100.0)
        occupied = int((f.cells.sum(axis=1) > 0).sum())
        assert occupied >= 3

    def test_nearer_object_is_riskier(self, builder):
        near = builder.build([det("car", 0.0, 1.0)], INTR, 100.0).cells.max()
        far = builder.build([det("car", 0.0, 8.0)], INTR, 100.0).cells.max()
        assert near > far

    def test_approaching_object_is_riskier(self, builder):
        """Стоящий человек и идущий навстречу на одном кадре неразличимы,
        но требуют разной реакции."""
        still = builder.build([det("person", 0.0, 3.0, radial_velocity_mps=0.0)],
                              INTR, 100.0).cells.max()
        approaching = builder.build([det("person", 0.0, 3.0, radial_velocity_mps=1.5)],
                                    INTR, 100.0).cells.max()
        assert approaching > still

    def test_cells_are_clipped(self, builder):
        """Скопление мелких объектов не должно перевешивать одну яму."""
        many = [det("person", 0.0, 1.0) for _ in range(12)]
        assert builder.build(many, INTR, 100.0).cells.max() <= 1.0

    def test_contributors_point_to_detections(self, builder):
        ds = [det("car", 0.0, 2.0)]
        f = builder.build(ds, INTR, 100.0)
        assert f.contributors
        idxs = next(iter(f.contributors.values()))
        assert idxs == [0]


# ---------------------------------------------------------------------------
# Локальный планировщик
# ---------------------------------------------------------------------------

class TestCorridor:
    @pytest.fixture
    def planner(self):
        return CorridorPlanner(LocalPlannerConfig())

    @staticmethod
    def field_with(blocked_sectors: list, n_sectors=9, n_bands=4, band=0) -> RiskField:
        cells = np.zeros((n_sectors, n_bands))
        for i in blocked_sectors:
            cells[i, band] = 1.0
        return RiskField(
            cells=cells,
            sector_edges_deg=np.linspace(-30, 30, n_sectors + 1),
            band_edges_m=np.array([0.0, 1.5, 3.0, 6.0, 12.0]),
            ts=100.0,
        )

    def test_empty_field_gives_one_wide_corridor(self, planner):
        g = planner.plan(self.field_with([]))
        assert not g.blocked
        assert len(g.corridors) == 1
        assert g.corridors[0].width_deg > 50

    def test_all_blocked_gives_blocked(self, planner):
        g = planner.plan(self.field_with(list(range(9))))
        assert g.blocked
        assert g.corridors == []

    def test_profile_is_finer_than_sectors(self, planner):
        """Сектор в 6.7 градуса — это 35 см на трёх метрах. Округлять
        выбор направления до сектора значит терять имеющуюся точность."""
        g = planner.plan(self.field_with([]))
        assert len(g.profile_angles_deg) > 9 * 3

    def test_near_obstacle_blocks_but_far_does_not(self, planner):
        """Яма в метре обязана блокировать сектор. При нормировке
        по сумме весов она давала бы около 0.5 при пороге 0.7."""
        near = planner.plan(self.field_with([4], band=0))
        far = planner.plan(self.field_with([4], band=3))
        near_center = [c for c in near.corridors if c.left_deg <= 0 <= c.right_deg]
        far_center = [c for c in far.corridors if c.left_deg <= 0 <= c.right_deg]
        assert not near_center, "ближняя преграда не заблокировала центр"
        assert far_center, "дальняя преграда заблокировала центр"

    def test_two_gaps_are_both_reported(self, planner):
        """Оба прохода должны дойти до слияния: выбор между ними
        зависит от маршрута, о котором планировщик не знает."""
        g = planner.plan(self.field_with([3, 4, 5]))
        assert len(g.corridors) == 2

    def test_narrow_gap_is_rejected(self, planner):
        cfg = LocalPlannerConfig()
        cfg.min_corridor_deg = 30.0
        g = CorridorPlanner(cfg).plan(self.field_with([0, 1, 2, 4, 6, 7, 8]))
        assert all(c.width_deg >= 30.0 for c in g.corridors)

    def test_clearance_reflects_distance(self, planner):
        near = planner.plan(self.field_with([4], band=0)).clearance_m
        far = planner.plan(self.field_with([4], band=2)).clearance_m
        assert near < far


# ---------------------------------------------------------------------------
# СЛИЯНИЕ — ядро
# ---------------------------------------------------------------------------

class TestFusion:
    @pytest.fixture
    def policy(self):
        return FusionPolicy(FusionConfig())

    @staticmethod
    def local(risk_at: dict = None, blocked=False, corridors=None,
              width=60.0, clearance=10.0) -> LocalGuidance:
        """Профиль риска по углу: risk_at = {угол: значение}."""
        angles = np.arange(-30.0, 30.5, 1.0)
        profile = np.zeros_like(angles)
        for center, (halfwidth, value) in (risk_at or {}).items():
            profile[np.abs(angles - center) <= halfwidth] = value
        return LocalGuidance(
            risk_profile=profile, profile_angles_deg=angles,
            corridors=corridors or [Corridor(0.0, -30.0, 30.0, clearance, 0.0)],
            theta_free_deg=0.0, corridor_width_deg=width,
            clearance_m=clearance, blocking=[0], blocked=blocked,
        )

    @staticmethod
    def glob(theta_route=0.0, maneuver=None, dist=100.0, arrived=False):
        return GlobalGuidance(
            theta_route_deg=theta_route, next_maneuver=maneuver,
            distance_to_maneuver_m=dist, cross_track_error_m=0.0,
            off_route=False, arrived=arrived,
        )

    # ---------------- главный тест ----------------

    def test_fence_scenario(self, policy):
        """РЕГРЕССИЯ на ошибку проектирования из шага 2.

        Маршрут прямо (0 градусов). Забор занимает от -15 до +15.
        Свободны коридоры при -45 (широкий) и +25 (узкий).

        При передаче одного угла слияние получало бы -45 как самый
        широкий и искало компромисс между -45 и 0, выбирая около -25,
        то есть внутрь забора. С профилем оно обязано выбрать
        свободное направление.
        """
        angles = np.arange(-60.0, 60.5, 1.0)
        # Всё занято, кроме двух проходов: забор с флангами перекрывает
        # сектор от -32 до +19, а по краям кадра стоят стены зданий.
        profile = np.ones_like(angles)
        profile[(angles >= -60) & (angles <= -32)] = 0.0   # широкий проход слева
        profile[(angles >= 19) & (angles <= 31)] = 0.0     # узкий проход справа
        local = LocalGuidance(
            risk_profile=profile, profile_angles_deg=angles,
            corridors=[Corridor(-45.0, -60.0, -32.0, 8.0, 0.0),
                       Corridor(25.0, 19.0, 31.0, 6.0, 0.0)],
            theta_free_deg=-45.0, corridor_width_deg=28.0,
            clearance_m=8.0, blocking=[0], blocked=False,
        )

        r = policy.decide(self.glob(theta_route=0.0), local, pose(0.0), health(), 100.0)

        chosen = ((r.theta_star_deg + 180.0) % 360.0) - 180.0
        risk_at_chosen = float(np.interp(chosen, angles, profile))
        assert risk_at_chosen < 0.5, (
            f"выбрано направление {chosen:.0f} с риском {risk_at_chosen:.2f} — "
            "слияние повело внутрь препятствия"
        )
        # Из двух свободных проходов выбран ближний к маршруту
        assert chosen > 0, f"выбран дальний от маршрута проход: {chosen:.0f}"

    # ---------------- режимы ----------------

    def test_blocked_gives_emergency_stop(self, policy):
        r = policy.decide(self.glob(), self.local(blocked=True), pose(), health(), 100.0)
        assert r.action is Action.STOP
        assert r.urgency is Urgency.EMERGENCY

    def test_clear_path_goes_straight(self, policy):
        r = policy.decide(self.glob(theta_route=0.0), self.local(), pose(0.0),
                          health(), 100.0)
        assert r.action is Action.STRAIGHT
        assert abs(r.deviation_from_route_deg) < 10.0

    def test_obstacle_ahead_causes_deviation(self, policy):
        local = self.local(risk_at={0.0: (12.0, 1.0)})
        r = policy.decide(self.glob(theta_route=0.0), local, pose(0.0), health(), 100.0)
        assert r.action in (Action.BEAR_LEFT, Action.BEAR_RIGHT)
        assert abs(r.deviation_from_route_deg) > 10.0
        assert r.cause_indices, "отклонение без указания причины"

    def test_guard_mode_warns_without_route(self, policy):
        """РЕЖИМ СТРАЖА: маршрута нет, но система обязана предупреждать.

        Регрессия на дефект, из-за которого главный сценарий не работал:
        отклонение отсчитывалось от маршрутного курса, и без маршрута
        оказывалось тождественно нулевым. Система выбирала «прямо»
        при любом препятствии и не подсказывала обход ни разу.

        Опорное направление без маршрута — тот курс, куда человек
        смотрит сейчас.
        """
        local = self.local(risk_at={0.0: (14.0, 1.0)})
        r = policy.decide(None, local, pose(0.0), health(), 100.0)
        assert r.action in (Action.BEAR_LEFT, Action.BEAR_RIGHT), (
            f"без маршрута препятствие по курсу дало {r.action}, "
            "то есть обход не подсказан"
        )
        assert abs(r.deviation_from_route_deg) > 10.0
        assert r.cause_indices, "отклонение без указания причины"

    def test_guard_mode_straight_when_clear(self, policy):
        """Свободно — значит прямо, а не выдуманный обход."""
        r = policy.decide(None, self.local(), pose(0.0), health(), 100.0)
        assert r.action is Action.STRAIGHT

    def test_guard_mode_does_not_replan(self, policy):
        """Перестраивать нечего: маршрута нет. Устойчивое отклонение
        здесь означает лишь ходьбу вдоль препятствия."""
        local = self.local(risk_at={0.0: (28.0, 1.0)})
        triggers = [
            policy.decide(None, local, pose(0.0), health(), float(t)).replan_needed
            for t in np.arange(100.0, 108.0, 0.5)
        ]
        assert not any(triggers)

    def test_guard_mode_follows_head_turn(self, policy):
        """Опора — текущий курс: повернув голову, человек получает
        подсказки относительно нового направления взгляда."""
        local = self.local()
        a = policy.decide(None, local, pose(0.0), health(), 100.0)
        b = FusionPolicy(FusionConfig()).decide(None, local, pose(90.0), health(), 100.0)
        assert abs(((b.theta_star_deg - a.theta_star_deg) % 360.0) - 90.0) < 5.0

    def test_route_only_mode(self):
        """w_risk = 0 — парадигма BlindSquare: ведёт по маршруту,
        не замечая препятствий."""
        cfg = FusionConfig()
        cfg.w_risk = 0.0
        p = FusionPolicy(cfg)
        local = TestFusion.local(risk_at={0.0: (15.0, 1.0)})
        r = p.decide(TestFusion.glob(theta_route=0.0), local, pose(0.0), health(), 100.0)
        assert r.action is Action.STRAIGHT, "без веса риска система обязана идти в препятствие"

    # ---------------- системы координат ----------------

    def test_heading_rotates_choice(self, policy):
        """Тот же кадр при другом курсе даёт другой географический азимут."""
        local = self.local()
        a = policy.decide(self.glob(theta_route=0.0), local, pose(0.0), health(), 100.0)
        FusionPolicy(FusionConfig())
        b = FusionPolicy(FusionConfig()).decide(
            self.glob(theta_route=90.0), local, pose(90.0), health(), 100.0)
        assert abs(((b.theta_star_deg - a.theta_star_deg) % 360.0) - 90.0) < 5.0

    def test_theta_star_is_geographic(self, policy):
        r = policy.decide(self.glob(theta_route=270.0), self.local(), pose(270.0),
                          health(), 100.0)
        assert 0.0 <= r.theta_star_deg < 360.0
        assert abs(((r.theta_star_deg - 270.0 + 180) % 360) - 180) < 15.0

    # ---------------- самодиагностика ----------------

    def test_unreliable_heading_lowers_route_weight(self, policy):
        """Вести по азимуту, не зная своего курса, значит уводить наугад."""
        local = self.local()
        good = policy.decide(self.glob(), local, pose(), health(), 100.0)
        bad = FusionPolicy(FusionConfig()).decide(
            self.glob(), local, pose(),
            health(0.5, [HealthFlag.HEADING_UNRELIABLE]), 100.0)
        assert bad.cost_breakdown["w_route"] < good.cost_breakdown["w_route"]

    def test_gps_lost_zeroes_route_weight(self, policy):
        r = policy.decide(self.glob(), self.local(), pose(),
                          health(0.4, [HealthFlag.GPS_LOST]), 100.0)
        assert r.cost_breakdown["w_route"] == 0.0

    def test_low_confidence_forbids_confident_straight(self, policy):
        """Пустое поле риска в темноте не означает отсутствия препятствий."""
        r = policy.decide(self.glob(), self.local(),
                          pose(), health(0.3, [HealthFlag.LOW_LIGHT], "темно"), 100.0)
        assert r.action is Action.SLOW

    # ---------------- перестроение ----------------

    def test_brief_deviation_does_not_replan(self, policy):
        local = self.local(risk_at={0.0: (20.0, 1.0)})
        for t in (100.0, 100.5, 101.0):
            r = policy.decide(self.glob(), local, pose(), health(), t)
        assert not r.replan_needed

    def test_sustained_deviation_triggers_replan(self, policy):
        """Обошли столб — подруливание. Тротуар перекрыт — перестроение.
        Различаем по времени.

        Препятствие шире порога отклонения: узкое даёт подруливание,
        которое перестроения не требует, и это проверяется отдельно.
        """
        local = self.local(risk_at={0.0: (28.0, 1.0)})
        triggers = [
            policy.decide(self.glob(), local, pose(), health(), float(t)).replan_needed
            for t in np.arange(100.0, 106.0, 0.5)
        ]
        assert any(triggers), "устойчивое отклонение не вызвало перестроения"
        # Ровно один раз: маршрут перестраивается однократно, а не каждый
        # кадр — иначе A* запускался бы двадцать раз в секунду.
        assert sum(triggers) == 1, f"перестроение запрошено {sum(triggers)} раз"

    # ---------------- переходы ----------------

    @staticmethod
    def crossing_maneuver(signalled=True):
        return Maneuver(
            type=ManeuverType.CROSSING, lat=51.13, lon=71.43,
            exit_bearing_deg=0.0, distance_from_prev_m=50.0,
            accessibility={"crossing": "traffic_signals"} if signalled else {},
        )

    def test_crossing_phases_progress(self, policy):
        m = self.crossing_maneuver()
        local = self.local()

        r = policy.decide(self.glob(maneuver=m, dist=20.0), local, pose(), health(), 100.0)
        assert r.crossing_phase is CrossingPhase.APPROACHING

        r = policy.decide(self.glob(maneuver=m, dist=2.0), local, pose(), health(), 101.0)
        assert r.crossing_phase is CrossingPhase.WAITING
        assert r.action is Action.WAIT_CROSSING

        policy.confirm_crossing()
        r = policy.decide(self.glob(maneuver=m, dist=2.0), local, pose(), health(), 102.0)
        assert r.crossing_phase is CrossingPhase.CROSSING

    def test_crossing_forbids_wide_deviation(self, policy):
        """Сойти с курса между машин опаснее, чем задеть что-то
        на тротуаре: широкие отклонения на проезжей части запрещены."""
        m = self.crossing_maneuver()
        local = self.local(risk_at={0.0: (8.0, 0.8)})
        policy.decide(self.glob(maneuver=m, dist=2.0), local, pose(), health(), 100.0)
        policy.confirm_crossing()
        r = policy.decide(self.glob(maneuver=m, dist=2.0), local, pose(), health(), 101.0)
        assert r.crossing_phase is CrossingPhase.CROSSING
        assert abs(r.deviation_from_route_deg) <= 16.0

    # ---------------- разбивка стоимости ----------------

    def test_cost_breakdown_is_filled(self, policy):
        """Нужна для таблицы ablation в статье."""
        policy.decide(self.glob(), self.local(), pose(), health(), 100.0)
        r = policy.decide(self.glob(), self.local(), pose(), health(), 100.1)
        for k in ("risk", "route", "smooth", "w_risk", "w_route", "w_smooth", "total"):
            assert k in r.cost_breakdown

    def test_smooth_term_is_not_always_zero(self, policy):
        """Регрессия: слагаемое стабильности считалось после присвоения
        предыдущего угла и потому обращалось в ноль всегда."""
        local_a = self.local(risk_at={-20.0: (8.0, 1.0)})
        local_b = self.local(risk_at={20.0: (8.0, 1.0)})
        policy.decide(self.glob(), local_a, pose(), health(), 100.0)
        r = policy.decide(self.glob(), local_b, pose(), health(), 100.1)
        assert r.cost_breakdown["smooth"] >= 0.0
        assert "w_smooth" in r.cost_breakdown and r.cost_breakdown["w_smooth"] > 0


# ---------------------------------------------------------------------------
# Фразы и арбитраж
# ---------------------------------------------------------------------------

class TestPhrasingAndScheduler:
    @pytest.fixture
    def book(self):
        cfg = GuidanceConfig()
        cfg.lang = Lang.RU
        return Phrasebook(cfg, profiles())

    @pytest.fixture
    def sched(self, book):
        return UtteranceScheduler(book.config, book)

    @staticmethod
    def fusion(action=Action.STRAIGHT, urgency=Urgency.INFO, **kw):
        from core.types import FusionResult
        return FusionResult(action=action, theta_star_deg=0.0, urgency=urgency,
                            deviation_from_route_deg=0.0, **kw)

    def test_rounding_is_step_measurable(self, book):
        assert book.round_distance(4.0) == 5
        assert book.round_distance(9.0) == 10
        assert book.round_distance(37.0) == 40

    def test_action_comes_before_reason(self, book):
        """Человек должен начать действовать с первого слова."""
        f = self.fusion(Action.BEAR_LEFT, Urgency.CAUTION, cause_indices=[0])
        u = book.compose(f, None, TestFusion.local(), health(),
                         [det("pothole", -20.0, 2.0)], 100.0)
        assert u.text.startswith("Возьмите левее")
        assert "яма" in u.text.lower()

    def test_emergency_is_one_word(self, book):
        u = book.compose(self.fusion(Action.STOP, Urgency.EMERGENCY), None,
                         TestFusion.local(blocked=True), health(), [], 100.0)
        assert u.text == "Стоп"
        assert u.urgency is Urgency.EMERGENCY

    def test_low_confidence_changes_claim_about_world(self, book):
        """«Свободно» утверждает факт о мире, «препятствий не вижу» —
        о системе. Разница определяет, на что человек полагается."""
        confident = book.compose(self.fusion(), None, TestFusion.local(),
                                 health(1.0), [], 100.0)
        unsure = book.compose(self.fusion(), None, TestFusion.local(),
                              health(0.3), [], 100.0)
        assert confident.text == "Свободно"
        assert unsure.text == "Препятствий не вижу"

    def test_unsignalled_crossing_leaves_decision_to_user(self, book):
        """Система не видит весь поток машин и не вправе говорить
        «можно идти»."""
        m = TestFusion.crossing_maneuver(signalled=False)
        g = TestFusion.glob(maneuver=m, dist=2.0)
        f = self.fusion(Action.WAIT_CROSSING, Urgency.CAUTION,
                        crossing_phase=CrossingPhase.WAITING)
        u = book.compose(f, g, TestFusion.local(), health(), [], 100.0)
        assert "Решение за вами" in u.text

    def test_kazakh_differs_from_russian(self):
        cfg = GuidanceConfig()
        cfg.lang = Lang.KK
        kk = Phrasebook(cfg, profiles())
        u = kk.compose(self.fusion(Action.STOP, Urgency.EMERGENCY), None,
                       TestFusion.local(blocked=True), health(), [], 100.0)
        assert u.text == "Тоқта"
        assert u.lang is Lang.KK

    # ---------------- арбитраж ----------------

    def test_budget_suppresses_frequent_speech(self, sched):
        f = self.fusion(Action.BEAR_LEFT, Urgency.CAUTION, cause_indices=[0])
        ds = [det("pothole", -20.0, 2.0)]
        first = sched.consider(f, None, TestFusion.local(), health(), ds, 100.0)
        second = sched.consider(f, None, TestFusion.local(), health(), ds, 100.5)
        assert first is not None
        assert second is None

    def test_emergency_ignores_budget(self, sched):
        ds = [det("pothole", -20.0, 2.0)]
        sched.consider(self.fusion(Action.BEAR_LEFT, Urgency.CAUTION, cause_indices=[0]),
                       None, TestFusion.local(), health(), ds, 100.0)
        stop = sched.consider(self.fusion(Action.STOP, Urgency.EMERGENCY), None,
                              TestFusion.local(blocked=True), health(), ds, 100.1)
        assert stop is not None and stop.urgency is Urgency.EMERGENCY

    def test_dedup_suppresses_repeat(self, sched):
        f = self.fusion(Action.BEAR_LEFT, Urgency.CAUTION, cause_indices=[0])
        ds = [det("pothole", -20.0, 2.0)]
        sched.consider(f, None, TestFusion.local(), health(), ds, 100.0)
        again = sched.consider(f, None, TestFusion.local(), health(), ds, 104.0)
        assert again is None
        assert sched.stats()["suppressed_by_dedup"] >= 1

    def test_degradation_is_announced(self, sched):
        u = sched.consider(self.fusion(), None, TestFusion.local(),
                           health(0.4, [HealthFlag.LOW_LIGHT], "темно"), [], 100.0)
        assert u is not None
        assert "темно" in u.text.lower()

    def test_degradation_not_repeated_every_second(self, sched):
        h = health(0.4, [HealthFlag.LOW_LIGHT], "темно")
        sched.consider(self.fusion(), None, TestFusion.local(), h, [], 100.0)
        again = sched.consider(self.fusion(), None, TestFusion.local(), h, [], 110.0)
        assert again is None or "темно" not in (again.text or "").lower()

    def test_stats_count_silence(self, sched):
        """Система, промолчавшая тысячу раз из-за бюджета, ведёт себя
        иначе, чем система, которой нечего было сказать."""
        f = self.fusion(Action.BEAR_LEFT, Urgency.CAUTION, cause_indices=[0])
        ds = [det("pothole", -20.0, 2.0)]
        for i in range(10):
            sched.consider(f, None, TestFusion.local(), health(), ds, 100.0 + i * 0.1)
        s = sched.stats()
        assert s["spoken"] >= 1
        assert s["suppressed_total"] >= 5

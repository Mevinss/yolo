# -*- coding: utf-8 -*-
"""
СЛИЯНИЕ — научное ядро работы.

ДВА РЕЖИМА
----------
Система работает и БЕЗ маршрута. Это не урезанный вариант, а основной
сценарий: человек просто идёт своей дорогой и хочет знать, что перед ним.
Маршрут добавляется, когда нужно куда-то дойти.

    режим СТРАЖА     маршрута нет. Опорное направление — куда человек
                     смотрит сейчас. Система предупреждает о том, что
                     впереди, и подсказывает, куда взять, чтобы обойти.

    режим ВЕДЕНИЯ    маршрут есть. Опорное направление — маршрутный курс.
                     К предупреждениям добавляются манёвры и переходы.

Разница только в том, ОТ ЧЕГО отсчитывается отклонение. Всё остальное —
поле риска, поиск коридора, арбитраж речи — работает одинаково.

Раньше опорным направлением всегда был маршрут, и без него отклонение
оказывалось тождественно нулевым: система выбирала команду «прямо»
при любом препятствии и не предупреждала об обходе вообще. Режим стража
не работал, хотя формально существовал.

ПРОБЛЕМА
--------
Глобальный уровень говорит «идите прямо, через 40 м направо».
Локальный уровень говорит «прямо нельзя, впереди яма, свободно слева».

Существующие системы решают это одним из двух способов, и оба плохи:

  * Маршрутные приложения (BlindSquare, Lazarillo, Google Maps) не видят
    препятствий вообще — ведут в яму.
  * Приложения распознавания (Lookout, Seeing AI) не знают маршрута —
    предупредят о яме, но человек уйдёт не туда.

Наивная последовательная схема «сначала маршрут, потом объезд» тоже
не работает: при каждом препятствии человек сходит с маршрута, а вернуться
на него без зрения трудно. Накопленное отклонение приводит к тому,
что через три препятствия пользователь идёт по другой улице.

РЕШЕНИЕ
-------
Не переключаться между уровнями, а оптимизировать общую стоимость по углу:

    cost(theta) = w_risk   * R(theta)
                + w_route  * |theta - theta_route| / 180
                + w_smooth * |theta - theta_prev|  / 180

    theta* = argmin cost(theta)

Три слагаемых отвечают за три требования:
  R(theta)         — безопасность (не врезаться)
  |theta - route|  — целенаправленность (не сойти с маршрута)
  |theta - prev|   — стабильность (не дёргать человека)

Соотношение w_risk / w_route — это и есть характер системы. Кривая
компромисса «пройденные маршруты против пропущенных опасностей»
при разных соотношениях — центральный график статьи.

СИСТЕМЫ КООРДИНАТ
-----------------
Маршрут задан в географических азимутах (0 = север).
Поле риска — в углах относительно оптической оси камеры.
Перевод делается ЗДЕСЬ и только здесь:

    theta_geo = normalize(pose.heading_deg + theta_camera)

Ошибка в этом месте даёт систему, которая уверенно ведёт не туда,
причём без единого исключения в логах.

ВЕСА ЗАВИСЯТ ОТ СОСТОЯНИЯ СИСТЕМЫ
---------------------------------
Постоянные веса были бы неверны. Когда курс определён плохо, перевод
углов камеры в азимуты недостоверен, и маршрутное слагаемое теряет
смысл: система не знает, куда повёрнута. Вести человека по азимуту,
не зная своего курса, значит уводить его наугад — поэтому вес маршрута
снижается пропорционально достоверности курса.

Когда темно, пустое поле риска перестаёт означать отсутствие
препятствий, и право на команду «прямо» пропадает: остаётся «медленно».

ПЕРЕСТРОЕНИЕ
------------
Отклонение бывает двух видов, и их нельзя путать:
  * подруливание — обошли столб, вернулись на маршрут. Перестраивать не надо.
  * устойчивый обход — тротуар перекрыт, идём другим путём. Надо перестроить.

Различаем по времени: отклонение больше deviation_threshold_deg,
державшееся дольше sustained_deviation_s, поднимает replan_needed.
"""

from __future__ import annotations

import numpy as np

from core.config import FusionConfig
from core.types import (
    Action, CrossingPhase, FusionResult, GlobalGuidance, HealthFlag,
    LocalGuidance, ManeuverType, Pose, SystemHealth, Urgency,
)

#: Отклонение меньше этого считается движением прямо, градусы
STRAIGHT_DEG = 10.0
#: Между этим и предыдущим — «возьмите левее/правее»
BEAR_DEG = 35.0

#: За сколько метров до манёвра команда поворота вытесняет подруливание
MANEUVER_IMMINENT_M = 6.0

#: Ширина коридора ниже этой требует замедлить пользователя, градусы
NARROW_CORRIDOR_DEG = 20.0

#: Доверие ниже этого лишает права на команду «прямо»
LOW_CONFIDENCE = 0.55

#: Разброс курса, при котором вес маршрута падает вдвое, градусы
HEADING_HALF_WEIGHT_DEG = 30.0

#: Во сколько раз повышается вес маршрута на проезжей части
CROSSING_ROUTE_BOOST = 2.5
#: Максимальное отклонение от курса во время перехода, градусы
CROSSING_MAX_DEVIATION_DEG = 15.0

#: Расстояние, с которого объявляется приближение к переходу, м
CROSSING_APPROACH_M = 25.0
#: Расстояние, на котором пользователь считается стоящим у края, м
CROSSING_EDGE_M = 3.0


class FusionPolicy:
    def __init__(self, config: FusionConfig):
        self.config = config
        self._prev_theta_geo: float | None = None
        self._deviation_started_ts: float | None = None
        self._crossing_phase: CrossingPhase | None = None
        self._crossing_confirmed: bool = False

    # -----------------------------------------------------------------

    def confirm_crossing(self) -> None:
        """Пользователь подтвердил начало движения через переход.

        Отдельный вызов, потому что на нерегулируемом переходе решение
        принимает человек: система не видит весь поток машин надёжно
        и не вправе брать эту ответственность на себя.
        """
        self._crossing_confirmed = True

    # -----------------------------------------------------------------

    def decide(
        self,
        global_guidance: GlobalGuidance | None,
        local_guidance: LocalGuidance,
        pose: Pose,
        health: SystemHealth,
        ts: float,
    ) -> FusionResult:
        phase = self._crossing_phase_update(global_guidance, ts)

        # --- 1. Коридора нет вовсе ---------------------------------------
        if local_guidance.blocked:
            self._deviation_started_ts = None
            return FusionResult(
                action=Action.STOP,
                theta_star_deg=self._prev_theta_geo or pose.heading_deg,
                urgency=Urgency.EMERGENCY,
                deviation_from_route_deg=0.0,
                reason="проход перекрыт",
                cause_indices=list(local_guidance.blocking),
                crossing_phase=phase,
                cost_breakdown={},
            )

        # --- 2. Прибытие --------------------------------------------------
        if global_guidance is not None and global_guidance.arrived:
            return FusionResult(
                action=Action.ARRIVED,
                theta_star_deg=pose.heading_deg,
                urgency=Urgency.NAVIGATION,
                deviation_from_route_deg=0.0,
                reason="цель достигнута",
                crossing_phase=phase,
            )

        # --- 3. Ожидание у перехода ---------------------------------------
        if phase is CrossingPhase.WAITING:
            return FusionResult(
                action=Action.WAIT_CROSSING,
                theta_star_deg=(global_guidance.theta_route_deg
                                if global_guidance else pose.heading_deg),
                urgency=Urgency.CAUTION,
                deviation_from_route_deg=0.0,
                reason="переход впереди",
                crossing_phase=phase,
            )

        # --- 4. Стоимость по углу -----------------------------------------
        w_risk, w_route, w_smooth = self._weights(health, phase)

        angles_cam = np.asarray(local_guidance.profile_angles_deg, dtype=float)
        risk = np.asarray(local_guidance.risk_profile, dtype=float)
        angles_geo = (pose.heading_deg + angles_cam) % 360.0

        # Опорное направление: маршрутный курс, а без маршрута — тот,
        # куда человек смотрит сейчас. Без этого в режиме стража
        # отклонение всегда равнялось нулю, и подсказка «возьмите левее»
        # не звучала ни разу, сколько бы препятствий ни было впереди.
        guard_mode = global_guidance is None
        theta_ref = (pose.heading_deg if guard_mode
                     else global_guidance.theta_route_deg)

        cost = w_risk * risk

        if w_route > 0.0:
            dev_route = np.abs(self._angular_diff(angles_geo, theta_ref))
            if phase is CrossingPhase.CROSSING:
                # На проезжей части широкие отклонения запрещены:
                # сойти с курса между машин опаснее, чем задеть
                # что-то на тротуаре.
                cost = cost + np.where(
                    dev_route > CROSSING_MAX_DEVIATION_DEG, 1e3, 0.0
                )
            cost = cost + w_route * dev_route / 180.0
        else:
            dev_route = np.zeros_like(angles_geo)

        prev_theta = self._prev_theta_geo
        if prev_theta is not None and w_smooth > 0.0:
            dev_prev = np.abs(self._angular_diff(angles_geo, prev_theta))
            cost = cost + w_smooth * dev_prev / 180.0

        best = int(np.argmin(cost))
        theta_star = float(angles_geo[best])
        deviation = float(dev_route[best])
        # знак отклонения нужен для выбора «левее» или «правее»
        signed_dev = float(self._angular_diff(theta_star, theta_ref))

        self._prev_theta_geo = theta_star

        # --- 5. Устойчивость отклонения -----------------------------------
        # Перестраивать нечего, если маршрута нет: в режиме стража
        # устойчивый обход означает лишь, что человек идёт вдоль
        # препятствия, а не что маршрут перекрыт.
        replan = (False if guard_mode
                  else self._update_deviation_timer(deviation, ts))

        # --- 6. Команда ----------------------------------------------------
        action, urgency = self._classify_action(
            signed_dev, global_guidance, local_guidance, health, phase
        )

        cause = self._cause_indices(local_guidance, angles_cam, risk, signed_dev)

        return FusionResult(
            action=action,
            theta_star_deg=theta_star,
            urgency=urgency,
            deviation_from_route_deg=signed_dev,
            reason=self._reason(action, local_guidance, health),
            cause_indices=cause,
            replan_needed=replan,
            crossing_phase=phase,
            cost_breakdown={
                "risk": round(float(w_risk * risk[best]), 4),
                "route": round(float(w_route * deviation / 180.0), 4),
                # prev_theta взят ДО присвоения theta_star: иначе
                # слагаемое стабильности всегда обращалось бы в ноль
                "smooth": round(float(
                    w_smooth * abs(self._angular_diff(theta_star, prev_theta)) / 180.0
                ), 4) if prev_theta is not None else 0.0,
                "w_risk": round(w_risk, 3),
                "w_route": round(w_route, 3),
                "w_smooth": round(w_smooth, 3),
                "total": round(float(cost[best]), 4),
            },
        )

    # -----------------------------------------------------------------
    # Веса
    # -----------------------------------------------------------------

    def _weights(self, health: SystemHealth, phase: CrossingPhase | None) -> tuple:
        """Веса стоимости с поправкой на состояние системы и фазу перехода."""
        c = self.config
        w_risk, w_route, w_smooth = c.w_risk, c.w_route, c.w_smooth

        # Ненадёжный курс обесценивает маршрутное слагаемое: перевод
        # углов камеры в азимуты становится недостоверным.
        if HealthFlag.HEADING_UNRELIABLE in health.flags:
            w_route *= 0.25
        if HealthFlag.GPS_LOST in health.flags:
            w_route = 0.0

        # На проезжей части удержание курса важнее обхода мелочей
        if phase is CrossingPhase.CROSSING:
            w_route *= CROSSING_ROUTE_BOOST

        return w_risk, w_route, w_smooth

    # -----------------------------------------------------------------
    # Переход
    # -----------------------------------------------------------------

    def _crossing_phase_update(
        self, glob: GlobalGuidance | None, ts: float
    ) -> CrossingPhase | None:
        """Автомат состояний перехода."""
        if glob is None or glob.next_maneuver is None:
            if self._crossing_phase is CrossingPhase.CROSSING:
                return CrossingPhase.CROSSING       # маршрут пропал прямо на дороге
            self._crossing_phase = None
            self._crossing_confirmed = False
            return None

        is_crossing = glob.next_maneuver.type is ManeuverType.CROSSING
        d = glob.distance_to_maneuver_m

        if not is_crossing:
            if self._crossing_phase in (CrossingPhase.CROSSING, CrossingPhase.WAITING):
                self._crossing_phase = CrossingPhase.COMPLETED
                self._crossing_confirmed = False
            elif self._crossing_phase is CrossingPhase.COMPLETED:
                self._crossing_phase = None
            return self._crossing_phase

        if self._crossing_phase is CrossingPhase.CROSSING:
            return CrossingPhase.CROSSING
        if d <= CROSSING_EDGE_M:
            self._crossing_phase = (CrossingPhase.CROSSING if self._crossing_confirmed
                                    else CrossingPhase.WAITING)
        elif d <= CROSSING_APPROACH_M:
            self._crossing_phase = CrossingPhase.APPROACHING
        return self._crossing_phase

    # -----------------------------------------------------------------
    # Команда
    # -----------------------------------------------------------------

    def _classify_action(
        self,
        signed_dev: float,
        glob: GlobalGuidance | None,
        local: LocalGuidance,
        health: SystemHealth,
        phase: CrossingPhase | None,
    ) -> tuple:
        """Отклонение и обстановка -> команда пользователю и её срочность."""
        # Манёвр маршрута вытесняет подруливание: если поворот вот-вот,
        # человеку нужна команда поворота, а не коррекция на пару градусов.
        if glob is not None and glob.next_maneuver is not None \
                and glob.distance_to_maneuver_m <= MANEUVER_IMMINENT_M:
            t = glob.next_maneuver.type
            if t in (ManeuverType.TURN_LEFT, ManeuverType.SLIGHT_LEFT):
                return Action.TURN_LEFT, Urgency.NAVIGATION
            if t in (ManeuverType.TURN_RIGHT, ManeuverType.SLIGHT_RIGHT):
                return Action.TURN_RIGHT, Urgency.NAVIGATION

        # Узкий проход или мало времени до преграды — замедлиться
        if local.clearance_m <= 1.0:
            return Action.SLOW, Urgency.CAUTION
        if local.corridor_width_deg < NARROW_CORRIDOR_DEG:
            return Action.SLOW, Urgency.CAUTION

        # При низком доверии право на уверенное «прямо» пропадает:
        # пустое поле риска в темноте не означает отсутствия препятствий
        if health.confidence < LOW_CONFIDENCE and abs(signed_dev) < STRAIGHT_DEG:
            return Action.SLOW, Urgency.CAUTION

        if abs(signed_dev) < STRAIGHT_DEG:
            return Action.STRAIGHT, Urgency.INFO
        if abs(signed_dev) < BEAR_DEG:
            return ((Action.BEAR_RIGHT if signed_dev > 0 else Action.BEAR_LEFT),
                    Urgency.CAUTION)
        return ((Action.TURN_RIGHT if signed_dev > 0 else Action.TURN_LEFT),
                Urgency.CAUTION)

    def _reason(self, action: Action, local: LocalGuidance,
                health: SystemHealth) -> str | None:
        if action in (Action.BEAR_LEFT, Action.BEAR_RIGHT, Action.STOP):
            return "препятствие по курсу"
        if action is Action.SLOW:
            if health.confidence < LOW_CONFIDENCE:
                return health.detail or "низкая уверенность"
            return "узкий проход"
        return None

    def _cause_indices(self, local: LocalGuidance, angles_cam: np.ndarray,
                       risk: np.ndarray, signed_dev: float) -> list:
        """Из-за чего отклонились. Пусто, если идём прямо.

        Объяснимость здесь не украшение: непонятная команда исполняется
        хуже понятной, а «возьмите левее — впереди яма» человек выполнит
        увереннее, чем просто «возьмите левее».
        """
        if abs(signed_dev) < STRAIGHT_DEG:
            return []
        return list(local.blocking[:3])

    # -----------------------------------------------------------------
    # Устойчивое отклонение
    # -----------------------------------------------------------------

    def _update_deviation_timer(self, deviation: float, ts: float) -> bool:
        if deviation < self.config.deviation_threshold_deg:
            self._deviation_started_ts = None
            return False
        if self._deviation_started_ts is None:
            self._deviation_started_ts = ts
            return False
        if ts - self._deviation_started_ts >= self.config.sustained_deviation_s:
            self._deviation_started_ts = None       # не повторять каждый кадр
            return True
        return False

    # -----------------------------------------------------------------
    # Углы
    # -----------------------------------------------------------------

    def _to_geographic(self, theta_camera_deg: float, heading_deg: float) -> float:
        """Углы камеры -> географический азимут. Единственная точка перевода."""
        return (heading_deg + theta_camera_deg) % 360.0

    @staticmethod
    def _angular_diff(a, b):
        """Кратчайшая разность азимутов в диапазоне [-180, 180].

        Работает и со скалярами, и с массивами: стоимость считается
        сразу по всей угловой сетке.
        """
        return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0

    def reset(self) -> None:
        self._prev_theta_geo = None
        self._deviation_started_ts = None
        self._crossing_phase = None
        self._crossing_confirmed = False

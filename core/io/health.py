# -*- coding: utf-8 -*-
"""
Монитор состояния системы.

ПОЧЕМУ ЭТО ОТДЕЛЬНЫЙ СЛОЙ
-------------------------
Деградация — сквозное свойство: она возникает в io (GPS, компас),
в perception (темнота, отсутствие depth), в сети (задержка) и в routing
(маршрут не построен). Если каждый слой сообщает о своих бедах сам,
пользователь получает поток разрозненных жалоб.

Монитор собирает признаки в одно состояние и отвечает на единственный
вопрос: насколько системе можно доверять прямо сейчас.

ГЛАВНОЕ ТРЕБОВАНИЕ
------------------
Незрячий не может проверить, работает ли приложение. Молчание системы
неотличимо от «впереди чисто». Поэтому деградация обязана объявляться,
а не подразумеваться: аид, отказывающий молча, вытесняет собственную
осторожность пользователя, ничего не давая взамен.

Отсюда два правила:
  1. О переходе в деградированное состояние сообщается однократно и сразу;
  2. Пока деградация держится, напоминание повторяется не чаще, чем
     раз в reminder_interval_s — иначе речь превращается в шум,
     а слух незрячему нужен для ориентации.

ПОРОГИ
------
Числа ниже — начальные оценки, подлежащие проверке на шаге 6.
Порог освещённости взят из работы предшественника, где полнота
детекции падала до 41 процента ниже 50 люкс: ниже этой границы
отсутствие детекций перестаёт означать отсутствие препятствий.
"""

from __future__ import annotations

from typing import Optional

from core.types import HealthFlag, SystemHealth

#: Точность GPS хуже этого — позиция на маршруте недостоверна, м
GPS_DEGRADED_M = 20.0
GPS_LOST_M = 50.0

#: Разброс курса выше этого — перевод углов камеры в азимуты ненадёжен
HEADING_UNRELIABLE_DEG = 35.0

#: Освещённость ниже этого — детекция теряет полноту (см. докстроку модуля)
LOW_LIGHT_LUX = 50.0
#: Оценка по кадру, когда датчика освещённости нет: средняя яркость 0..255
LOW_LIGHT_MEAN_INTENSITY = 45.0

#: Кадров не было дольше этого — камера считается вставшей, с.
#:
#: Полторы секунды оказались слишком строгим порогом: захват кадра
#: в Expo Go занимает сотни миллисекунд, и при обычной работе система
#: то и дело объявляла камеру неисправной. Ложная тревога обесценивает
#: настоящую, поэтому порог поднят до величины, которую нормальная
#: съёмка не достигает, но которая всё ещё короче двух шагов человека.
CAMERA_STALL_S = 2.5

#: Сквозная задержка выше этой выводит реакцию за человеческое окно, мс
HIGH_LATENCY_MS = 300.0

#: Вклад каждого флага в снижение доверия
CONFIDENCE_PENALTY = {
    HealthFlag.GPS_DEGRADED: 0.15,
    HealthFlag.GPS_LOST: 0.35,
    HealthFlag.HEADING_UNRELIABLE: 0.30,
    HealthFlag.LOW_LIGHT: 0.40,
    HealthFlag.CAMERA_STALLED: 0.60,
    HealthFlag.DEPTH_UNAVAILABLE: 0.15,
    HealthFlag.HIGH_LATENCY: 0.20,
    HealthFlag.ROUTE_UNAVAILABLE: 0.20,
    HealthFlag.OFF_ROUTE_NO_REPLAN: 0.25,
}

#: Человекочитаемые причины для озвучки
FLAG_DETAIL_RU = {
    HealthFlag.GPS_DEGRADED: "связь со спутниками слабая",
    HealthFlag.GPS_LOST: "положение потеряно",
    HealthFlag.HEADING_UNRELIABLE: "направление определяется неточно",
    HealthFlag.LOW_LIGHT: "темно",
    HealthFlag.CAMERA_STALLED: "камера не отвечает",
    HealthFlag.DEPTH_UNAVAILABLE: "глубина недоступна",
    HealthFlag.HIGH_LATENCY: "система реагирует с задержкой",
    HealthFlag.ROUTE_UNAVAILABLE: "маршрут не построен",
    HealthFlag.OFF_ROUTE_NO_REPLAN: "не удаётся вернуться на маршрут",
}


class HealthMonitor:
    def __init__(self, reminder_interval_s: float = 30.0, indoor: bool = False):
        self.reminder_interval_s = reminder_interval_s
        # В помещении спутников не видно, и это не неисправность.
        # Сообщать «положение потеряно» человеку, который ходит
        # по своей квартире, значит приучать не слушать систему.
        self.indoor = indoor
        self._degraded_since: Optional[float] = None
        self._last_frame_ts: Optional[float] = None
        self._last_announced_ts: Optional[float] = None

    def assess(
        self,
        pose,
        ts: float,
        *,
        mean_intensity: Optional[float] = None,
        lux: Optional[float] = None,
        depth_available: bool = True,
        route_available: bool = True,
        off_route_no_replan: bool = False,
        last_latency_ms: Optional[float] = None,
    ) -> SystemHealth:
        """Собрать флаги деградации и вычислить доверие к системе."""
        flags: list = []

        # --- положение ---
        if not self.indoor:
            if pose is None or pose.accuracy_m >= GPS_LOST_M:
                flags.append(HealthFlag.GPS_LOST)
            elif pose.accuracy_m >= GPS_DEGRADED_M:
                flags.append(HealthFlag.GPS_DEGRADED)

        # --- курс ---
        if pose is not None and pose.heading_sigma_deg >= HEADING_UNRELIABLE_DEG:
            flags.append(HealthFlag.HEADING_UNRELIABLE)

        # --- освещённость ---
        # Датчик света точнее, но есть не всегда; яркость кадра —
        # запасная оценка того же самого.
        if lux is not None:
            if lux < LOW_LIGHT_LUX:
                flags.append(HealthFlag.LOW_LIGHT)
        elif mean_intensity is not None and mean_intensity < LOW_LIGHT_MEAN_INTENSITY:
            flags.append(HealthFlag.LOW_LIGHT)

        # --- камера ---
        if self._last_frame_ts is not None and ts - self._last_frame_ts > CAMERA_STALL_S:
            flags.append(HealthFlag.CAMERA_STALLED)
        self._last_frame_ts = ts

        # --- прочее ---
        if not depth_available:
            flags.append(HealthFlag.DEPTH_UNAVAILABLE)
        if not route_available:
            flags.append(HealthFlag.ROUTE_UNAVAILABLE)
        if off_route_no_replan:
            flags.append(HealthFlag.OFF_ROUTE_NO_REPLAN)
        if last_latency_ms is not None and last_latency_ms > HIGH_LATENCY_MS:
            flags.append(HealthFlag.HIGH_LATENCY)

        # Доверие перемножается, а не вычитается: два независимых отказа
        # ухудшают положение сильнее, чем сумма их вкладов. Темнота
        # при потерянном курсе — это не «двадцать плюс сорок процентов»,
        # а положение, в котором системе верить нельзя вовсе.
        confidence = 1.0
        for f in flags:
            confidence *= (1.0 - CONFIDENCE_PENALTY.get(f, 0.1))

        if flags and self._degraded_since is None:
            self._degraded_since = ts
        elif not flags:
            self._degraded_since = None
            self._last_announced_ts = None

        return SystemHealth(
            flags=flags,
            confidence=round(confidence, 4),
            degraded_since_ts=self._degraded_since,
            detail=self._detail(flags),
        )

    @staticmethod
    def _detail(flags: list) -> Optional[str]:
        """Причина для озвучки: называется САМАЯ тяжёлая, а не все сразу.

        Перечислять пользователю четыре неисправности бессмысленно —
        действие от этого не меняется, а речь занята.
        """
        if not flags:
            return None
        worst = max(flags, key=lambda f: CONFIDENCE_PENALTY.get(f, 0.0))
        return FLAG_DETAIL_RU.get(worst)

    def should_announce(self, health: SystemHealth, ts: float) -> bool:
        """Пора ли сообщить о деградации.

        True при переходе в деградированное состояние и далее не чаще
        одного раза в reminder_interval_s, пока оно держится.

        Возврат в норму отдельно не объявляется: сообщение «всё
        восстановилось» само по себе занимает речь, а изменение
        поведения системы пользователь замечает по формулировкам.
        """
        if not health.degraded:
            self._last_announced_ts = None
            return False
        if self._last_announced_ts is None:
            self._last_announced_ts = ts
            return True
        if ts - self._last_announced_ts >= self.reminder_interval_s:
            self._last_announced_ts = ts
            return True
        return False

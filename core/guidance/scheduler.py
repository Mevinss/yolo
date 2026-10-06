# -*- coding: utf-8 -*-
"""
Арбитраж речи: какая реплика прозвучит, а какая будет подавлена.

ЭТО ВОПРОС БЕЗОПАСНОСТИ, А НЕ УДОБСТВА
--------------------------------------
Незрячий ориентируется на слух: шум машин, эхо от стен, шаги других людей.
Приложение, которое говорит непрерывно, отбирает у него основной канал
восприятия. Система, предупреждающая обо всём, опаснее системы,
предупреждающей о главном.

Поэтому речь — ограниченный ресурс с бюджетом, а не побочный эффект
детектирования. Реплики конкурируют за него.

ПРАВИЛА АРБИТРАЖА
-----------------
1. Бюджет: не чаще одной реплики в min_interval_s.
2. Вытеснение: EMERGENCY прерывает текущую речь и игнорирует бюджет.
3. Дедупликация: реплика с тем же dedup_key подавляется в течение
   dedup_window_s. «Впереди человек» пять раз подряд бесполезно.
   Ключ включает track_id, поэтому два разных человека — два события,
   а один и тот же человек в течение окна — одно.
4. Срок годности: реплика, не произнесённая за ttl_s, отбрасывается,
   а не произносится с опозданием. Устаревшая подсказка вреднее молчания.
5. Понижение точности: при большой sigma вместо «в трёх метрах» — «близко».

ИЗМЕРЯЕМЫЙ РЕЗУЛЬТАТ
--------------------
Отношение «реплик в минуту» к «пропущенным опасностям» — метрика
когнитивной нагрузки. Кривая строится по разным min_interval_s.
Профиль ablation "no_scheduler" (min_interval_s = 0) даёт верхнюю
точку этой кривой и показывает цену отсутствия арбитража.

Счётчики подавленных реплик по причинам пишутся в stats() и попадают
в статью: важно показать не только сколько система сказала,
но и сколько промолчала и почему.
"""

from __future__ import annotations

from typing import Optional

from core.config import GuidanceConfig
from core.types import (
    FusionResult, GlobalGuidance, LocalGuidance, SystemHealth, Urgency, Utterance,
)

#: Окно дедупликации для сообщений о деградации, с.
#: Дольше обычного: «темно» каждые восемь секунд — это шум,
#: но и молчать об этом всю прогулку нельзя.
HEALTH_DEDUP_WINDOW_S = 30.0

#: Окно для подтверждающих реплик («свободно», «препятствий не вижу»), с.
#:
#: Такая реплика не сообщает о событии, а подтверждает, что система жива
#: и обстановка не изменилась. Повторять её каждые восемь секунд — значит
#: занимать слух ради нулевой информации; молчать всю прогулку тоже нельзя,
#: потому что для незрячего молчание неотличимо от отказа. Отсюда редкое,
#: но регулярное подтверждение.
CLEAR_DEDUP_WINDOW_S = 40.0


class UtteranceScheduler:
    def __init__(self, config: GuidanceConfig, phrasing):
        self.config = config
        self.phrasing = phrasing            # core.guidance.phrasing.Phrasebook
        self._last_spoken_ts: float = -1e9
        self._recent_keys: dict = {}        # dedup_key -> ts
        self._minute_window: list = []      # ts последних реплик
        self._suppressed = {"dedup": 0, "budget": 0, "rate": 0, "none": 0}
        self._spoken_total = 0
        self._preempted = 0

    # -----------------------------------------------------------------

    def consider(
        self,
        fusion: FusionResult,
        global_guidance: Optional[GlobalGuidance],
        local_guidance: LocalGuidance,
        health: SystemHealth,
        detections: list,
        ts: float,
        target=None,
    ) -> Optional[Utterance]:
        """Вернуть реплику или None, если говорить не время."""
        candidate = self._pick_candidate(
            fusion, global_guidance, local_guidance, health, detections, ts, target
        )
        if candidate is None:
            self._suppressed["none"] += 1
            return None

        # --- вытеснение --------------------------------------------------
        # EMERGENCY проходит мимо всех проверок и обрывает текущую речь:
        # дослушивание предыдущей фразы стоит полутора секунд,
        # за которые человек делает два шага.
        if candidate.urgency is Urgency.EMERGENCY and self.config.emergency_preempts:
            self._preempted += 1
            return self._accept(candidate, ts)

        # --- дедупликация -------------------------------------------------
        key = candidate.dedup_key
        if key is not None:
            last = self._recent_keys.get(key)
            if last is not None and ts - last < self._dedup_window(key):
                self._suppressed["dedup"] += 1
                return None

        # --- бюджет --------------------------------------------------------
        if ts - self._last_spoken_ts < self.config.min_interval_s:
            self._suppressed["budget"] += 1
            return None

        # --- предел частоты ---------------------------------------------------
        if self.utterances_per_minute(ts) >= self.config.max_utterances_per_min:
            self._suppressed["rate"] += 1
            return None

        return self._accept(candidate, ts)

    # -----------------------------------------------------------------

    def _pick_candidate(
        self,
        fusion: FusionResult,
        glob: Optional[GlobalGuidance],
        local: LocalGuidance,
        health: SystemHealth,
        detections: list,
        ts: float,
        target=None,
    ) -> Optional[Utterance]:
        """Выбрать, о чём говорить.

        Сообщение о деградации идёт ПЕРЕД навигационным, но уступает
        экстренному: человек должен знать, что система ослепла, раньше,
        чем получит от неё очередную подсказку, — но если прямо сейчас
        надо остановиться, сначала «стоп».
        """
        navigation = self.phrasing.compose(
            fusion, glob, local, health, detections, ts, target
        )
        if navigation is not None and navigation.urgency is Urgency.EMERGENCY:
            return navigation

        if health.degraded:
            degraded = self.phrasing.degraded_phrase(health, ts)
            if degraded is not None:
                key = degraded.dedup_key
                last = self._recent_keys.get(key)
                if last is None or ts - last >= HEALTH_DEDUP_WINDOW_S:
                    return degraded

        return navigation

    def _dedup_window(self, key: str) -> float:
        if key.startswith("health:"):
            return HEALTH_DEDUP_WINDOW_S
        if key.startswith("clear"):
            return CLEAR_DEDUP_WINDOW_S
        return self.config.dedup_window_s

    def _accept(self, utterance: Utterance, ts: float) -> Utterance:
        self._last_spoken_ts = ts
        self._spoken_total += 1
        self._minute_window.append(ts)
        if utterance.dedup_key:
            self._recent_keys[utterance.dedup_key] = ts
        # чистим устаревшие ключи, чтобы словарь не рос всю прогулку
        if len(self._recent_keys) > 512:
            cutoff = ts - max(self.config.dedup_window_s, HEALTH_DEDUP_WINDOW_S)
            self._recent_keys = {k: v for k, v in self._recent_keys.items() if v >= cutoff}
        return utterance

    # -----------------------------------------------------------------

    def utterances_per_minute(self, ts: float) -> float:
        """Текущая частота реплик — метрика когнитивной нагрузки."""
        cutoff = ts - 60.0
        self._minute_window = [t for t in self._minute_window if t >= cutoff]
        return float(len(self._minute_window))

    def stats(self) -> dict:
        """Сколько сказано и сколько подавлено, по причинам.

        Вторая половина не менее важна первой: система, которая
        промолчала тысячу раз из-за бюджета, ведёт себя иначе,
        чем система, которой нечего было сказать.
        """
        total_suppressed = sum(self._suppressed.values())
        return {
            "spoken": self._spoken_total,
            "preempted": self._preempted,
            "suppressed_total": total_suppressed,
            "suppressed_by_dedup": self._suppressed["dedup"],
            "suppressed_by_budget": self._suppressed["budget"],
            "suppressed_by_rate": self._suppressed["rate"],
            "nothing_to_say": self._suppressed["none"],
        }

    def reset(self) -> None:
        self._last_spoken_ts = -1e9
        self._recent_keys.clear()
        self._minute_window.clear()
        self._suppressed = {k: 0 for k in self._suppressed}
        self._spoken_total = 0
        self._preempted = 0

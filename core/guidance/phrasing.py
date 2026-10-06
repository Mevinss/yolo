# -*- coding: utf-8 -*-
"""
Генерация фраз на казахском, русском и английском.

ПОЧЕМУ НЕ ПРОСТО СЛОВАРЬ СТРОК
------------------------------
Фраза должна выражать ОБА уровня навигации одновременно:

    «Идите прямо сорок метров. Возьмите левее — впереди яма.
     Потом поворот направо.»

    маршрут -----------------  препятствие --------  маршрут

Это не конкатенация трёх независимых сообщений: их надо упорядочить
по срочности и уместить в бюджет времени. Человек идёт со скоростью
около 1.2 м/с, и фраза длиной шесть секунд описывает участок,
который он уже прошёл.

ПРАВИЛА (docs/mockups/voice_scenarios.md)
-----------------------------------------
1. Сначала действие, потом причина: «возьмите левее — впереди яма»,
   а не «впереди яма, возьмите левее». Незрячий должен начать
   действовать с первого слова, не дожидаясь конца фразы.
2. Экстренное сообщение — одно слово: «Стоп».
3. Расстояние округляется до величин, отмеряемых шагами: 5, 10, 20, 40.
4. Метры произносятся, только если оценке можно верить (sigma).
   Ложная точность подрывает доверие ровно тогда, когда оно нужнее.
5. Направление — в терминах тела (левее, правее), не в градусах
   и не в сторонах света.
6. При низком доверии утверждения о мире заменяются утверждениями
   о системе: не «свободно», а «препятствий не вижу».

КАЗАХСКИЙ ЯЗЫК
--------------
Не перевод русских фраз, а отдельный набор: порядок слов в казахском
другой (SOV), и калька звучит неестественно там, где важна каждая
доля секунды.

Формулировки ниже — рабочая версия, ТРЕБУЮЩАЯ ПРОВЕРКИ НОСИТЕЛЕМ.
Особенно «Кедергі көрмеймін» (препятствий не вижу): если при правке
она превратится в «Кедергі жоқ» (препятствий нет), смысл изменится
на противоположный и правило безопасности будет нарушено.
"""

from __future__ import annotations

from typing import Optional

from core.config import GuidanceConfig
from core.types import (
    Action, CrossingPhase, FusionResult, GlobalGuidance, Lang, LocalGuidance,
    ManeuverType, SystemHealth, Urgency, Utterance,
)

#: Доверие ниже этого включает смягчённые формулировки
SOFTEN_BELOW = 0.6

#: Дальность объявления, раздельно по тяжести.
#:
#: Разделение не косметическое. Скамейка в пяти метрах сбоку — не
#: опасность, а шум; яма в четырёх метрах по курсу — опасность.
#: Одинаковый порог для того и другого превращает стража в болтуна,
#: а система, предупреждающая обо всём, опаснее предупреждающей
#: о главном: она занимает слух, которым незрячий ориентируется.
#: Значения по умолчанию. Действующие берутся из GuidanceConfig,
#: потому что дома и на улице они разные.
HAZARD_RANGE_DANGER_M = 5.0
HAZARD_RANGE_OBSTACLE_M = 3.0

#: Радиус, в пределах которого наличие ЛЮБОГО объекта запрещает говорить
#: «свободно». Заведомо шире порогов объявления.
#:
#: Молчание и «свободно» — разные вещи. Система молчит, когда объект
#: ещё не требует действия; «свободно» утверждает, что впереди пусто.
#: Сказать это при машине в пяти метрах значит дать ложное успокоение —
#: хуже, чем не сказать ничего.
CLEAR_REQUIRES_EMPTY_M = 8.0

#: Передний сектор, в котором опасность считается лежащей на пути.
#: Ширина плеч на трёх метрах — около двадцати градусов; берём чуть
#: шире, чтобы учесть неточность курса. Всё за пределами — мимо,
#: и предупреждение о нём только отвлекает.
HAZARD_ANNOUNCE_CONE_DEG = 25.0

#: Вплотную: называем коротко и без чисел — время дороже подробностей, м
HAZARD_IMMEDIATE_M = 1.8

#: Команды движения. Ключ — Action, значение — по языкам.
ACTIONS = {
    Action.STOP:        {"ru": "Стоп",           "kk": "Тоқта",            "en": "Stop"},
    Action.SLOW:        {"ru": "Медленнее",      "kk": "Баяулаңыз",        "en": "Slow down"},
    Action.STRAIGHT:    {"ru": "Идите прямо",    "kk": "Тіке жүріңіз",     "en": "Go straight"},
    Action.BEAR_LEFT:   {"ru": "Возьмите левее", "kk": "Солға қарай жүріңіз", "en": "Bear left"},
    Action.BEAR_RIGHT:  {"ru": "Возьмите правее", "kk": "Оңға қарай жүріңіз", "en": "Bear right"},
    Action.TURN_LEFT:   {"ru": "Поворот налево", "kk": "Солға бұрылыңыз",  "en": "Turn left"},
    Action.TURN_RIGHT:  {"ru": "Поворот направо", "kk": "Оңға бұрылыңыз",  "en": "Turn right"},
    Action.WAIT_CROSSING: {"ru": "Стойте",       "kk": "Тұрыңыз",          "en": "Wait"},
    Action.ARRIVED:     {"ru": "Вы на месте",    "kk": "Жеттіңіз",         "en": "You have arrived"},
}

#: Заготовки предложений
TEMPLATES = {
    "maneuver_ahead": {
        "ru": "Через {d} метров {turn}",
        "kk": "{d} метрден кейін {turn}",
        "en": "In {d} meters {turn}",
    },
    "because": {
        "ru": "{action} — впереди {what}",
        "kk": "{action} — алдыда {what}",
        "en": "{action} — {what} ahead",
    },
    "crossing_approach_signalled": {
        "ru": "Через {d} метров переход. Регулируемый",
        "kk": "{d} метрден кейін өткел. Реттелетін",
        "en": "In {d} meters a crossing. Signalled",
    },
    "crossing_approach_unsignalled": {
        "ru": "Через {d} метров переход. Нерегулируемый — будьте осторожны",
        "kk": "{d} метрден кейін өткел. Реттелмейді — абай болыңыз",
        "en": "In {d} meters a crossing. Unsignalled — be careful",
    },
    "crossing_wait": {
        "ru": "Край тротуара. Стойте",
        "kk": "Жаяу жол шеті. Тұрыңыз",
        "en": "Kerb edge. Wait",
    },
    "crossing_decide": {
        "ru": "Решение за вами",
        "kk": "Шешім сізде",
        "en": "The decision is yours",
    },
    "crossing_hold": {
        "ru": "Держите прямо",
        "kk": "Тіке ұстаңыз",
        "en": "Hold straight",
    },
    "crossing_done": {
        "ru": "Переход пройден",
        "kk": "Өткелден өттіңіз",
        "en": "Crossing complete",
    },
    "landmark": {
        "ru": "{side} {what}",
        "kk": "{side} {what}",
        "en": "{what} on the {side}",
    },
    "clear": {
        "ru": "Свободно",
        "kk": "Жол бос",
        "en": "Clear ahead",
    },
    # ВАЖНО: утверждение о системе, а не о мире. См. докстроку модуля.
    "clear_uncertain": {
        "ru": "Препятствий не вижу",
        "kk": "Кедергі көрмеймін",
        "en": "I do not see obstacles",
    },
    "degraded": {
        "ru": "{detail}. Будьте внимательнее",
        "kk": "{detail}. Мұқият болыңыз",
        "en": "{detail}. Be more careful",
    },
    "replanned": {
        "ru": "Путь перекрыт. Перестраиваю маршрут",
        "kk": "Жол жабық. Бағытты қайта құрамын",
        "en": "Path blocked. Rerouting",
    },
    "near": {"ru": "близко", "kk": "жақын", "en": "close"},
    # --- поиск предмета ---
    "target_reached": {
        "ru": "{what} прямо перед вами",
        "kk": "{what} тура алдыңызда",
        "en": "{what} right in front of you",
    },
    "target_ahead": {
        "ru": "{what} прямо, {dist}",
        "kk": "{what} тіке, {dist}",
        "en": "{what} ahead, {dist}",
    },
    "target_side": {
        "ru": "{what} {side}, {dist}",
        "kk": "{what} {side}, {dist}",
        "en": "{what} on the {side}, {dist}",
    },
    "target_behind": {
        "ru": "{what} сзади, развернитесь",
        "kk": "{what} артта, бұрылыңыз",
        "en": "{what} behind you, turn around",
    },
    "target_remembered": {
        "ru": "{what} была {side}",
        "kk": "{what} {side} еді",
        "en": "{what} was on the {side}",
    },
    "target_not_found": {
        "ru": "{what} не вижу. Осмотритесь вокруг",
        "kk": "{what} көрмеймін. Айналаңызды қараңыз",
        "en": "I do not see {what}. Look around",
    },
    "turn_to": {
        "ru": "поверните {side}",
        "kk": "{side} бұрылыңыз",
        "en": "turn {side}",
    },
    "hazard": {
        "ru": "{side} {what}, {dist}",
        "kk": "{side} {what}, {dist}",
        "en": "{what} {side}, {dist}",
    },
    "hazard_now": {
        "ru": "{what} {side}",
        "kk": "{side} {what}",
        "en": "{what} {side}",
    },
    "stairs_up": {
        "ru": "Через {d} метров лестница вверх",
        "kk": "{d} метрден кейін жоғары баспалдақ",
        "en": "In {d} meters stairs up",
    },
    "stairs_down": {
        "ru": "Через {d} метров лестница вниз",
        "kk": "{d} метрден кейін төмен баспалдақ",
        "en": "In {d} meters stairs down",
    },
    "stairs_now_up": {
        "ru": "Лестница вверх", "kk": "Жоғары баспалдақ", "en": "Stairs up",
    },
    "stairs_now_down": {
        "ru": "Лестница вниз", "kk": "Төмен баспалдақ", "en": "Stairs down",
    },
    "handrail": {
        "ru": ", перила есть", "kk": ", тұтқасы бар", "en": ", handrail available",
    },
    "no_handrail": {
        "ru": ", перил нет", "kk": ", тұтқасы жоқ", "en": ", no handrail",
    },
}

SIDES = {
    "left":   {"ru": "Слева", "kk": "Солда", "en": "left"},
    "right":  {"ru": "Справа", "kk": "Оңда", "en": "right"},
    "center": {"ru": "Прямо", "kk": "Алдыда", "en": "ahead"},
}

#: Направление поворота — в терминах тела, не сторон света
TURN_SIDES = {
    "left":  {"ru": "налево", "kk": "солға", "en": "left"},
    "right": {"ru": "направо", "kk": "оңға", "en": "right"},
}

#: Причины деградации на трёх языках (русские — в core/io/health.py)
DEGRADED_KK = {
    "связь со спутниками слабая": "жерсерік байланысы әлсіз",
    "положение потеряно": "орналасқан жер белгісіз",
    "направление определяется неточно": "бағыт дәл емес",
    "темно": "қараңғы",
    "камера не отвечает": "камера жауап бермейді",
    "глубина недоступна": "тереңдік қолжетімсіз",
    "система реагирует с задержкой": "жүйе кешігіп жауап береді",
    "маршрут не построен": "бағыт құрылмаған",
    "не удаётся вернуться на маршрут": "бағытқа орала алмаймын",
}


class Phrasebook:
    def __init__(self, config: GuidanceConfig, profiles,
                 warn_distances_m: tuple = (40.0, 15.0, 5.0)):
        self.config = config
        self.profiles = profiles          # core.risk.scoring.ObjectProfiles
        self.lang: Lang = config.lang
        # Рубежи предупреждения о манёвре. Объявление привязано к ним,
        # а не к таймеру: иначе «через сорок метров направо» повторяется
        # каждые несколько секунд, пока эти сорок метров не пройдены.
        self.warn_distances_m = tuple(sorted(warn_distances_m, reverse=True))
        self._announced: set = set()      # (id манёвра, рубеж)

    # -----------------------------------------------------------------

    def compose(
        self,
        fusion: FusionResult,
        global_guidance: Optional[GlobalGuidance],
        local_guidance: LocalGuidance,
        health: SystemHealth,
        detections: list,
        ts: float,
        target=None,
    ) -> Optional[Utterance]:
        """Собрать реплику-кандидата.

        Решение «произносить ли» принимает scheduler; здесь только текст,
        срочность и ключ дедупликации.

        Порядок ветвей отражает приоритет: экстренное вытесняет всё,
        переход важнее подруливания, ориентир — в последнюю очередь.
        """
        L = self.lang.value
        soften = health.confidence < SOFTEN_BELOW

        # --- экстренное ---------------------------------------------------
        if fusion.action is Action.STOP:
            return self._mk(ACTIONS[Action.STOP][L], Urgency.EMERGENCY, ts,
                            key="stop", ttl=1.5, haptic="long")

        # --- перестроение --------------------------------------------------
        if fusion.replan_needed:
            return self._mk(TEMPLATES["replanned"][L], Urgency.NAVIGATION, ts,
                            key="replan", ttl=4.0)

        # --- переход --------------------------------------------------------
        crossing = self._crossing_phrase(fusion, global_guidance, ts, L)
        if crossing is not None:
            return crossing

        # --- прибытие --------------------------------------------------------
        if fusion.action is Action.ARRIVED:
            return self._mk(ACTIONS[Action.ARRIVED][L], Urgency.NAVIGATION, ts,
                            key="arrived", ttl=6.0)

        # --- манёвр маршрута --------------------------------------------------
        maneuver = self._maneuver_phrase(global_guidance, L, ts)
        if maneuver is not None:
            return maneuver

        # --- отклонение из-за препятствия -------------------------------------
        if fusion.action in (Action.BEAR_LEFT, Action.BEAR_RIGHT):
            action_text = ACTIONS[fusion.action][L]
            what = self._cause_name(fusion, detections, L)
            text = (TEMPLATES["because"][L].format(action=action_text, what=what)
                    if what else action_text)
            return self._mk(text, Urgency.CAUTION, ts,
                            key=f"{fusion.action.value}:{what}", ttl=2.5,
                            haptic="short-double")

        # --- опасность впереди, даже если обходить не требуется ----------------
        #
        # Главный сценарий работы — предупреждение, а не навигация.
        # Раньше об опасности сообщалось лишь как о ПРИЧИНЕ обхода,
        # поэтому объект сбоку от выбранного направления не назывался
        # вовсе: система молча обходила его, а человек не знал, что рядом.
        hazard = self._hazard_phrase(detections, L, ts, soften)
        if hazard is not None:
            return hazard

        # --- искомый предмет ---------------------------------------------------
        # Идёт ПОСЛЕ опасностей: довести до кружки важно, но не ценой
        # удара о стол по дороге. Безопасность имеет приоритет всегда.
        if target is not None:
            found = self._target_phrase(target, L, ts)
            if found is not None:
                return found

        # --- замедлиться -------------------------------------------------------
        #
        # Идёт ПОСЛЕ поиска намеренно. «Медленнее» не называет ни причины,
        # ни предмета: это подсказка о темпе, а не сведения об обстановке.
        # Стоя выше поиска, она вытесняла указание, где лежит искомое,
        # и человек слышал «медленнее» вместо «кружка справа, метр».
        if fusion.action is Action.SLOW:
            return self._mk(ACTIONS[Action.SLOW][L], Urgency.CAUTION, ts,
                            key="slow", ttl=2.5)

        # --- ориентир ----------------------------------------------------------
        landmark = self._landmark_phrase(detections, L, ts)
        if landmark is not None:
            return landmark

        # --- всё спокойно -------------------------------------------------------
        if fusion.action is Action.STRAIGHT and self._really_clear(detections):
            key = "clear_uncertain" if soften else "clear"
            return self._mk(TEMPLATES[key][L], Urgency.INFO, ts, key=key, ttl=4.0)

        return None

    # -----------------------------------------------------------------

    def degraded_phrase(self, health: SystemHealth, ts: float) -> Optional[Utterance]:
        """Сообщение о деградации. Порождается отдельно от навигации:
        оно описывает состояние системы, а не обстановку."""
        if not health.detail:
            return None
        L = self.lang.value
        detail = health.detail
        if self.lang is Lang.KK:
            detail = DEGRADED_KK.get(detail, detail)
        text = TEMPLATES["degraded"][L].format(detail=detail.capitalize()
                                               if self.lang is not Lang.KK else detail)
        return self._mk(text, Urgency.CAUTION, ts, key=f"health:{health.detail}", ttl=5.0)

    # -----------------------------------------------------------------

    def _crossing_phrase(self, fusion: FusionResult, glob: Optional[GlobalGuidance],
                         ts: float, L: str) -> Optional[Utterance]:
        phase = fusion.crossing_phase
        if phase is None:
            return None

        if phase is CrossingPhase.APPROACHING and glob and glob.next_maneuver:
            acc = glob.next_maneuver.accessibility or {}
            signalled = acc.get("crossing") in ("traffic_signals", "signals")
            key = "crossing_approach_signalled" if signalled else "crossing_approach_unsignalled"
            d = self.round_distance(glob.distance_to_maneuver_m)
            return self._mk(TEMPLATES[key][L].format(d=d), Urgency.NAVIGATION, ts,
                            key=f"crossing_approach:{d}", ttl=3.0)

        if phase is CrossingPhase.WAITING:
            acc = (glob.next_maneuver.accessibility or {}) if glob and glob.next_maneuver else {}
            signalled = acc.get("crossing") in ("traffic_signals", "signals")
            text = TEMPLATES["crossing_wait"][L]
            if not signalled:
                # Система не говорит «можно идти»: она не видит весь поток
                # машин надёжно и не вправе брать эту ответственность.
                text = f"{text}. {TEMPLATES['crossing_decide'][L]}"
            return self._mk(text, Urgency.CAUTION, ts, key="crossing_wait",
                            ttl=4.0, haptic="short-double")

        if phase is CrossingPhase.CROSSING:
            return self._mk(TEMPLATES["crossing_hold"][L], Urgency.CAUTION, ts,
                            key="crossing_hold", ttl=2.0)

        if phase is CrossingPhase.COMPLETED:
            return self._mk(TEMPLATES["crossing_done"][L], Urgency.NAVIGATION, ts,
                            key="crossing_done", ttl=3.0)
        return None

    def _maneuver_phrase(self, glob: Optional[GlobalGuidance], L: str,
                         ts: float) -> Optional[Utterance]:
        if glob is None or glob.next_maneuver is None:
            return None
        m = glob.next_maneuver
        threshold = self._pending_threshold(m, glob.distance_to_maneuver_m)
        if threshold is None:
            return None

        d = self.round_distance(glob.distance_to_maneuver_m)
        imminent = threshold <= min(self.warn_distances_m)

        # Лестница для незрячего опаснее поворота: пропущенная ступень
        # вниз — это падение. Поэтому она объявляется как манёвр,
        # а наличие перил называется явно.
        if m.type in (ManeuverType.STAIRS_UP, ManeuverType.STAIRS_DOWN):
            up = m.type is ManeuverType.STAIRS_UP
            if imminent:
                text = TEMPLATES["stairs_now_up" if up else "stairs_now_down"][L]
            else:
                text = TEMPLATES["stairs_up" if up else "stairs_down"][L].format(d=d)
            rail = (m.accessibility or {}).get("handrail")
            if rail in ("yes", "both", "left", "right"):
                text += TEMPLATES["handrail"][L]
            elif rail == "no":
                text += TEMPLATES["no_handrail"][L]
            return self._mk(text, Urgency.CAUTION, ts,
                            key=f"stairs:{m.type.value}:{'now' if imminent else d}",
                            ttl=3.0, haptic="short-double")

        action = {
            ManeuverType.TURN_LEFT: Action.TURN_LEFT,
            ManeuverType.SLIGHT_LEFT: Action.TURN_LEFT,
            ManeuverType.TURN_RIGHT: Action.TURN_RIGHT,
            ManeuverType.SLIGHT_RIGHT: Action.TURN_RIGHT,
        }.get(m.type)
        if action is None:
            return None
        turn = ACTIONS[action][L].lower()

        # Вплотную к манёвру команда звучит без расстояния: «поворот направо»
        if imminent:
            return self._mk(ACTIONS[action][L], Urgency.NAVIGATION, ts,
                            key=f"maneuver:{m.type.value}:now", ttl=2.5)

        text = TEMPLATES["maneuver_ahead"][L].format(d=d, turn=turn)
        return self._mk(text, Urgency.NAVIGATION, ts,
                        key=f"maneuver:{m.type.value}:{d}", ttl=3.5)

    def _pending_threshold(self, maneuver, distance_m: float):
        """Ближайший непройденный рубеж предупреждения или None.

        Каждый рубеж для каждого манёвра объявляется ровно один раз.
        Повторять «через сорок метров направо», пока эти сорок метров
        не пройдены, значит занимать речь без новой информации.
        """
        if distance_m is None:
            return None
        key_base = (round(maneuver.lat, 6), round(maneuver.lon, 6), maneuver.type.value)
        for th in self.warn_distances_m:
            if distance_m <= th and (key_base, th) not in self._announced:
                self._announced.add((key_base, th))
                # рубежи крупнее пройденного тоже считаем объявленными:
                # если подошли сразу к пяти метрам, «через сорок» уже незачем
                for bigger in self.warn_distances_m:
                    if bigger >= th:
                        self._announced.add((key_base, bigger))
                return th
        return None

    def reset(self) -> None:
        self._announced.clear()

    def _really_clear(self, detections: list) -> bool:
        """Действительно ли впереди пусто.

        Не «нечего сказать», а именно пусто: ни одного объекта, дающего
        риск, в переднем секторе на обозримой дальности. Разница
        принципиальна — молчание оставляет человеку его собственную
        осторожность, а «свободно» её снимает.
        """
        for d in detections:
            if not self.profiles.contributes_to_risk(d.label):
                continue
            if abs(d.bearing_deg) > self.config.hazard_cone_deg * 1.6:
                continue
            if d.distance_m is None or d.distance_m <= self.config.clear_requires_empty_m:
                return False
        return True

    def _hazard_phrase(self, detections: list, L: str, ts: float,
                       soften: bool) -> Optional[Utterance]:
        """Назвать ближайшую опасность впереди.

        Отбор: только то, что наполняет поле риска, в переднем секторе
        и в пределах дальности, где предупреждение ещё осмысленно.
        Далёкий объект называть незачем — человек дойдёт до него нескоро,
        а речь будет занята.
        """
        candidates = []
        for det in detections:
            if not self.profiles.contributes_to_risk(det.label):
                continue
            if det.distance_m is None:
                continue
            if abs(det.bearing_deg) > self.config.hazard_cone_deg:
                continue
            limit = (self.config.hazard_range_danger_m
                     if self.profiles.is_danger(det.label)
                     else self.config.hazard_range_obstacle_m)
            # Приближающийся объект заслуживает слова раньше стоящего:
            # он сам сокращает расстояние, пока человек идёт.
            if (det.radial_velocity_mps or 0.0) > 0.3:
                limit *= 1.5
            if det.distance_m <= limit:
                candidates.append(det)

        if not candidates:
            return None

        d = min(candidates, key=lambda x: x.distance_m)
        side = SIDES[self._side(d.bearing_deg)][L]
        what = self.profiles.name(d.label, L)
        danger = self.profiles.is_danger(d.label)

        # Вплотную — коротко и без чисел: время дороже подробностей.
        if d.distance_m <= self.config.hazard_immediate_m:
            text = TEMPLATES["hazard_now"][L].format(what=what, side=side.lower())
            urgency = Urgency.EMERGENCY if danger else Urgency.CAUTION
        else:
            text = TEMPLATES["hazard"][L].format(
                side=side, what=what,
                dist=self.distance_text(d.distance_m, d.distance_sigma_m, L),
            )
            urgency = Urgency.CAUTION

        # Ключ включает track_id: два разных человека подряд — два
        # события, один и тот же в течение окна — одно.
        key = f"hazard:{d.label}:{d.track_id}:{self._side(d.bearing_deg)}"
        return self._mk(text[0].upper() + text[1:], urgency, ts, key=key,
                        ttl=2.5, haptic="short-double" if danger else None)

    def _target_phrase(self, target, L: str, ts: float) -> Optional[Utterance]:
        """Куда идти к искомому предмету.

        Три состояния вместо двух: «нашли», «не нашли» и самое частое —
        «вижу, но вы стоите боком». Последнему нужно указание поворота,
        а не констатация, иначе человек слышит, что предмет рядом,
        и не понимает, что делать.
        """
        what = self.profiles.name(target.label, L)
        side_key = self._side(target.bearing_deg)
        side = SIDES[side_key][L]
        dist = self.distance_text(target.distance_m, None, L)

        if target.reached:
            text = TEMPLATES["target_reached"][L].format(what=what)
            key = f"target:{target.label}:reached"
            return self._mk(text[0].upper() + text[1:], Urgency.NAVIGATION, ts,
                            key=key, ttl=4.0, haptic="short-double")

        if abs(target.bearing_deg) > 120.0:
            text = TEMPLATES["target_behind"][L].format(what=what)
            key = f"target:{target.label}:behind"
            return self._mk(text[0].upper() + text[1:], Urgency.NAVIGATION, ts,
                            key=key, ttl=3.0)

        if not target.visible:
            # По памяти: предмет ушёл из кадра, но человек к нему идёт.
            # Молчать здесь значит бросить его на полпути.
            turn = TURN_SIDES.get(side_key)
            text = TEMPLATES["target_remembered"][L].format(what=what, side=side.lower())
            if turn is not None:
                text = f"{text}, {TEMPLATES['turn_to'][L].format(side=turn[L])}"
            key = f"target:{target.label}:memory:{side_key}"
            return self._mk(text[0].upper() + text[1:], Urgency.INFO, ts,
                            key=key, ttl=3.0)

        if target.centered:
            text = TEMPLATES["target_ahead"][L].format(what=what, dist=dist)
        else:
            text = TEMPLATES["target_side"][L].format(what=what, side=side.lower(), dist=dist)

        # Ключ включает сторону и грубую дальность: пока человек идёт,
        # обстановка меняется, и повтор той же фразы бесполезен,
        # а изменение стороны или дистанции — новая информация.
        bucket = "близко" if (target.distance_m or 9) < 1.5 else "далеко"
        key = f"target:{target.label}:{side_key}:{bucket}"
        return self._mk(text[0].upper() + text[1:], Urgency.NAVIGATION, ts,
                        key=key, ttl=2.5)

    def _landmark_phrase(self, detections: list, L: str, ts: float) -> Optional[Utterance]:
        """Ориентиры: незрячий подтверждает по ним своё положение.

        Приоритет INFO — вытесняются всем остальным. Молчание о них
        не опасно, но и пользы от них немало.
        """
        marks = [d for d in detections
                 if self.profiles.is_landmark(d.label) and d.distance_m
                 and d.distance_m < 8.0]
        if not marks:
            return None
        d = min(marks, key=lambda x: x.distance_m)
        side = SIDES[self._side(d.bearing_deg)][L]
        what = self.profiles.name(d.label, L)
        text = TEMPLATES["landmark"][L].format(side=side, what=what)
        return self._mk(text, Urgency.INFO, ts,
                        key=f"landmark:{d.label}:{d.track_id}", ttl=3.0)

    # -----------------------------------------------------------------

    def _cause_name(self, fusion: FusionResult, detections: list, L: str) -> Optional[str]:
        for i in fusion.cause_indices:
            if 0 <= i < len(detections):
                return self.profiles.name(detections[i].label, L)
        return None

    @staticmethod
    def _side(bearing_deg: float) -> str:
        if bearing_deg < -12:
            return "left"
        if bearing_deg > 12:
            return "right"
        return "center"

    def distance_text(self, distance_m: Optional[float],
                      sigma_m: Optional[float], L: str) -> str:
        """Метры или «близко» — в зависимости от достоверности оценки.

        Ложная точность подрывает доверие ровно тогда, когда оно нужнее
        всего: услышав «три метра» там, где пять, человек перестаёт
        полагаться на систему.
        """
        if distance_m is None:
            return TEMPLATES["near"][L]
        if sigma_m is None or sigma_m > self.config.metric_sigma_threshold_m:
            return TEMPLATES["near"][L]
        return f"{self.round_distance(distance_m)}"

    def round_distance(self, meters: float) -> int:
        """Округление до величин, которые человек способен отмерить шагами."""
        if meters is None:
            return 0
        for step in (5, 10, 20, 40, 60, 100):
            if meters <= step * 1.3:
                return step
        return int(round(meters / 50.0) * 50)

    def _mk(self, text: str, urgency: Urgency, ts: float, key: str,
            ttl: float = 3.0, haptic: Optional[str] = None) -> Utterance:
        return Utterance(text=text, lang=self.lang, urgency=urgency, ts=ts,
                         ttl_s=ttl, dedup_key=key, haptic=haptic)

# -*- coding: utf-8 -*-
"""
Веса доступности: OSM-теги -> множитель стоимости ребра.

ЭТО ОТДЕЛЬНЫЙ ВКЛАД РАБОТЫ
--------------------------
Обычный пешеходный маршрутизатор минимизирует длину. Для незрячего
кратчайший путь регулярно оказывается худшим: он ведёт через
нерегулируемый переход, лестницу без перил и участок без тротуара.

Здесь стоимость ребра:

    cost = length_m * base_multiplier(тип) * П modifier_i(теги)

Множители перемножаются, а не складываются: лестница без перил
в неосвещённом месте должна быть хуже, чем каждое по отдельности.

ПРИНЦИП «НЕИЗВЕСТНО ЗНАЧИТ ОПАСНО»
----------------------------------
Отсутствие тега НЕ означает благополучия. Переход без crossing=*
считается нерегулируемым; лестница без handrail=* — без перил.
Для зрячего маршрутизатора такая трактовка избыточно осторожна,
для незрячего — единственно допустимая: цена ошибки несимметрична.

Побочный эффект: в городе с плохим покрытием тегами система становится
чрезмерно осторожной. Это не дефект метода, а измеримое следствие
качества данных — его и меряет tools/audit_osm_coverage.py.

ОТКУДА ЧИСЛА
------------
Значения по умолчанию — экспертная оценка, откалиброванная по смыслу:
w_crossing_unmarked = 4.0 означает, что нерегулируемый переход в 10 м
«стоит» как 40 м тротуара, поэтому обход длиной до 30 м предпочтительнее.

Это исходная гипотеза, а не результат. Обоснование запланировано
на шаге 6: попарные сравнения с респондентами КОС либо, при отсутствии
доступа, ссылка на литературу по accessible routing с оговоркой
в разделе Limitations. Чувствительность результатов к этим числам
проверяется отдельным экспериментом (sensitivity analysis).
"""

from __future__ import annotations

from core.config import RoutingConfig

# ---------------------------------------------------------------------------
# Классификация путей
# ---------------------------------------------------------------------------

#: Пути, предназначенные для пешеходов
PEDESTRIAN_HIGHWAYS = {
    "footway", "path", "pedestrian", "steps", "corridor", "living_street",
}

#: Дороги, вдоль которых пешеход идёт, если нет отдельного тротуара
ROAD_HIGHWAYS = {
    "residential", "service", "unclassified", "tertiary", "tertiary_link",
    "secondary", "secondary_link", "primary", "primary_link",
    "trunk", "trunk_link", "road",
}

#: Пешеходу вход запрещён
FORBIDDEN_HIGHWAYS = {
    "motorway", "motorway_link", "construction", "proposed", "raceway", "busway",
}

#: Покрытия, плохие для трости и для устойчивости
POOR_SURFACES = {
    "gravel", "fine_gravel", "ground", "dirt", "earth", "grass", "sand",
    "mud", "pebblestone", "cobblestone", "sett", "unpaved", "woodchips",
}

#: Значения smoothness, означающие проблемы
POOR_SMOOTHNESS = {"bad", "very_bad", "horrible", "very_horrible", "impassable"}

#: Барьеры, физически мешающие проходу
BLOCKING_BARRIERS = {"wall", "fence", "hedge", "retaining_wall", "guard_rail"}

#: Максимальный множитель. Без потолка один тег делает ребро неудалимо
#: дорогим, и маршрут перестаёт находиться вовсе.
MAX_MULTIPLIER = 12.0


class AccessibilityWeights:
    def __init__(self, config: RoutingConfig):
        self.config = config

    # -----------------------------------------------------------------
    # Основной интерфейс
    # -----------------------------------------------------------------

    def edge_multiplier(self, tags: dict) -> float:
        """OSM-теги ребра -> множитель стоимости (>= 0.5)."""
        m = self._base_multiplier(tags)
        m *= self._surface_modifier(tags)
        m *= self._lighting_modifier(tags)
        m *= self._tactile_modifier(tags)
        m *= self._incline_modifier(tags)
        m *= self._wheelchair_modifier(tags)
        return max(0.5, min(m, MAX_MULTIPLIER))

    def is_impassable(self, tags: dict) -> bool:
        """Жёсткие запреты — ребро вообще не попадает в граф."""
        hw = tags.get("highway", "")
        if hw in FORBIDDEN_HIGHWAYS:
            return True
        if tags.get("foot") in ("no", "private"):
            return True
        if tags.get("access") in ("no", "private") and tags.get("foot") not in (
            "yes", "designated", "permissive", "public"
        ):
            return True
        if tags.get("barrier") in BLOCKING_BARRIERS:
            return True
        return False

    # -----------------------------------------------------------------
    # Составляющие
    # -----------------------------------------------------------------

    def _base_multiplier(self, tags: dict) -> float:
        """Множитель по типу пути."""
        c = self.config
        hw = tags.get("highway", "")
        footway = tags.get("footway", "")

        if hw == "steps":
            # Лестница без перил — одно из худших мест для незрячего
            if tags.get("handrail") in ("yes", "both", "left", "right"):
                return c.w_steps_with_handrail
            return c.w_steps_no_handrail

        if hw == "footway":
            if footway == "crossing" or tags.get("crossing"):
                return self._crossing_multiplier(tags)
            if footway == "sidewalk":
                return c.w_sidewalk
            return c.w_footway

        if hw in ("path", "corridor"):
            return c.w_footway * 1.3        # часто без покрытия и без разметки

        if hw == "pedestrian":
            return c.w_footway * 0.95       # пешеходная зона — машин нет

        if hw == "living_street":
            return c.w_footway * 1.6        # общее пространство с машинами

        if hw == "cycleway":
            # Велодорожка: пешеходу можно только если явно разрешено
            if tags.get("foot") in ("yes", "designated", "permissive"):
                return c.w_footway * 1.4
            return MAX_MULTIPLIER           # де-факто избегаем

        if hw in ROAD_HIGHWAYS:
            return self._road_multiplier(tags)

        return c.w_footway * 1.5            # незнакомый тип — осторожно

    def _crossing_multiplier(self, tags: dict) -> float:
        """Пешеходный переход — самое опасное место маршрута.

        Иерархия по убыванию безопасности:
          светофор со звуковым сигналом  <- ориентир доступен на слух
          светофор без звука
          размеченный нерегулируемый
          неразмеченный / без тега       <- считаем худшим случаем
        """
        c = self.config
        crossing = tags.get("crossing", "")
        crossing_ref = tags.get("crossing_ref", "")

        signalled = (
            crossing in ("traffic_signals", "signals", "pelican", "toucan")
            or crossing_ref in ("pelican", "toucan", "puffin")
            or tags.get("crossing:signals") == "yes"
        )

        if signalled:
            # Звуковой сигнал — решающий фактор: переход становится
            # доступен без зрения, а не просто безопаснее статистически.
            if tags.get("traffic_signals:sound") in ("yes", "walk", "always"):
                return c.w_crossing_audio_signal
            return c.w_crossing_signalled

        marked = (
            crossing in ("marked", "zebra", "uncontrolled")
            or tags.get("crossing:markings") not in (None, "no")
            or crossing_ref == "zebra"
        )
        if marked:
            # Размечен, но нерегулируемый: водитель обязан пропустить,
            # однако незрячий не может убедиться, что его пропускают.
            return c.w_crossing_unmarked * 0.6

        # Нет тега вообще -> трактуем как нерегулируемый (unknown != safe)
        return c.w_crossing_unmarked

    def _road_multiplier(self, tags: dict) -> float:
        """Проезжая часть: всё решает наличие тротуара."""
        c = self.config
        sidewalk = tags.get("sidewalk", "")
        sw_left = tags.get("sidewalk:left", "")
        sw_right = tags.get("sidewalk:right", "")
        sw_both = tags.get("sidewalk:both", "")

        has_sidewalk = (
            sidewalk in ("both", "left", "right", "yes")
            or sw_both == "yes"
            or sw_left in ("yes", "sidewalk")
            or sw_right in ("yes", "sidewalk")
        )
        # sidewalk=separate значит, что тротуар есть, но нарисован
        # отдельной линией — по самой дороге идти всё равно не надо
        separate = sidewalk == "separate" or sw_both == "separate"

        if separate:
            return c.w_no_sidewalk * 1.5
        if has_sidewalk:
            return c.w_sidewalk * 1.3
        if sidewalk == "no" or sw_both == "no":
            return c.w_no_sidewalk * 1.4

        # Тега нет. Для жилой улицы это терпимо, для магистрали — опасно.
        hw = tags.get("highway", "")
        if hw in ("residential", "service", "unclassified", "road"):
            return c.w_no_sidewalk * 0.8
        return c.w_no_sidewalk * 1.6

    def _surface_modifier(self, tags: dict) -> float:
        m = 1.0
        if tags.get("surface", "") in POOR_SURFACES:
            m *= 1.4
        if tags.get("smoothness", "") in POOR_SMOOTHNESS:
            m *= 1.5
        # Узкий проход: трость и встречный поток
        try:
            width = float(str(tags.get("width", "")).replace("m", "").strip())
            if width < 1.0:
                m *= 1.4
        except (TypeError, ValueError):
            pass
        return m

    def _lighting_modifier(self, tags: dict) -> float:
        """Освещение важно дважды: для остаточного зрения пользователя
        и для камеры, которая в темноте резко теряет полноту (в работе
        предшественника — падение покрытия до 41 % ниже 50 люкс)."""
        if tags.get("lit") == "no":
            return self.config.w_unlit
        return 1.0

    def _tactile_modifier(self, tags: dict) -> float:
        if tags.get("tactile_paving") == "yes":
            return self.config.w_tactile_paving
        return 1.0

    def _incline_modifier(self, tags: dict) -> float:
        raw = str(tags.get("incline", "")).strip().rstrip("%")
        if raw in ("up", "down"):
            return 1.2
        try:
            if abs(float(raw)) >= 8.0:
                return 1.5
            if abs(float(raw)) >= 4.0:
                return 1.2
        except (TypeError, ValueError):
            pass
        return 1.0

    def _wheelchair_modifier(self, tags: dict) -> float:
        """Косвенный признак: путь, непроходимый для коляски, обычно
        плох и для трости (ступени, бордюры, узость)."""
        wc = tags.get("wheelchair", "")
        if wc == "no":
            return 1.8
        if wc == "limited":
            return 1.3
        return 1.0

    # -----------------------------------------------------------------
    # Аудит покрытия
    # -----------------------------------------------------------------

    def coverage_report(self, tags: dict) -> dict:
        """Какие атрибуты доступности заданы у объекта.

        Возвращает {атрибут: True/False}. Проверяются только атрибуты,
        ОСМЫСЛЕННЫЕ для данного типа объекта: спрашивать про handrail
        у тротуара бессмысленно и исказило бы статистику покрытия.

        Используется tools/audit_osm_coverage.py.
        """
        hw = tags.get("highway", "")
        footway = tags.get("footway", "")
        is_crossing = footway == "crossing" or hw == "crossing" or "crossing" in tags

        rep: dict = {}

        if is_crossing:
            rep["crossing_type"] = "crossing" in tags or "crossing_ref" in tags
            rep["tactile_paving"] = "tactile_paving" in tags
            rep["kerb"] = "kerb" in tags
            rep["audio_signal"] = "traffic_signals:sound" in tags
            rep["markings"] = "crossing:markings" in tags or tags.get("crossing") in (
                "marked", "zebra", "uncontrolled", "traffic_signals"
            )
        elif hw == "steps":
            rep["handrail"] = "handrail" in tags
            rep["step_count"] = "step_count" in tags
            rep["tactile_paving"] = "tactile_paving" in tags
            rep["incline"] = "incline" in tags
            rep["lit"] = "lit" in tags
        elif hw in PEDESTRIAN_HIGHWAYS or hw == "footway":
            rep["surface"] = "surface" in tags
            rep["lit"] = "lit" in tags
            rep["width"] = "width" in tags
            rep["tactile_paving"] = "tactile_paving" in tags
            rep["smoothness"] = "smoothness" in tags
            rep["wheelchair"] = "wheelchair" in tags
        elif hw in ROAD_HIGHWAYS:
            rep["sidewalk"] = any(
                k in tags for k in ("sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both")
            )
            rep["lit"] = "lit" in tags
            rep["surface"] = "surface" in tags

        return rep

"""
Конфигурация системы.

ВСЕ настраиваемые числа живут здесь и только здесь. Причина не в удобстве:
в статье придётся отчитаться, при каких параметрах получены результаты,
и провести ablation (что будет, если убрать слагаемое). Разбросанные по коду
магические константы делают это невозможным.

Профиль сохраняется рядом с логом прогулки, поэтому любой результат
воспроизводим: config + запись = те же цифры.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from core.types import Lang


@dataclass
class PerceptionConfig:
    model_path: str = "data/models/yolov8m.pt"
    profiles_path: str = "data/object_profiles.csv"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cuda:0"          # ноутбук с NVIDIA; "cpu" для сравнения в статье
    max_detections: int = 50

    # --- трекинг между кадрами ---
    # Даёт скорость сближения: стоящий человек в трёх метрах и идущий
    # навстречу человек в трёх метрах требуют разной реакции.
    # Заодно связывает историю расстояний с объектом, а не с позицией в кадре.
    use_tracking: bool = True
    tracker: str = "bytetrack.yaml"   # bytetrack.yaml | botsort.yaml

    # --- оценка глубины ---
    use_depth: bool = True
    # depth_anything_v2_s|b|l — через transformers с HuggingFace;
    # MiDaS_small|DPT_Hybrid — через torch.hub, наследие предшественника,
    # оставлено для сравнения на той же модели, что была у него
    depth_model: str = "depth_anything_v2_s"
    # вес слияния: 1.0 = только bbox-оценка, 0.0 = только depth-модель
    depth_blend: float = 0.5
    # запускать depth не на каждом кадре (дорого) — раз в N кадров
    depth_every_n_frames: int = 3
    # Предельный возраст карты глубины для порождения опасностей без класса, с.
    # Устаревшая карта опаснее её отсутствия: она утверждает о геометрии
    # там, где человек уже сместился. При 1.2 м/с полсекунды — это 60 см.
    depth_max_age_s: float = 0.5


@dataclass
class RiskConfig:
    """Параметры построения поля риска."""

    n_sectors: int = 9              # 9 секторов по 60/9 ≈ 6.7° — примерно ширина плеча на 3 м
    n_bands: int = 4
    band_edges_m: tuple = (0.0, 1.5, 3.0, 6.0, 12.0)

    # риск объекта = class_priority * f(distance) * confidence
    # затухание риска с расстоянием: risk ~ exp(-distance / decay_m)
    distance_decay_m: float = 3.0
    # объекты дальше этого порога в поле риска не попадают
    max_range_m: float = 12.0
    # приоритеты классов задаются в data/object_profiles.csv,
    # значение по умолчанию для незнакомого класса
    default_class_priority: float = 0.3


@dataclass
class RoutingConfig:
    """Веса доступности для пешеходного графа.

    Это НЕ длина пути. cost = length_m * multiplier, где multiplier зависит
    от того, насколько участок пригоден для незрячего. Подбор этих чисел —
    отдельный результат статьи, поэтому они вынесены явно.
    """

    osm_file: str = "data/osm/astana_pedestrian.osm.pbf"

    # множители стоимости (1.0 = нейтрально, >1 = избегать, <1 = предпочитать)
    w_tactile_paving: float = 0.7        # тактильная плитка — предпочитаем
    w_sidewalk: float = 0.9
    w_footway: float = 1.0
    w_crossing_signalled: float = 1.2    # регулируемый переход
    w_crossing_unmarked: float = 4.0     # нерегулируемый — сильно избегаем
    w_crossing_audio_signal: float = 0.9 # со звуковым сигналом — почти как тротуар
    w_steps_no_handrail: float = 6.0
    w_steps_with_handrail: float = 2.5
    w_kerb_raised: float = 2.0           # бордюр без съезда
    w_no_sidewalk: float = 3.0           # идти по краю проезжей части
    w_unlit: float = 1.5                 # темнота: и для человека, и для камеры

    # порог схода с маршрута, м — дальше запускается перестроение
    off_route_threshold_m: float = 15.0
    # минимальный интервал между перестроениями, с (защита от дребезга)
    replan_cooldown_s: float = 10.0
    # за сколько метров предупреждать о манёвре
    maneuver_warn_distances_m: tuple = (40.0, 15.0, 5.0)


@dataclass
class LocalPlannerConfig:
    """Поиск свободного коридора."""

    # коридор уже этого угла считается непроходимым
    min_corridor_deg: float = 12.0
    # риск выше этого порога = сектор заблокирован
    block_threshold: float = 0.7
    # ниже этой дистанции по курсу — экстренная остановка
    emergency_clearance_m: float = 1.0
    # сглаживание по времени: доля нового значения (0..1). Малое = стабильнее.
    smoothing_alpha: float = 0.35


@dataclass
class FusionConfig:
    """Веса функции стоимости слияния — ЯДРО НОВИЗНЫ.

        cost(theta) = w_risk * R(theta)
                    + w_route * |theta - theta_route| / 180
                    + w_smooth * |theta - theta_prev| / 180

    Соотношение w_risk / w_route задаёт характер системы:
    высокий w_risk = осторожная, часто сходит с маршрута;
    высокий w_route = упрямая, ведёт по маршруту в ущерб безопасности.
    Кривая этого компромисса — центральный график статьи.
    """

    w_risk: float = 1.0
    w_route: float = 0.6
    w_smooth: float = 0.25

    # если отклонение держится дольше этого времени — перестраиваем маршрут
    sustained_deviation_s: float = 4.0
    # отклонение больше этого угла считается «обходом», а не «подруливанием»
    deviation_threshold_deg: float = 25.0


@dataclass
class GuidanceConfig:
    """Арбитраж речи. Ограничение потока реплик — вопрос безопасности:
    незрячий использует слух для ориентации, и заглушать его нельзя."""

    lang: Lang = Lang.RU
    # минимальный интервал между обычными репликами, с
    min_interval_s: float = 2.5
    # EMERGENCY игнорирует интервал и вытесняет текущую реплику
    emergency_preempts: bool = True
    # не повторять реплику с тем же dedup_key в течение этого времени, с
    dedup_window_s: float = 8.0
    # максимум реплик в минуту — измеряемая метрика когнитивной нагрузки
    max_utterances_per_min: int = 20
    # озвучивать метры только если sigma оценки ниже порога, иначе «близко/далеко»
    metric_sigma_threshold_m: float = 0.8

    # --- пороги предупреждения об опасности ---
    # Вынесены в конфиг, потому что дома и на улице они разные:
    # комната — три-четыре метра, и предупреждение за пять метров
    # относится к соседней комнате.
    hazard_range_danger_m: float = 5.0
    hazard_range_obstacle_m: float = 3.0
    hazard_cone_deg: float = 25.0
    hazard_immediate_m: float = 1.8
    clear_requires_empty_m: float = 8.0

    tts_engine: str = "piper"        # piper | pyttsx3 | system
    tts_voice_kk: str = "data/models/tts/kk_KZ.onnx"
    tts_voice_ru: str = "data/models/tts/ru_RU.onnx"
    speech_rate: float = 1.0


@dataclass
class IOConfig:
    camera_focal_px: float = 700.0
    camera_hfov_deg: float = 60.0
    camera_height_m: float = 1.35
    camera_pitch_deg: float = 10.0
    # калибровка сохраняется сюда tools/calibrate_focal.py
    calibration_file: str = "data/models/focal_calibration.json"
    # сглаживание компаса: телефон в руке трясётся сильно
    heading_smoothing_alpha: float = 0.2


@dataclass
class Config:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    local_planner: LocalPlannerConfig = field(default_factory=LocalPlannerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    io: IOConfig = field(default_factory=IOConfig)

    # имя профиля — попадает в лог и в подпись к таблице в статье
    profile_name: str = "default"

    #: Помещение. Меняет и категории объектов, и дистанции: дома опасно
    #: другое и на других расстояниях, а GPS не работает вовсе.
    indoor: bool = False

    def as_indoor(self) -> "Config":
        """Настройки для дома.

        Меняются согласованно, а не по одной: сокращённые дистанции
        без пересмотра категорий дали бы систему, которая молчит
        о ноже и предупреждает о машине за окном.
        """
        import copy
        cfg = copy.deepcopy(self)
        cfg.indoor = True
        cfg.profile_name = "indoor"

        # Комната — три-четыре метра. Предупреждать за пять значит
        # говорить о том, что стоит в другом конце помещения.
        cfg.guidance.hazard_range_danger_m = 2.5
        cfg.guidance.hazard_range_obstacle_m = 1.5
        cfg.guidance.hazard_immediate_m = 0.8
        cfg.guidance.clear_requires_empty_m = 3.0
        # Дома чаще поворачиваешься, чем идёшь прямо: сектор шире.
        cfg.guidance.hazard_cone_deg = 35.0

        # Поле риска сжимается вслед за помещением
        cfg.risk.band_edges_m = (0.0, 0.7, 1.5, 3.0, 6.0)
        cfg.risk.max_range_m = 6.0
        cfg.risk.distance_decay_m = 1.5

        # Ближняя граница остановки: между мебелью манёвра меньше
        cfg.local_planner.emergency_clearance_m = 0.5
        cfg.local_planner.min_corridor_deg = 18.0
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """TODO: загрузка из YAML. Пока используется профиль по умолчанию."""
        raise NotImplementedError("Реализуем, когда появится первый ablation-прогон")


# Профили для ablation-исследования в статье.
# Каждый отключает ровно одно слагаемое — это и есть строки таблицы ablation.
ABLATION_PROFILES = {
    "full": {},
    "no_fusion_route": {"fusion.w_route": 0.0},      # только избегание препятствий
    "no_fusion_risk": {"fusion.w_risk": 0.0},        # только следование маршруту
    "no_smoothing": {"fusion.w_smooth": 0.0},        # дребезг подсказок
    "no_depth": {"perception.use_depth": False},
    "no_scheduler": {"guidance.min_interval_s": 0.0},  # речь без арбитража
}

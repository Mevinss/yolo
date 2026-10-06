"""
Оркестратор: связывает слои в один цикл.

Это единственное место, где слои встречаются. Сам он не содержит логики —
только порядок вызовов и замер таймингов. Благодаря этому один и тот же
Pipeline обслуживает три сценария:

  1. live      — сервер получает кадры с iPhone         (server/main.py)
  2. replay    — прогон записанной прогулки для статьи  (eval/replay.py)
  3. ablation  — тот же прогон с отключённым слагаемым  (eval/replay.py)

Во всех трёх случаях код core/ идентичен. Именно это делает цифры в статье
воспроизводимыми: результат зависит от записи и конфига, а не от того,
на улице мы или за столом.
"""

from __future__ import annotations

import time
from typing import Optional

from core.config import Config
from core.types import (
    Frame, Pose, Route, SystemHealth, Tick, Utterance,
    GlobalGuidance, LocalGuidance, FusionResult, RiskField,
)


class Pipeline:
    """Один проход = один кадр.

    Слои передаются в конструктор, а не создаются внутри. Это позволяет
    подменить любой из них заглушкой в тестах и в baseline-конфигурациях
    (eval/baselines/ — «только маршрут» и «только зрение»).
    """

    def __init__(
        self,
        config: Config,
        detector,          # core.perception.detector.Detector
        risk_builder,      # core.risk.field.RiskFieldBuilder
        router,            # core.routing.planner.Router
        local_planner,     # core.local_planner.corridor.CorridorPlanner
        fusion,            # core.fusion.policy.FusionPolicy
        guidance,          # core.guidance.scheduler.UtteranceScheduler
        health_monitor,    # core.io.health.HealthMonitor
        depth_estimator=None,   # core.perception.depth.DepthEstimator | None
        target_tracker=None,    # core.search.target.TargetTracker | None
    ):
        self.config = config
        self.detector = detector
        self.risk_builder = risk_builder
        self.router = router
        self.local_planner = local_planner
        self.fusion = fusion
        self.guidance = guidance
        self.health_monitor = health_monitor
        self.depth_estimator = depth_estimator
        self.target_tracker = target_tracker

        self.route: Optional[Route] = None
        self._last_replan_ts = 0.0
        self._last_total_ms: Optional[float] = None
        self._replan_failed = False
        # Ставил ли пользователь цель. Без этого отсутствие маршрута —
        # штатный режим стража, а не неисправность: сообщать «маршрут
        # не построен» тому, кто и не просил его строить, значит приучать
        # не слушать предупреждения.
        self._destination_requested = False
        # Работает ли модель глубины. Берётся у самого модуля, если он передан.
        self.depth_ok: bool = bool(
            depth_estimator.available if depth_estimator is not None else False
        )

    # ------------------------------------------------------------------
    # Постановка задачи
    # ------------------------------------------------------------------

    def set_destination(self, pose: Pose, dest_lat: float, dest_lon: float) -> Route:
        """Построить маршрут от текущей позиции до цели."""
        self._destination_requested = True
        self.route = self.router.plan(
            start=(pose.lat, pose.lon),
            goal=(dest_lat, dest_lon),
        )
        self._replan_failed = False
        return self.route

    # ------------------------------------------------------------------
    # Глубина
    # ------------------------------------------------------------------

    def _refine_with_depth(self, depth_map, detections: list, frame: Frame) -> None:
        """Уточнить расстояния по карте глубины.

        MiDaS даёт относительную величину, поэтому масштаб берётся
        от детекций с надёжной геометрической оценкой — объектов
        известной высоты, стоящих на земле. Уточняются только те,
        чья собственная оценка хуже: подменять достоверное измерение
        менее достоверным смысла нет.
        """
        from core.perception.distance import fuse
        from core.types import DistanceSource

        reference = [d for d in detections
                     if d.distance_m is not None and d.distance_sigma_m is not None
                     and d.distance_sigma_m < 1.0]
        if len(reference) < 2:
            return

        for det in detections:
            est = self.depth_estimator.refine_detection_distance(
                depth_map, det, reference
            )
            if est is None:
                continue
            d_depth, sigma_depth = est
            if det.distance_m is None:
                det.distance_m, det.distance_sigma_m = d_depth, sigma_depth
                det.distance_source = DistanceSource.MONO_DEPTH
                continue
            merged = fuse([
                (det.distance_m, det.distance_sigma_m or 1.0, det.distance_source),
                (d_depth, sigma_depth, DistanceSource.MONO_DEPTH),
            ])
            if merged is not None:
                det.distance_m, det.distance_sigma_m, det.distance_source = merged

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------

    def step(self, frame: Frame, pose: Pose) -> Tick:
        """Обработать один кадр. Возвращает полный срез состояния.

        Порядок вызовов зафиксирован и важен:
        здоровье -> восприятие -> риск -> (локальный | глобальный)
        -> слияние -> речь.

        Самодиагностика идёт ПЕРВОЙ, потому что её результат влияет
        на трактовку всего остального: при потерянном курсе перевод
        углов камеры в азимуты недостоверен, и слияние должно об этом
        знать до того, как примет решение.
        """
        t0 = time.perf_counter()
        timings: dict = {}

        # --- 0. Самодиагностика ---------------------------------------
        t = time.perf_counter()
        self.depth_ok = bool(self.depth_estimator is not None and self.depth_estimator.available)
        health: SystemHealth = self.health_monitor.assess(
            pose=pose,
            ts=frame.ts,
            mean_intensity=float(frame.image.mean()),
            # Отключённая конфигом глубина — не деградация, а выбор режима.
            # Флаг поднимается только когда глубина ВКЛЮЧЕНА и не работает:
            # иначе система вечно повторяет «глубина недоступна» о том,
            # что и не должно было работать, и приучает не слушать её.
            depth_available=(not self.config.perception.use_depth) or self.depth_ok,
            route_available=(self.route is not None
                             or not self._destination_requested),
            off_route_no_replan=self._replan_failed,
            last_latency_ms=self._last_total_ms,
        )
        timings["health_ms"] = (time.perf_counter() - t) * 1000

        # --- 1. Восприятие -------------------------------------------------
        t = time.perf_counter()
        detections = self.detector.detect(frame)
        timings["perception_ms"] = (time.perf_counter() - t) * 1000

        # --- 1b. Глубина: уточнение расстояний и опасности без класса -------
        #
        # Детектор находит только то, чему обучен. Самое опасное для
        # незрячего часто не имеет класса вовсе: открытый люк, край
        # платформы, ступень вниз, вырытая траншея. Геометрия видит их,
        # не требуя знать имя.
        t = time.perf_counter()
        depth_map = None
        if self.depth_estimator is not None and self.config.perception.use_depth:
            depth_map, age_s = self.depth_estimator.estimate(frame.image, frame.seq, frame.ts)
            if depth_map is not None:
                # Устаревшая карта глубины опаснее её отсутствия: она
                # утверждает о геометрии там, где человек уже сместился.
                if age_s <= self.config.perception.depth_max_age_s:
                    self._refine_with_depth(depth_map, detections, frame)
                    detections.extend(self.depth_estimator.geometric_hazards(
                        depth_map, frame.intrinsics,
                        n_sectors=self.config.risk.n_sectors, ts=frame.ts,
                    ))
        timings["depth_ms"] = (time.perf_counter() - t) * 1000

        # --- 2. Поле риска -------------------------------------------------
        t = time.perf_counter()
        risk: RiskField = self.risk_builder.build(detections, frame.intrinsics, frame.ts)
        timings["risk_ms"] = (time.perf_counter() - t) * 1000

        # --- 3. Локальный уровень ------------------------------------------
        t = time.perf_counter()
        local: LocalGuidance = self.local_planner.plan(risk)
        timings["local_ms"] = (time.perf_counter() - t) * 1000

        # --- 4. Глобальный уровень -----------------------------------------
        t = time.perf_counter()
        glob: Optional[GlobalGuidance] = None
        if self.route is not None:
            glob = self.router.track(self.route, pose)
        timings["routing_ms"] = (time.perf_counter() - t) * 1000

        # --- 4a. Искомый предмет -------------------------------------------
        # Поиск НЕ влияет на выбор направления: искомая кружка не должна
        # притягивать человека к столу, о который он ударится. Он лишь
        # добавляет собственную реплику и уступает дорогу опасности.
        target = None
        if self.target_tracker is not None:
            target = self.target_tracker.update(detections, pose.heading_deg, frame.ts)

        # --- 5. Слияние ----------------------------------------------------
        t = time.perf_counter()
        fusion: FusionResult = self.fusion.decide(
            global_guidance=glob,
            local_guidance=local,
            pose=pose,
            health=health,
            ts=frame.ts,
        )
        timings["fusion_ms"] = (time.perf_counter() - t) * 1000

        # --- 5a. Перестроение маршрута, если обход стал устойчивым ----------
        if fusion.replan_needed and self.route is not None:
            if frame.ts - self._last_replan_ts > self.config.routing.replan_cooldown_s:
                goal = self.route.polyline[-1]
                try:
                    self.route = self.router.plan((pose.lat, pose.lon), goal)
                    self._replan_failed = False
                except ValueError:
                    # Перестроить не удалось. Молчать об этом нельзя:
                    # пользователь считает, что его ведут, а его уже не ведут.
                    self._replan_failed = True
                self._last_replan_ts = frame.ts

        # --- 6. Речь -------------------------------------------------------
        t = time.perf_counter()
        utterance: Optional[Utterance] = self.guidance.consider(
            fusion=fusion,
            global_guidance=glob,
            local_guidance=local,
            health=health,
            detections=detections,
            target=target,
            ts=frame.ts,
        )
        timings["guidance_ms"] = (time.perf_counter() - t) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        self._last_total_ms = timings["total_ms"]

        return Tick(
            # seq берётся у кадра, а не считается здесь: при потере кадров
            # собственный счётчик разошёлся бы с записью video.mp4,
            # по которой размечаются опасности
            seq=frame.seq,
            ts=frame.ts,
            pose=pose,
            health=health,
            detections=detections,
            risk=risk,
            global_guidance=glob,
            local_guidance=local,
            fusion=fusion,
            utterance=utterance,
            timings_ms=timings,
        )

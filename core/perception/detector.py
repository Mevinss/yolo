# -*- coding: utf-8 -*-
"""
Детектор объектов. Обёртка над YOLO, изолирующая остальную систему от него.

Наружу отдаётся list[Detection] из core.types — не результат ultralytics.
Благодаря этому смена детектора (YOLOv8 -> RT-DETR -> дообученная модель
на казахстанских данных) не затрагивает ни один другой слой.

ИСТОЧНИК
--------
Логика взята из проекта Ж. Каржаубай (yolo-main/models.py, blind_nav.py,
distance.py). Поведение сохранено, добавлено три вещи:

  1. bearing_deg — угол на объект. Без него поле риска не построить,
     а исходный проект обходился делением кадра на три полосы
     («слева / прямо / справа»), чего для планирования недостаточно.

  2. Трекинг между кадрами -> скорость сближения. Стоящий человек
     в трёх метрах и идущий навстречу человек в трёх метрах требуют
     разной реакции, хотя на одном кадре неразличимы.

  3. sigma оценки расстояния (см. distance.py).

УГОЛ НА ОБЪЕКТ
--------------
    bearing = atan((x_center - cx) / focal_px)

Отсчёт от оптической оси: минус слева, плюс справа. Пересчёт
в географический азимут делает ТОЛЬКО fusion, здесь его нет.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.config import PerceptionConfig
from core.perception.distance import (
    DistanceSmoother, estimate_from_bbox_height, estimate_from_ground_plane, fuse,
)
from core.types import CameraIntrinsics, Detection, DistanceSource, Frame


class Detector:
    def __init__(self, config: PerceptionConfig, profiles):
        self.config = config
        self.profiles = profiles          # core.risk.scoring.ObjectProfiles
        self._model = None
        self._smoother = DistanceSmoother()
        self._untracked_counter = 0

    # -----------------------------------------------------------------

    def load(self) -> "Detector":
        from ultralytics import YOLO
        self._model = YOLO(self.config.model_path)
        try:
            self._model.to(self.config.device)
        except Exception:
            # Отсутствие CUDA — не повод падать: система обязана
            # работать на CPU, это её заявленное свойство
            self._model.to("cpu")
        # Predictor/tracker initialization otherwise happens on the first user
        # frame, after the server has already announced readiness.
        self._run_model(np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8))
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()
        return self

    @property
    def device(self) -> str:
        try:
            return str(next(self._model.model.parameters()).device)
        except Exception:
            return "?"

    # -----------------------------------------------------------------

    def detect(self, frame: Frame) -> list:
        """Кадр -> list[Detection] с углами, расстояниями и скоростями."""
        if self._model is None:
            raise RuntimeError("Модель не загружена: вызовите load()")

        results = self._run_model(frame.image)
        detections: list = []

        for box in self._iter_boxes(results):
            label, conf, xyxy, track_id = box
            if conf < self.config.conf_threshold:
                continue

            x1, y1, x2, y2 = (int(v) for v in xyxy)
            bearing = self._bearing_deg(x1, x2, frame.intrinsics)

            key = track_id if track_id is not None else self._positional_key(x1, x2, label)
            distance_m, sigma_m, source = self._estimate_distance(
                label, x1, y1, x2, y2, frame.intrinsics
            )
            if distance_m is not None:
                distance_m = self._smoother.update(key, distance_m, frame.ts)

            detections.append(Detection(
                label=label,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                bearing_deg=bearing,
                distance_m=distance_m,
                distance_source=source,
                distance_sigma_m=sigma_m,
                track_id=track_id,
                radial_velocity_mps=self._smoother.velocity(key, frame.ts),
                ts=frame.ts,
            ))

            if len(detections) >= self.config.max_detections:
                break

        self._smoother.prune(frame.ts)
        return detections

    # -----------------------------------------------------------------

    def _run_model(self, image: np.ndarray):
        """Прогон модели. Трекинг включается через persist=True.

        Ultralytics хранит состояние трекера внутри модели, поэтому
        вызовы должны идти последовательно по кадрам одного потока.
        Для офлайн-прогона это выполняется естественно.
        """
        if self.config.use_tracking:
            return self._model.track(
                image, persist=True, verbose=False,
                conf=self.config.conf_threshold, iou=self.config.iou_threshold,
                imgsz=self.config.imgsz, tracker=self.config.tracker,
            )
        return self._model.predict(
            image, verbose=False,
            conf=self.config.conf_threshold, iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
        )

    def _iter_boxes(self, results):
        """-> (label, conf, xyxy, track_id)"""
        if not results:
            return
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return

        names = r.names
        ids = boxes.id.int().tolist() if getattr(boxes, "id", None) is not None else None

        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            yield (
                names.get(cls, str(cls)) if isinstance(names, dict) else names[cls],
                float(boxes.conf[i].item()),
                boxes.xyxy[i].tolist(),
                int(ids[i]) if ids is not None else None,
            )

    # -----------------------------------------------------------------

    @staticmethod
    def _bearing_deg(x1: int, x2: int, intr: CameraIntrinsics) -> float:
        """Угол на центр объекта относительно оптической оси, градусы."""
        cx = intr.width / 2.0
        center = (x1 + x2) / 2.0
        if intr.focal_px <= 0:
            # Запасной путь через поле зрения, если калибровки нет
            return (center - cx) / max(intr.width, 1) * intr.hfov_deg
        return math.degrees(math.atan((center - cx) / intr.focal_px))

    def _positional_key(self, x1: int, x2: int, label: str) -> str:
        """Ключ для сглаживания, когда трекинг отключён.

        Грубая привязка к позиции: без трекинга разделить два объекта
        одного класса нечем, и это ограничение надо признать явно,
        а не маскировать.
        """
        return f"{label}:{int((x1 + x2) / 2 / 40)}"

    def _estimate_distance(self, label: str, x1: int, y1: int, x2: int, y2: int,
                           intr: CameraIntrinsics) -> tuple:
        """-> (distance_m, sigma_m, DistanceSource)"""
        estimates = []

        bbox_est = estimate_from_bbox_height(
            bbox_height_px=float(y2 - y1),
            real_height_m=self.profiles.height_m(label),
            intrinsics=intr,
        )
        if bbox_est:
            estimates.append((bbox_est[0], bbox_est[1], DistanceSource.BBOX_HEIGHT))

        if self.profiles.ground_compatible(label):
            ground_est = estimate_from_ground_plane(float(y2), intr)
            if ground_est:
                estimates.append((ground_est[0], ground_est[1], DistanceSource.GROUND_PLANE))

        result = fuse(estimates)
        if result is None:
            return None, None, DistanceSource.UNKNOWN
        return result

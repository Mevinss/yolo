# -*- coding: utf-8 -*-
"""
Построение поля риска: список объектов -> полярная сетка.

ЗАЧЕМ ЭТОТ СЛОЙ ВООБЩЕ НУЖЕН
----------------------------
Без него планировщик пришлось бы учить работать со списком боксов YOLO.
Тогда любая замена детектора (YOLOv8 -> RT-DETR, добавление ультразвука,
добавление сегментации тротуара) ломала бы планировщик.

Поле риска — это абстракция «насколько опасно двигаться в сторону X
на дистанции Y». Её умеет наполнять любой источник: детектор, карта
глубины, семантическая сегментация покрытия, ультразвуковой датчик.
Планировщик знает только про неё.

МОДЕЛЬ РИСКА
------------
Вклад одного объекта в ячейку (сектор i, полоса j):

    r = class_priority(label)
        * confidence
        * exp(-distance / decay_m)
        * approach_factor(time_to_contact)
        * angular_overlap(bbox, sector_i)

Четыре множителя отвечают за четыре вопроса:
  насколько опасен этот класс, насколько мы уверены, что он там есть,
  насколько он близко и сближается ли он с нами.

УГЛОВОЕ РАСПРЕДЕЛЕНИЕ
---------------------
Объект вносит вклад во ВСЕ перекрываемые им сектора пропорционально
доле своего углового размера, а не только в сектор своего центра.
Человек не проходит сквозь стул, задев его краем: препятствие занимает
ширину, и планировщик обязан это видеть.

СБЛИЖЕНИЕ
---------
Стоящий человек в трёх метрах и идущий навстречу человек в трёх метрах
на одном кадре неразличимы, но требуют разной реакции. Трекинг даёт
скорость сближения, из неё выводится время до контакта, и оно повышает
риск тем сильнее, чем меньше остаётся времени.

Это то, чего не было у предшественника: там расстояние оценивалось
покадрово, без связи наблюдений между собой.

КЛИППИРОВАНИЕ
-------------
Риски складываются, затем ограничиваются единицей. Без ограничения
скопление мелких объектов даёт суммарный риск выше, чем одна яма,
и планировщик уводит человека от группы прохожих прямо в люк.
"""

from __future__ import annotations

import math

import numpy as np

from core.config import RiskConfig
from core.types import CameraIntrinsics, Detection, RiskField

#: Время до контакта, ниже которого риск удваивается, с.
#: Порядок величины человеческой реакции с последующим действием.
CRITICAL_TTC_S = 2.0

#: Максимальный множитель за сближение
MAX_APPROACH_FACTOR = 2.5


class RiskFieldBuilder:
    def __init__(self, config: RiskConfig, profiles):
        """profiles: core.risk.scoring.ObjectProfiles.

        Передаются профили целиком, а не словарь приоритетов: построителю
        нужен ещё и признак contributes_to_risk, иначе ориентиры
        (светофор, дверь) попадут в поле риска и перекроют коридор.
        """
        self.config = config
        self.profiles = profiles

    # -----------------------------------------------------------------

    def build(
        self,
        detections: list,
        intrinsics: CameraIntrinsics,
        ts: float,
    ) -> RiskField:
        """Собрать поле риска из детекций."""
        cfg = self.config
        band_edges = np.asarray(cfg.band_edges_m, dtype=float)
        n_bands = len(band_edges) - 1
        n_sectors = cfg.n_sectors

        half_fov = intrinsics.hfov_deg / 2.0
        sector_edges = np.linspace(-half_fov, half_fov, n_sectors + 1)

        cells = np.zeros((n_sectors, n_bands), dtype=float)
        contributors: dict = {}

        for idx, det in enumerate(detections):
            if not self._contributes(det.label):
                continue
            distance = det.distance_m
            if distance is None or distance > cfg.max_range_m:
                continue

            weight = self.profiles.risk_weight(
                det.label, det.confidence, distance, cfg.distance_decay_m
            )
            if weight <= 0.0:
                continue
            weight *= self._approach_factor(det)

            band = int(np.clip(np.searchsorted(band_edges, distance) - 1, 0, n_bands - 1))
            left_deg, right_deg = self._angular_extent(det.bbox, intrinsics)
            span = max(right_deg - left_deg, 1e-6)

            for i in range(n_sectors):
                overlap = min(right_deg, sector_edges[i + 1]) - max(left_deg, sector_edges[i])
                if overlap <= 0.0:
                    continue
                cells[i, band] += weight * (overlap / span)
                contributors.setdefault(f"{i},{band}", []).append(idx)

        np.clip(cells, 0.0, 1.0, out=cells)

        return RiskField(
            cells=cells,
            sector_edges_deg=sector_edges,
            band_edges_m=band_edges,
            ts=ts,
            contributors=contributors,
        )

    # -----------------------------------------------------------------

    def _angular_extent(self, bbox: tuple, intrinsics: CameraIntrinsics) -> tuple:
        """bbox -> (левый угол, правый угол) в градусах от оптической оси.

        Прямолинейная pinhole-модель: angle = atan((x - cx) / focal_px).
        """
        x1, _, x2, _ = bbox
        cx = intrinsics.width / 2.0
        f = intrinsics.focal_px

        if f <= 0:
            scale = intrinsics.hfov_deg / max(intrinsics.width, 1)
            return (x1 - cx) * scale, (x2 - cx) * scale

        return (
            math.degrees(math.atan((float(x1) - cx) / f)),
            math.degrees(math.atan((float(x2) - cx) / f)),
        )

    @staticmethod
    def _approach_factor(det: Detection) -> float:
        """Множитель за сближение. 1.0 для неподвижного или удаляющегося.

        Растёт обратно пропорционально времени до контакта: объект,
        до которого две секунды, опаснее того же объекта на том же
        расстоянии, но стоящего на месте.
        """
        ttc = det.time_to_contact_s
        if ttc is None or ttc <= 0:
            return 1.0
        return float(min(MAX_APPROACH_FACTOR, 1.0 + CRITICAL_TTC_S / max(ttc, 0.2)))

    def _class_priority(self, label: str) -> float:
        return self.profiles.priority(label)

    def _contributes(self, label: str) -> bool:
        """Ориентиры не наполняют поле риска — см. core/risk/scoring.py."""
        return self.profiles.contributes_to_risk(label)

    # -----------------------------------------------------------------

    def explain(self, field: RiskField, detections: list, sector: int, band: int) -> list:
        """Какие объекты породили риск в ячейке.

        Нужно для объяснимой подсказки: система обязана уметь сказать
        «возьмите левее — впереди яма», а не просто «возьмите левее».
        Непонятная команда исполняется хуже понятной.
        """
        idxs = field.contributors.get(f"{sector},{band}", [])
        return [detections[i] for i in idxs if 0 <= i < len(detections)]

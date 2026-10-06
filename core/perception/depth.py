# -*- coding: utf-8 -*-
"""
Монокулярная оценка глубины и ОПАСНОСТИ БЕЗ КЛАССА.

РОЛЬ В СИСТЕМЕ
--------------
Детектор находит только те объекты, которым его учили. Но самое опасное
для незрячего обычно не имеет класса в COCO: открытый люк, край платформы,
ступенька вниз, наледь, вырытая траншея, отсутствующая плитка.

Карта глубины видит их как геометрию, не требуя знать имя. Поэтому
глубина питает поле риска двумя способами:

  1. уточняет distance_m у детекций (слияние с bbox-оценкой);
  2. даёт риск БЕЗ КЛАССА — отклонение от плоскости земли перед человеком.

Второго в исходном проекте не было, и именно он оправдывает присутствие
depth-модели: уточнить расстояние до распознанного стула можно и дешевле,
а увидеть неразмеченную яму больше нечем.

ПОЧЕМУ DEPTH ANYTHING V2, А НЕ MiDaS
------------------------------------
Исходный проект грузил MiDaS через torch.hub. Это тянет ДВА сторонних
репозитория (intel-isl/MiDaS и его бэкбон rwightman/gen-efficientnet-pytorch),
которые скачиваются и ИСПОЛНЯЮТСЯ во время работы, требуют сети при
первом запуске и подтверждения доверия.

Для работы, обещающей воспроизводимость, это плохая опора: результат
перестанет повторяться, если репозитории изменятся или исчезнут.

Depth Anything V2 грузится через transformers с HuggingFace — оттуда же,
откуда казахский голос. Одна зависимость вместо двух репозиториев,
версия модели фиксируется, кода со стороны не исполняется.

MiDaS оставлен запасным вариантом: сравнение с работой предшественника
должно проводиться на той же модели, что была у него.

КАК ОБНАРУЖИВАЮТСЯ ОПАСНОСТИ БЕЗ КЛАССА
---------------------------------------
Приём опирается на свойство перспективы: для плоской земли ОБРАТНАЯ
глубина линейна по строке изображения.

    d(v) = h * f / ((v - cy) cos(theta) + f sin(theta))

    => 1/d(v) линейна по v

    а модель выдаёт  M(v) = a * (1/d) + b  при неизвестных a и b

Значит по пикселям земли значения модели должны ложиться на прямую
относительно номера строки. Отсюда:

  * прямая оценивается робастно (по медианам полос, устойчиво к выбросам);
  * остатки от прямой и есть отклонения от плоскости;
  * остаток вверх  = ближе плоскости = преграда над землёй;
  * остаток вниз   = дальше плоскости = провал: яма, люк, ступень вниз.

Неизвестные a и b исчезают при переходе к остаткам, поэтому метод
работает без привязки к метрам, которая сама по себе ненадёжна.

КОНВЕНЦИЯ ЗНАКА ПРОВЕРЯЕТСЯ, А НЕ ПРИНИМАЕТСЯ НА ВЕРУ
-----------------------------------------------------
Всё изложенное верно, только если модель выдаёт ОБРАТНУЮ глубину
(больше значение = ближе). Если конвенция окажется обратной,
«провал» и «преграда» поменяются местами, и система начнёт
предупреждать ровно наоборот — при этом ничего не упадёт
и цифры останутся правдоподобными.

Поэтому конвенция определяется эмпирически на первом кадре:
у камеры, смотрящей вперёд и слегка вниз, нижние строки — это
близкая земля, а строки ближе к горизонту — далёкая. Сравнение
их медиан однозначно даёт направление шкалы. Карта приводится
к обратной глубине сразу после инференса, и весь остальной код
работает с единственной конвенцией.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.config import PerceptionConfig
from core.types import CameraIntrinsics, Detection, DistanceSource

#: Доля кадра снизу, в которой ищем землю. Выше — небо, стены, горизонт.
GROUND_REGION_FRACTION = 0.45

#: Число горизонтальных полос для робастной оценки плоскости
N_PLANE_BANDS = 12

#: Порог остатка в единицах робастного разброса (MAD). Ниже — шум.
RESIDUAL_SIGMA_THRESHOLD = 3.0

#: Минимальная доля пикселей сектора с аномалией, чтобы счесть её реальной
MIN_ANOMALY_FRACTION = 0.12

#: Нижняя граница робастного разброса остатков, доля от размаха значений
#: поверхности.
#:
#: Без неё на гладкой поверхности разброс стремится к нулю, и деление
#: на него превращает любое микроотклонение в опасность: система
#: заваливает человека предупреждениями на ровном тротуаре. На реальных
#: картах от этого спасает шум модели, но полагаться на шум ради
#: корректности нельзя — свежий асфальт, наледь или снег дают почти
#: идеально гладкую глубину.
MIN_MAD_FRACTION_OF_RANGE = 0.02

#: Модели HuggingFace
HF_MODELS = {
    "depth_anything_v2_s": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_b": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_v2_l": "depth-anything/Depth-Anything-V2-Large-hf",
}

#: Модели, загружаемые через torch.hub (наследие проекта-предшественника)
HUB_MODELS = {"MiDaS_small", "MiDaS", "DPT_Hybrid", "DPT_Large"}


class DepthEstimator:
    def __init__(self, config: PerceptionConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._transform = None
        self._device = None
        self.device_name = "?"
        self.backend = "?"
        self.model_id = "?"

        self._last_map: Optional[np.ndarray] = None
        self._last_seq: int = -10_000
        self._last_ts: float = 0.0
        self.available: bool = False
        self.last_infer_ms: float = 0.0

        #: True — модель выдаёт обратную глубину (больше = ближе).
        #: Определяется на первом кадре, см. докстроку модуля.
        self.higher_is_closer: Optional[bool] = None

    # -----------------------------------------------------------------
    # Загрузка
    # -----------------------------------------------------------------

    def load(self, device: Optional[str] = None) -> "DepthEstimator":
        """Загрузка модели глубины. Отказ не фатален: система обязана
        работать и без неё, объявив об этом флагом DEPTH_UNAVAILABLE.

        device задаётся явно, а не через переменные окружения: torch
        читает CUDA_VISIBLE_DEVICES один раз при импорте, поэтому
        выставление её из кода уже ничего не меняет — модель молча
        уезжает на GPU, и замер «на CPU» оказывается замером на GPU.
        """
        name = self.config.depth_model
        try:
            import torch
            want = device or self.config.device
            if want and want.startswith("cpu"):
                self._device = torch.device("cpu")
            elif torch.cuda.is_available():
                self._device = torch.device(want if want and want != "auto" else "cuda")
            else:
                self._device = torch.device("cpu")
            self.device_name = str(self._device)

            if name in HUB_MODELS:
                self._load_hub(name)
            else:
                self._load_hf(name)
            self.available = True
        except Exception as ex:
            print(f"Модель глубины недоступна ({type(ex).__name__}: {ex}); "
                  "система продолжит без неё")
            self.available = False
        return self

    def _load_hf(self, name: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_id = HF_MODELS.get(name, name)
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self._model.to(self._device).eval()
        self.backend = "transformers"
        self.model_id = model_id

    def _load_hub(self, name: str) -> None:
        """Запасной путь для сравнения с проектом-предшественником."""
        import torch
        self._model = torch.hub.load("intel-isl/MiDaS", name, trust_repo=True)
        self._model.to(self._device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = (transforms.dpt_transform
                           if name in ("DPT_Large", "DPT_Hybrid")
                           else transforms.small_transform)
        self.backend = "torch.hub"
        self.model_id = f"intel-isl/MiDaS:{name}"

    # -----------------------------------------------------------------
    # Инференс
    # -----------------------------------------------------------------

    def estimate(self, image: np.ndarray, seq: int, ts: float = 0.0) -> tuple:
        """-> (карта ОБРАТНОЙ глубины, её возраст в секундах)

        Считается не каждый кадр: инференс стоит десятки-сотни
        миллисекунд, и гнать его на каждом кадре означает потерять
        реальное время ради величины, которая меняется медленно.

        Возраст возвращается наружу, потому что решение о доверии
        принимает вызывающий: уточнить расстояние по карте полусекундной
        давности допустимо, а утверждать по ней о наличии ямы — нет.
        """
        if not self.available:
            return None, float("inf")
        if seq - self._last_seq < self.config.depth_every_n_frames:
            return self._last_map, max(0.0, ts - self._last_ts)

        import time as _time
        t0 = _time.perf_counter()
        raw = (self._infer_hf(image) if self.backend == "transformers"
               else self._infer_hub(image))

        if self.higher_is_closer is None:
            self.higher_is_closer = self._detect_convention(raw)
            kind = ("обратная (больше = ближе)" if self.higher_is_closer
                    else "метрическая (больше = дальше)")
            print(f"  глубина: конвенция определена эмпирически — {kind}")

        if not self.higher_is_closer:
            # Приводим к единственной конвенции: обратная глубина.
            # Разветвление по типу модели ниже по коду стало бы
            # источником тихих ошибок знака.
            raw = 1.0 / (np.abs(raw) + 1e-6)

        self._last_map = raw
        self._last_seq = seq
        self._last_ts = ts
        self.last_infer_ms = (_time.perf_counter() - t0) * 1000
        return self._last_map, 0.0

    def _infer_hf(self, image: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        pred = torch.nn.functional.interpolate(
            out.predicted_depth.unsqueeze(1), size=rgb.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
        return pred.float().cpu().numpy()

    def _infer_hub(self, image: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            batch = self._transform(rgb).to(self._device)
            pred = self._model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        return pred.float().cpu().numpy()

    @staticmethod
    def _detect_convention(depth_map: np.ndarray) -> bool:
        """Определить направление шкалы по геометрии кадра.

        У камеры, смотрящей вперёд и слегка вниз, нижние строки —
        близкая земля, а строки ближе к середине кадра — далёкая.
        Если нижняя полоса даёт БОЛЬШИЕ значения, шкала обратная.

        Медианы, а не средние: в нижней части кадра часто оказываются
        ноги и посторонние предметы, и среднее к ним чувствительно.
        """
        h = depth_map.shape[0]
        near = float(np.median(depth_map[int(h * 0.88):, :]))
        far = float(np.median(depth_map[int(h * 0.52):int(h * 0.62), :]))
        return near > far

    # -----------------------------------------------------------------
    # Уточнение расстояния до детекций
    # -----------------------------------------------------------------

    def refine_detection_distance(
        self,
        depth_map: Optional[np.ndarray],
        detection: Detection,
        reference: list,
    ) -> Optional[tuple]:
        """Уточнить расстояние по глубине, опираясь на объекты-эталоны.

        Модель даёт относительную величину, поэтому масштаб берётся
        от детекций с надёжной геометрической оценкой (человек, машина —
        известной высоты и стоящие на земле). Если таких в кадре нет,
        уточнять нечем, и метод честно возвращает None вместо догадки.
        """
        if depth_map is None or not reference:
            return None

        pairs = []
        for ref in reference:
            if ref.distance_m is None or ref.distance_sigma_m is None:
                continue
            val = self._median_in_bbox(depth_map, ref.bbox)
            if val is not None and val > 1e-6:
                pairs.append((val, ref.distance_m))
        if len(pairs) < 2:
            return None

        # обратная глубина M ~ a / d  =>  d ~ a / M. Оцениваем a медианой.
        a = float(np.median([v * d for v, d in pairs]))
        val = self._median_in_bbox(depth_map, detection.bbox)
        if val is None or val <= 1e-6:
            return None

        d = a / val
        resid = [abs(a / v - d_ref) / max(d_ref, 0.1) for v, d_ref in pairs]
        sigma = d * max(float(np.median(resid)), 0.15)
        return d, sigma

    @staticmethod
    def _median_in_bbox(depth_map: np.ndarray, bbox: tuple) -> Optional[float]:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = depth_map.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        # центральная часть бокса: края захватывают фон
        cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
        cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
        roi = depth_map[cy1:cy2 + 1, cx1:cx2 + 1]
        return float(np.median(roi)) if roi.size else None

    # -----------------------------------------------------------------
    # Опасности без класса
    # -----------------------------------------------------------------

    def geometric_hazards(
        self,
        depth_map: Optional[np.ndarray],
        intrinsics: CameraIntrinsics,
        n_sectors: int,
        ts: float = 0.0,
    ) -> list:
        """Отклонения от плоскости земли -> список Detection без класса.

        Возвращает объекты с метками "drop" (провал: яма, люк, ступень
        вниз, край платформы) и "rise" (преграда над землёй).

        Оформлены как Detection, чтобы попасть в поле риска общим путём:
        риск не должен знать, откуда пришёл вклад — от детектора или
        от геометрии. Это и есть смысл абстракции RiskField.
        """
        if depth_map is None:
            return []

        h, w = depth_map.shape[:2]
        v0 = int(h * (1.0 - GROUND_REGION_FRACTION))
        ground = depth_map[v0:, :]
        if ground.size == 0:
            return []

        fit = self._fit_ground_line(ground)
        if fit is None:
            return []
        slope, intercept, mad = fit
        if mad <= 1e-9:
            return []

        rows = np.arange(ground.shape[0], dtype=float)[:, None]
        expected = slope * rows + intercept
        residual = (ground - expected) / mad

        # Карта приведена к обратной глубине, поэтому знаки однозначны:
        # значение НИЖЕ ожидаемого = дальше плоскости = провал.
        drops = residual < -RESIDUAL_SIGMA_THRESHOLD
        rises = residual > RESIDUAL_SIGMA_THRESHOLD

        hazards: list = []
        edges = np.linspace(0, w, n_sectors + 1).astype(int)
        for i in range(n_sectors):
            x1, x2 = edges[i], edges[i + 1]
            if x2 <= x1:
                continue
            for mask, label in ((drops, "drop"), (rises, "rise")):
                col = mask[:, x1:x2]
                frac = float(col.mean())
                if frac < MIN_ANOMALY_FRACTION:
                    continue
                rows_hit = np.where(col.any(axis=1))[0]
                if rows_hit.size == 0:
                    continue
                # ближайшая к пользователю строка аномалии = её нижний край
                v_img = v0 + int(rows_hit.max())
                dist = self._row_to_distance(v_img, intrinsics)
                hazards.append(Detection(
                    label=label,
                    confidence=float(min(1.0, frac / 0.5)),
                    bbox=(int(x1), int(v0 + rows_hit.min()), int(x2), int(v_img)),
                    bearing_deg=self._sector_bearing(x1, x2, intrinsics),
                    distance_m=dist,
                    distance_source=DistanceSource.GEOMETRIC,
                    # геометрия даёт положение уверенно, а дальность — грубо
                    distance_sigma_m=None if dist is None else max(0.3, dist * 0.25),
                    ts=ts,
                ))
        return hazards

    @staticmethod
    def _fit_ground_line(ground: np.ndarray) -> Optional[tuple]:
        """Робастная прямая «обратная глубина против строки».

        Оценка по медианам полос, а не методом наименьших квадратов:
        в нижней части кадра всегда есть посторонние объекты (ноги
        прохожих, столбы), и МНК притягивается к ним. Медиана полосы
        отражает преобладающую поверхность, то есть саму землю.

        Возвращает (наклон, свободный член, робастный разброс остатков).
        """
        n_rows = ground.shape[0]
        if n_rows < N_PLANE_BANDS * 2:
            return None

        band_edges = np.linspace(0, n_rows, N_PLANE_BANDS + 1).astype(int)
        xs, ys = [], []
        for k in range(N_PLANE_BANDS):
            a, b = band_edges[k], band_edges[k + 1]
            if b <= a:
                continue
            xs.append((a + b) / 2.0)
            ys.append(float(np.median(ground[a:b, :])))
        if len(xs) < 3:
            return None

        x = np.asarray(xs)
        y = np.asarray(ys)
        slope, intercept = np.polyfit(x, y, 1)

        rows = np.arange(n_rows, dtype=float)[:, None]
        resid = ground - (slope * rows + intercept)
        mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826

        # Пол разброса: см. MIN_MAD_FRACTION_OF_RANGE
        lo, hi = np.percentile(ground, [5, 95])
        floor = float(hi - lo) * MIN_MAD_FRACTION_OF_RANGE
        return float(slope), float(intercept), max(mad, floor)

    @staticmethod
    def _row_to_distance(v_px: int, intr: CameraIntrinsics) -> Optional[float]:
        """Строка изображения -> расстояние по плоскости земли."""
        import math
        cy = intr.height / 2.0
        theta = math.radians(intr.pitch_deg)
        denom = (float(v_px) - cy) * math.cos(theta) + intr.focal_px * math.sin(theta)
        if denom <= 1e-6:
            return None
        d = (intr.height_above_ground_m * intr.focal_px) / denom
        return d if 0 < d < 25.0 else None

    @staticmethod
    def _sector_bearing(x1: int, x2: int, intr: CameraIntrinsics) -> float:
        import math
        cx = intr.width / 2.0
        center = (x1 + x2) / 2.0
        if intr.focal_px <= 0:
            return (center - cx) / max(intr.width, 1) * intr.hfov_deg
        return math.degrees(math.atan((center - cx) / intr.focal_px))

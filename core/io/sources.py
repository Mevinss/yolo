"""
Источники кадров.

Три реализации одного интерфейса — это и есть причина, по которой
ядро можно прогонять офлайн:

  PhoneSource  — кадры по сети с iPhone            (полевой прогон)
  ReplaySource — кадры из записи data/walks/       (эксперименты статьи)
  WebcamSource — веб-камера ноутбука               (быстрая отладка)

ReplaySource обязана отдавать кадры С ИСХОДНЫМИ МЕТКАМИ ВРЕМЕНИ,
а не так быстро, как читается файл: иначе арбитраж речи, у которого
всё построено на интервалах, поведёт себя иначе, чем в поле,
и цифры в статье не будут соответствовать реальности.
"""

from __future__ import annotations

from core.types import Frame, Pose


class FrameSource:
    """Интерфейс источника."""

    def read(self) -> tuple:
        """-> (Frame, Pose) или (None, None), если поток закончился."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class PhoneSource(FrameSource):
    """TODO(шаг 4): кадры с iPhone через server/. Развитие yolo-main/sources.py."""


class ReplaySource(FrameSource):
    """TODO(шаг 6): чтение записи прогулки (видео + GPS + компас + разметка)."""


class WebcamSource(FrameSource):
    """TODO(шаг 4): cv2.VideoCapture для отладки за столом."""

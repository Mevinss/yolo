from types import SimpleNamespace

import pytest

from core.config import PerceptionConfig
from core.perception.detector import Detector


def test_load_does_not_report_success_when_first_inference_fails(monkeypatch):
    ultralytics = pytest.importorskip("ultralytics")
    class FailingModel:
        def to(self, device):
            return self
        def track(self, image, **kwargs):
            raise RuntimeError("inference unavailable")
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: FailingModel())
    with pytest.raises(RuntimeError, match="inference unavailable"):
        Detector(PerceptionConfig(), None).load()


def test_warmup_tracking_state_is_reset_before_user_frames(monkeypatch):
    ultralytics = pytest.importorskip("ultralytics")
    active_tracks = []
    class Tracker:
        def reset(self):
            active_tracks.clear()
    class Model:
        predictor = SimpleNamespace(trackers=[Tracker()])
        def to(self, device):
            return self
        def track(self, image, **kwargs):
            active_tracks.append("warmup artifact")
            return []
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: Model())
    Detector(PerceptionConfig(), None).load()
    assert active_tracks == []

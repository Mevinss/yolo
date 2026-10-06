import numpy as np
import pytest

from core.config import Config
from core.fusion.policy import FusionPolicy
from core.guidance.phrasing import Phrasebook
from core.guidance.scheduler import UtteranceScheduler
from core.io.health import HealthMonitor
from core.local_planner.corridor import CorridorPlanner
from core.pipeline import Pipeline
from core.risk.field import RiskFieldBuilder
from core.risk.scoring import ObjectProfiles
from core.types import CameraIntrinsics, Frame, HealthFlag, Pose


class EmptyDetector:
    def detect(self, frame):
        return []


def pipeline(use_depth):
    cfg = Config()
    cfg.perception.use_depth = use_depth
    profiles = ObjectProfiles.load()
    return Pipeline(cfg, EmptyDetector(), RiskFieldBuilder(cfg.risk, profiles),
                    None, CorridorPlanner(cfg.local_planner), FusionPolicy(cfg.fusion),
                    UtteranceScheduler(cfg.guidance, Phrasebook(cfg.guidance, profiles)),
                    HealthMonitor())


@pytest.mark.parametrize("enabled", [True, False])
def test_missing_depth_is_reported_only_when_requested(enabled):
    pipe = pipeline(enabled)
    frame = Frame(np.full((30, 40, 3), 128, np.uint8), 1.0, CameraIntrinsics(40, 40, 30))
    tick = pipe.step(frame, Pose(51, 71, 0, 1, accuracy_m=1, heading_sigma_deg=1))
    assert (HealthFlag.DEPTH_UNAVAILABLE in tick.health.flags) == enabled


def test_dark_frame_reaches_health_monitor():
    pipe = pipeline(False)
    frame = Frame(np.zeros((30, 40, 3), np.uint8), 1.0, CameraIntrinsics(40, 40, 30))
    tick = pipe.step(frame, Pose(51, 71, 0, 1, accuracy_m=1, heading_sigma_deg=1))
    assert HealthFlag.LOW_LIGHT in tick.health.flags

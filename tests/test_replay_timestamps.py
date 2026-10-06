import json

import numpy as np
import pytest

from eval.replay import WalkRecording


def recording(tmp_path, timestamps, frame_count):
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(tmp_path/"video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (32, 32))
    assert writer.isOpened()
    for i in range(frame_count):
        writer.write(np.full((32, 32, 3), i*40, np.uint8))
    writer.release()
    rows = [{"seq": i*3, "ts": ts, "lat": 51, "lon": 71} for i, ts in enumerate(timestamps)]
    (tmp_path/"track.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return WalkRecording(str(tmp_path))


def test_replay_uses_recorded_time_not_container_fps(tmp_path):
    rec = recording(tmp_path, [100.0, 100.7, 102.4], 3)
    frames = list(rec.frames())
    assert [r[0] for r in frames] == [0, 3, 6]
    assert [r[2] for r in frames] == [100.0, 100.7, 102.4]


@pytest.mark.parametrize("timestamps,count", [([100, 101], 3), ([100, 101, 102], 2)])
def test_replay_rejects_video_track_length_mismatch(tmp_path, timestamps, count):
    rec = recording(tmp_path, timestamps, count)
    with pytest.raises(ValueError, match="[Tt]rack|[Vv]ideo"):
        list(rec.frames())

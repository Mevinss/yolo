"""Exercise real inference and the HTTP/WebSocket API using a local image.

This checks integration only, not navigation accuracy or human safety.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT/"docs/article/smoke_latest.json")
    args = ap.parse_args()
    os.chdir(ROOT)
    from fastapi.testclient import TestClient
    from core.config import Config
    from server import main as server
    cfg = Config()
    cfg.perception.model_path = "data/models/yolov8n.pt"
    cfg.perception.device = "cpu"
    cfg.perception.imgsz = 320
    cfg.perception.use_depth = False
    server.session = server.Session("smoke_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    image = ROOT/"data/testframes/indoor.jpg"
    replies = []
    with TestClient(server.create_app(cfg)) as client:
        health = client.get("/health").json()
        assert health["ready"], health
        server.session.open_log(meta_extra={"data_kind": "integration_smoke", "is_field_data": False})
        assert client.post("/target", json={"query": "chair"}).status_code == 200
        with client.websocket_connect("/ws") as ws:
            for seq in range(3):
                ws.send_json({"seq": seq, "ts": 100+seq*.7,
                              "jpeg_b64": base64.b64encode(image.read_bytes()).decode(),
                              "lat": 51.1283, "lon": 71.4305,
                              "heading": 0, "accuracy": 5})
                reply = ws.receive_json()
                assert reply.get("seq") == seq and "error" not in reply, reply
                replies.append({"seq": seq, "state": reply["state"],
                                "timings_ms": reply["timings_ms"],
                                "utterance": reply.get("utterance"),
                                "audio_bytes_base64": len(reply.get("audio_b64") or "")})
        assert server.session.frames_processed == 3
        if health["audio_ready"]:
            assert any(r["audio_bytes_base64"] > 0 for r in replies), "No audio returned"
    report = {"kind": "integration_smoke_not_accuracy_evaluation", "health": health,
              "frames_processed": 3, "replies": replies,
              "recording": str(Path(server.session.walk_dir).resolve()),
              "limitations": ["No physical phone or user trial.", "Depth disabled for this CPU smoke.",
                              "Processing timings are not camera-to-ear latency."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "health": health, "report": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

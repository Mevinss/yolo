"""Start the prototype from any working directory with explicit runtime options."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="data/models/yolov8n.pt")
    ap.add_argument("--lang", default="ru", choices=["ru", "kk", "en"])
    ap.add_argument("--outdoor", action="store_true")
    ap.add_argument("--depth", action="store_true", help="requires optional depth dependencies and weights")
    args = ap.parse_args()
    os.chdir(ROOT)
    import uvicorn
    from core.config import Config
    from core.types import Lang
    from server.main import create_app
    cfg = Config()
    cfg.perception.device = args.device
    cfg.perception.model_path = args.model
    cfg.perception.use_depth = args.depth
    cfg.guidance.lang = Lang(args.lang)
    uvicorn.run(create_app(config=cfg, indoor=not args.outdoor), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

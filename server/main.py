# -*- coding: utf-8 -*-
"""
Сервер: телефон -> Pipeline -> телефон.

Работает на ноутбуке. Телефон раздаёт Wi-Fi (Personal Hotspot),
ноутбук лежит в рюкзаке. Схема проверена в исходном проекте:
в mobile_expo/App.js зашит адрес 172.20.10.3 — подсеть iPhone hotspot.

ОТЛИЧИЯ ОТ ИСХОДНОГО mobile_app/server.py
-----------------------------------------
  1. WebSocket вместо HTTP-поллинга: задержка ниже и стабильнее.
  2. Pipeline живёт в памяти процесса, а не запускается подпроцессом
     на каждый режим. Состояние маршрута, трекинга и арбитража речи
     сохраняется между кадрами — без этого двухуровневая навигация
     невозможна в принципе.
  3. Пишется JSONL-лог Tick'ов в том же формате, что читает
     eval/metrics.py. Полевой прогон и офлайн-прогон дают файлы
     одного вида, и метрики статьи считаются одним кодом.

ЗАМЕР СКВОЗНОЙ ЗАДЕРЖКИ
-----------------------
seq и ts проходят весь путь неизменными. Телефон помнит, когда
отправил кадр seq, и знает, когда получил ответ на него, поэтому
сквозная задержка меряется без синхронизации часов между устройствами.
Это устраняет главный источник ошибки в таких замерах.

ПОТЕРЯ КАДРОВ — ШТАТНОЕ ПОВЕДЕНИЕ
---------------------------------
Если обработка не успевает, новый кадр вытесняет необработанный,
а не встаёт в очередь. Очередь означала бы, что система реагирует
на обстановку секундной давности — для аида это хуже пропуска кадра.
Число вытесненных кадров пишется в лог: это честная характеристика,
а не то, что следует скрывать.

Запуск:
    python -m uvicorn server.main:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

PORT = 8765
LOG_DIR = os.path.join("data", "walks")


# ---------------------------------------------------------------------------
# Состояние прогона
# ---------------------------------------------------------------------------

class Session:
    """Один сеанс навигации: конвейер, лог, счётчики."""

    def __init__(self, walk_id: Optional[str] = None):
        self.walk_id = walk_id or datetime.now(timezone.utc).strftime("walk_%Y%m%d_%H%M%S")
        self.pipeline = None
        self.profiles = None
        self.target_tracker = None
        self.indoor = True
        self.pose_estimator = None
        self.tts = None
        self.writer = None
        self.video_writer = None
        self.video_size = None
        self.track_fh = None
        self.meta: dict = {}

        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.started_at = time.monotonic()
        self.last_tick = None
        self._busy = False

    # -----------------------------------------------------------------

    def build(self, config=None, indoor: bool = True):
        """Собрать конвейер. Модели грузятся один раз при старте.

        По умолчанию домашний режим: это основной сценарий работы —
        человек ходит по квартире и хочет знать, что перед ним.
        """
        from core.config import Config
        from core.guidance.phrasing import Phrasebook
        from core.guidance.scheduler import UtteranceScheduler
        from core.guidance.tts import TTSEngine
        from core.fusion.policy import FusionPolicy
        from core.io.health import HealthMonitor
        from core.io.pose import PoseEstimator
        from core.local_planner.corridor import CorridorPlanner
        from core.perception.detector import Detector
        from core.pipeline import Pipeline
        from core.risk.field import RiskFieldBuilder
        from core.risk.scoring import ObjectProfiles
        from core.routing.graph import PedestrianGraph
        from core.routing.planner import Router
        from core.routing.weights import AccessibilityWeights
        from core.search.target import TargetTracker

        cfg = config or Config()
        if indoor:
            cfg = cfg.as_indoor()
        self.config = cfg
        self.indoor = cfg.indoor

        profiles = ObjectProfiles.load(cfg.perception.profiles_path,
                                       indoor=cfg.indoor)
        detector = Detector(cfg.perception, profiles).load()
        self.profiles = profiles
        self.target_tracker = TargetTracker(profiles)

        # Граф нужен только маршрутам. Дома его может не быть вовсе,
        # и это не повод не запускаться: предупреждать об опасностях
        # можно и без единой карты.
        router = None
        try:
            weights = AccessibilityWeights(cfg.routing)
            graph = PedestrianGraph.load_cache(
                os.path.join("data", "osm", "graph_cache.pkl"), weights
            )
            router = Router(cfg.routing, graph, weights)
        except Exception as ex:
            print("  маршруты недоступны (" + type(ex).__name__ +
                  "); работаем без навигации")

        self.tts = TTSEngine(cfg.guidance).load(langs=[cfg.guidance.lang])
        self.pose_estimator = PoseEstimator(cfg.io)

        depth_estimator = None
        if cfg.perception.use_depth:
            from core.perception.depth import DepthEstimator
            depth_estimator = DepthEstimator(cfg.perception).load(device=detector.device)

        self.pipeline = Pipeline(
            config=cfg,
            detector=detector,
            risk_builder=RiskFieldBuilder(cfg.risk, profiles),
            router=router,
            local_planner=CorridorPlanner(cfg.local_planner),
            fusion=FusionPolicy(cfg.fusion),
            guidance=UtteranceScheduler(cfg.guidance, Phrasebook(cfg.guidance, profiles)),
            health_monitor=HealthMonitor(indoor=cfg.indoor),
            target_tracker=self.target_tracker,
            depth_estimator=depth_estimator,
        )
        return self

    # -----------------------------------------------------------------

    @property
    def walk_dir(self) -> str:
        return os.path.join(LOG_DIR, self.walk_id)

    def open_log(self, route=None, meta_extra: Optional[dict] = None) -> None:
        """Открыть запись прогулки: лог решений, трек и видео.

        Видео пишется ОБЯЗАТЕЛЬНО, а не по желанию: без него нельзя
        разметить опасности, а без разметки не считаются ни полнота
        предупреждений, ни их своевременность. Прогулка без видео —
        потерянный выход на улицу.
        """
        from core.serialization import TickWriter
        os.makedirs(self.walk_dir, exist_ok=True)

        self.meta = {
            "walk_id": self.walk_id,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": "iPhone -> laptop (edge-assisted)",
            **(meta_extra or {}),
        }
        with open(os.path.join(self.walk_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, ensure_ascii=False, indent=2)

        self.writer = TickWriter(
            os.path.join(self.walk_dir, "ticks.jsonl.gz"),
            config=self.config.to_dict(), route=route, meta=self.meta,
        )
        self.track_fh = open(os.path.join(self.walk_dir, "track.jsonl"),
                             "w", encoding="utf-8")

    def write_frame(self, image, msg: dict, ts: float) -> None:
        """Сохранить кадр в видео и строку трека.

        Контейнер имеет номинальные 20 fps. Фактические метки времени
        и seq сохраняются в track.jsonl, по одной строке на кадр.
        Replay использует эти строки, а не частоту контейнера.
        """
        import cv2
        if self.video_writer is None:
            h, w = image.shape[:2]
            self.video_writer = cv2.VideoWriter(
                os.path.join(self.walk_dir, "video.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h),
            )
            if not self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
                raise RuntimeError("Cannot open video recorder; refusing an incomplete research log")
            self.video_size = (w, h)
        if (image.shape[1], image.shape[0]) != self.video_size:
            image = cv2.resize(image, self.video_size)
        self.video_writer.write(image)

        if self.track_fh:
            self.track_fh.write(json.dumps({
                "seq": int(msg["seq"]), "ts": ts,
                "lat": msg.get("lat"), "lon": msg.get("lon"),
                "heading": msg.get("heading"), "accuracy": msg.get("accuracy"),
                "client_ts": msg.get("ts"),
            }, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None
        if self.track_fh:
            self.track_fh.close()
            self.track_fh = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.meta:
            self.meta["ended_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.meta["stats"] = self.stats()
            with open(os.path.join(self.walk_dir, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(self.meta, fh, ensure_ascii=False, indent=2)

    def stats(self) -> dict:
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        return {
            "walk_id": self.walk_id,
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "fps_in": round(self.frames_received / elapsed, 2),
            "fps_processed": round(self.frames_processed / elapsed, 2),
            "elapsed_s": round(elapsed, 1),
        }


session = Session()


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------

def create_app(config=None, indoor=True):
    """Собрать приложение.

    ВНИМАНИЕ К ИМПОРТАМ
    В модуле стоит from __future__ import annotations, поэтому ВСЕ
    аннотации становятся строками. FastAPI резолвит их по глобальным
    именам модуля, и если WebSocket импортирован внутри этой функции,
    имя не находится: параметр перестаёт считаться сокетом, обработчик
    не вызывается, а соединение отклоняется с 403 — без единой строки
    в трассировке. Поэтому импорты FastAPI живут на уровне модуля.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        """Собрать конвейер ДО приёма соединений.

        Модели грузятся около двадцати секунд. Делать это при первом
        кадре нельзя: телефон уже снимает, человек уже идёт, а система
        ещё загружается — и первые двадцать секунд прогулки теряются
        вместе с тем, что в них происходило.
        """
        print("Сборка конвейера...")
        t0 = time.perf_counter()
        try:
            session.build(config=config, indoor=indoor)
            print(f"  готово за {time.perf_counter() - t0:.1f} с")
            print(f"  детектор: {session.pipeline.detector.device}")
            print(f"  голоса: {[l.value for l in session.tts.loaded_langs]}")
        except Exception as ex:
            # Падать нельзя: пользователь увидит только то, что приложение
            # не подключается, без объяснения причины.
            print(f"  НЕ УДАЛОСЬ собрать конвейер: {type(ex).__name__}: {ex}")
        yield
        session.close()

    app = FastAPI(title="blind-nav", version="0.1", lifespan=lifespan)

    @app.get("/health")
    async def health():
        st = session.stats()
        st["ready"] = session.pipeline is not None
        st["recording"] = session.writer is not None
        st["audio_ready"] = bool(session.tts and session.tts.available)
        st["depth_ready"] = bool(session.pipeline and session.pipeline.depth_ok)
        st["routing_ready"] = bool(session.pipeline and session.pipeline.router)
        st["device"] = session.pipeline.detector.device if session.pipeline else None
        return JSONResponse(st)

    @app.post("/target")
    async def target(payload: dict):
        """Задать искомый предмет: кружка, унитаз, стул."""
        if session.target_tracker is None:
            return JSONResponse({"error": "конвейер не собран"}, status_code=503)
        q = str(payload.get("query", "")).strip()
        if not q:
            session.target_tracker.clear()
            return JSONResponse({"cleared": True})
        label = session.target_tracker.set_target(q)
        if label is None:
            return JSONResponse({"error": "не знаю такого предмета: " + q},
                                status_code=422)
        return JSONResponse({"label": label,
                             "name": session.profiles.name(label, "ru")})

    @app.post("/destination")
    async def destination(payload: dict):
        """Задать цель. Маршрут строится один раз и живёт в сессии."""
        if session.pipeline is None:
            return JSONResponse({"error": "конвейер не собран"}, status_code=503)
        if session.pipeline.router is None:
            return JSONResponse({"error": "маршруты недоступны"}, status_code=503)
        if session.last_tick is None:
            return JSONResponse({"error": "нет текущего положения"}, status_code=409)
        try:
            route = session.pipeline.set_destination(
                session.last_tick.pose,
                float(payload["lat"]), float(payload["lon"]),
            )
        except ValueError as ex:
            return JSONResponse({"error": str(ex)}, status_code=422)
        return JSONResponse({
            "distance_m": round(route.total_distance_m, 1),
            "maneuvers": len(route.maneuvers),
            "profile": route.profile,
        })

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                raw = await sock.receive_text()
                msg = json.loads(raw)

                # Служебные сообщения без кадра
                if msg.get("type") == "confirm_crossing":
                    # Решение перейти нерегулируемый переход принимает
                    # человек: система не видит весь поток машин надёжно
                    # и не вправе брать эту ответственность на себя.
                    if session.pipeline is not None:
                        session.pipeline.fusion.confirm_crossing()
                    await sock.send_text(json.dumps({"ack": "crossing"}))
                    continue

                if session.pipeline is None:
                    await sock.send_text(json.dumps(
                        {"error": "конвейер не собран — смотрите лог сервера"}))
                    continue

                session.frames_received += 1

                # Вытеснение вместо очереди: реагировать на обстановку
                # секундной давности хуже, чем пропустить кадр
                if session._busy:
                    session.frames_dropped += 1
                    continue

                session._busy = True
                try:
                    reply = await asyncio.get_event_loop().run_in_executor(
                        None, _process, msg
                    )
                finally:
                    session._busy = False

                await sock.send_text(json.dumps(reply, ensure_ascii=False))
        except WebSocketDisconnect:
            session.close()

    return app


def _process(msg: dict) -> dict:
    """Обработать один кадр. Выполняется в отдельном потоке."""
    import cv2
    from core.types import CameraIntrinsics, Frame

    jpeg = base64.b64decode(msg["jpeg_b64"])
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"seq": msg.get("seq"), "error": "кадр не декодируется"}

    h, w = image.shape[:2]
    cfg = session.config.io
    intr = CameraIntrinsics(
        focal_px=cfg.camera_focal_px, width=w, height=h,
        hfov_deg=cfg.camera_hfov_deg,
        height_above_ground_m=cfg.camera_height_m,
        pitch_deg=cfg.camera_pitch_deg,
    )

    ts = time.monotonic()
    gray = cv2.cvtColor(cv2.resize(image, (160, 120)), cv2.COLOR_BGR2GRAY)

    # В помещении спутников нет, и требовать координаты значит
    # не работать дома вообще. Решения о безопасности принимаются
    # по кадру и курсу; координата нужна только маршруту.
    lat, lon = msg.get("lat"), msg.get("lon")
    have_fix = lat is not None and lon is not None
    pose = session.pose_estimator.update(
        lat=float(lat) if have_fix else 0.0,
        lon=float(lon) if have_fix else 0.0,
        compass_deg=msg.get("heading"),
        accuracy_m=float(msg.get("accuracy") or 15.0) if have_fix else 999.0,
        ts=ts, gray=gray,
    )

    # Запись открывается на первом кадре, а не при старте сервера:
    # иначе каждый запуск создаёт папку прогулки, даже если никто
    # так и не подключился.
    if session.writer is None:
        session.open_log(route=session.pipeline.route)

    session.write_frame(image, msg, ts)

    frame = Frame(image=image, ts=ts, intrinsics=intr, seq=int(msg["seq"]))
    tick = session.pipeline.step(frame, pose)
    session.frames_processed += 1
    session.last_tick = tick

    if session.writer:
        session.writer.write(tick)

    audio_b64 = None
    if tick.utterance and session.tts and session.tts.available:
        wav = session.tts.synthesize(tick.utterance)
        if wav:
            audio_b64 = base64.b64encode(wav).decode("ascii")

    g = tick.global_guidance
    return {
        "seq": tick.seq,
        # ts возвращается тем же, что прислал телефон: так он измеряет
        # сквозную задержку по своим часам, без синхронизации с сервером
        "client_ts": msg.get("ts"),
        "utterance": None if not tick.utterance else {
            "text": tick.utterance.text,
            "lang": tick.utterance.lang.value,
            "urgency": int(tick.utterance.urgency),
        },
        "audio_b64": audio_b64,
        "interrupt": bool(session.tts and session.tts.consume_interrupt()),
        "haptic": tick.utterance.haptic if tick.utterance else None,
        "state": {
            "action": tick.fusion.action.value if tick.fusion else None,
            "crossing_phase": (tick.fusion.crossing_phase.value
                               if tick.fusion and tick.fusion.crossing_phase else None),
            "distance_to_maneuver_m": round(g.distance_to_maneuver_m, 1) if g else None,
            "off_route": bool(g.off_route) if g else None,
            "arrived": bool(g.arrived) if g else None,
            "blocked": bool(tick.local_guidance.blocked) if tick.local_guidance else None,
            "confidence": tick.health.confidence,
            "target": (session.target_tracker.query
                       if session.target_tracker and session.target_tracker.active
                       else None),
            "flags": [f.value for f in tick.health.flags],
            "detections": len(tick.detections),
        },
        "timings_ms": {k: round(v, 1) for k, v in tick.timings_ms.items()},
    }


app = create_app()

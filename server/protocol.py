"""
Протокол телефон <-> сервер.

Телефон -> сервер (WebSocket, бинарный кадр + JSON-метаданные):
    {"seq": int, "ts": float,
     "lat": float, "lon": float, "heading": float, "accuracy": float,
     "speed": float}
    + JPEG-кадр

Сервер -> телефон:
    {"seq": int,
     "utterance": {"text": str, "lang": str, "urgency": int} | null,
     "haptic": str | null,
     "state": {"action": str, "distance_to_maneuver": float,
               "off_route": bool, "blocked": bool},
     "audio": base64 | null}
    Поле audio — синтезированная речь; телефон только проигрывает.

ЗАМЕР ЗАДЕРЖКИ
--------------
seq и ts проходят весь путь неизменными, поэтому сквозную задержку
можно измерить на телефоне без синхронизации часов: телефон помнит,
когда отправил кадр seq, и знает, когда получил ответ на него.
Это даёт корректную цифру end-to-end latency для таблицы в статье.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhoneMessage:
    seq: int
    ts: float
    lat: float
    lon: float
    heading: float
    accuracy: float
    speed: float
    jpeg: bytes


@dataclass
class ServerMessage:
    seq: int
    utterance: dict | None
    haptic: str | None
    state: dict
    audio_b64: str | None

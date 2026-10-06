# -*- coding: utf-8 -*-
"""
Синтез речи: казахский, русский, английский. Офлайн.

АРХИТЕКТУРНОЕ РЕШЕНИЕ
---------------------
Синтез идёт НА НОУТБУКЕ, а не на телефоне, и это осознанный выбор:

  * в iOS казахского голоса нет вовсе;
  * pyttsx3 (исходный проект) не работает на мобильных платформах;
  * gTTS требует интернет, а аид обязан работать офлайн.

Ноутбук всё равно выполняет вычисления, поэтому там же синтезируется
речь и передаётся на телефон готовым звуком. Ограничение прототипа
превращается в решение задачи, которую в работе Ж. Каржаубай пришлось
оставить открытой (раздел 5.4: «offline Kazakh text-to-speech support
is limited»).

КАЗАХСКИЙ ГОЛОС
---------------
kk_KZ-issai-high из каталога Piper — модель ISSAI (Институт
интеллектуальных систем и искусственного интеллекта, Назарбаев
Университет). То есть казахская речь синтезируется моделью,
обученной в Казахстане, полностью офлайн, без обращения к облаку.

Доступны также kk_KZ-iseke-x_low и kk_KZ-raya-x_low (28 МБ против 128).
Они хуже по качеству, но быстрее — а задержка синтеза входит в сквозную
задержку системы и потому измеряется отдельной строкой в таблице.
Выбор между качеством и скоростью — предмет замера, а не вкуса.

ЗАДЕРЖКА КАК ЧАСТЬ БЕЗОПАСНОСТИ
-------------------------------
Экстренная реплика «Стоп» должна прозвучать раньше, чем человек
сделает следующий шаг, то есть примерно за 400 мс. Поэтому:

  * короткие экстренные фразы кэшируются заранее, а не синтезируются
    в момент опасности;
  * текущее воспроизведение прерывается (interrupt), а не дослушивается:
    дослушивание фразы стоит полутора секунд, за которые человек
    делает два шага.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil
import tempfile
import threading
import wave
from typing import Optional

from core.config import GuidanceConfig
from core.types import Lang, Urgency, Utterance

#: Каталог голосов
VOICES_DIR = os.path.join("data", "models", "tts")


def espeak_data_path(source: Path, cache_dir: Path) -> Path:
    """Windows eSpeak's native file API cannot open a non-ASCII data path."""
    source, cache_dir = Path(source), Path(cache_dir)
    if not (source/"phontab").is_file():
        raise FileNotFoundError(f"Missing eSpeak phontab in {source}")
    if str(source).isascii():
        return source
    target = cache_dir/"espeak-ng-data"
    if not str(target).isascii():
        raise ValueError("Set BLIND_NAV_TTS_CACHE to an ASCII-only directory")
    marker = target/".copy_complete"
    if not marker.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)
        marker.write_text("complete", encoding="ascii")
    return target

#: Голоса по умолчанию: язык -> имя модели Piper
#:
#: ДВА ГОЛОСА НА ЯЗЫК, ВЫБОР ПО СРОЧНОСТИ
#: Замер на Ryzen 5 7535HS (docs/article/tts_latency.json):
#:
#:     kk_KZ-issai-high    128 МБ   медиана 702 мс   p95 1063 мс
#:     kk_KZ-iseke-x_low    28 МБ   медиана 116 мс   p95  194 мс
#:     ru_RU-irina-medium   63 МБ   медиана 193 мс   p95  286 мс
#:
#: Качественный казахский голос в шесть раз медленнее лёгкого.
#: Для реплики «через сорок метров поворот направо» задержка в 700 мс
#: безразлична: она произносится с запасом в сорок метров. Для «стоп»
#: она недопустима — человек успевает сделать шаг.
#:
#: Поэтому срочные реплики синтезируются быстрым голосом, остальные —
#: качественным. Разборчивость важна там, где есть время слушать,
#: а скорость — там, где его нет.
FAST_VOICES = {
    Lang.KK: "kk_KZ-iseke-x_low",
    Lang.RU: "ru_RU-irina-medium",
    Lang.EN: "en_US-amy-low",
}

QUALITY_VOICES = {
    Lang.KK: "kk_KZ-issai-high",
    Lang.RU: "ru_RU-irina-medium",
    Lang.EN: "en_US-amy-medium",
}

#: Срочность, начиная с которой берётся быстрый голос
FAST_VOICE_FROM = Urgency.CAUTION

#: Оставлено для совместимости и для tools/fetch_voices.py
DEFAULT_VOICES = QUALITY_VOICES

#: Фразы, которые синтезируются заранее и держатся в памяти.
#: Все они экстренные: в момент опасности нет времени на синтез.
PRECACHE_PHRASES = {
    Lang.RU: ["Стоп", "Осторожно", "Стойте", "Машина"],
    Lang.KK: ["Тоқта", "Абайлаңыз", "Тұрыңыз", "Көлік"],
    Lang.EN: ["Stop", "Careful", "Wait", "Car"],
}


class TTSEngine:
    """Синтез через Piper с кэшем и прерыванием.

    Кэш двухуровневый: заранее подготовленные экстренные фразы
    и LRU для всего остального. Навигационные реплики повторяются
    («Идите прямо», «Поворот направо»), поэтому кэш попадает часто.
    """

    def __init__(self, config: GuidanceConfig, voices_dir: str = VOICES_DIR):
        self.config = config
        self.voices_dir = voices_dir
        # (Lang, "fast"|"quality") -> PiperVoice. Один и тот же файл
        # загружается один раз: для русского обе роли играет один голос.
        self._voices: dict = {}
        self._cache: dict = {}           # (lang, text) -> bytes
        self._precached: dict = {}
        self._lock = threading.Lock()
        self._interrupt = threading.Event()
        self.available: bool = False
        self.sample_rate: int = 22050

    # -----------------------------------------------------------------

    def load(self, langs: Optional[list] = None) -> "TTSEngine":
        """Загрузить голоса. Отсутствие голоса не фатально:
        система переключается на доступный язык и объявляет об этом."""
        try:
            from piper import PiperVoice
        except ImportError:
            print("piper-tts не установлен; синтез недоступен")
            return self

        loaded_files: dict = {}
        voice_options = {}
        if os.name == "nt":
            from importlib.metadata import version
            from piper.phonemize_espeak import ESPEAK_DATA_DIR
            cache = Path(os.environ.get("BLIND_NAV_TTS_CACHE") or
                         os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()),
                                      "blind-nav", "piper", version("piper-tts")))
            voice_options["espeak_data_dir"] = espeak_data_path(ESPEAK_DATA_DIR, cache)
        for lang in (langs or [self.config.lang]):
            for role, table in (("fast", FAST_VOICES), ("quality", QUALITY_VOICES)):
                name = table.get(lang)
                if not name:
                    continue
                path = os.path.join(self.voices_dir, f"{name}.onnx")
                if not os.path.exists(path):
                    continue
                try:
                    if name not in loaded_files:
                        loaded_files[name] = PiperVoice.load(path, **voice_options)
                    self._voices[(lang, role)] = loaded_files[name]
                    self.available = True
                except Exception as ex:
                    print(f"не удалось загрузить {name}: {ex}")

            # Если для роли голоса нет, подставляем имеющийся: отсутствие
            # быстрого голоса делает систему медленной, а не немой.
            for a, b in (("fast", "quality"), ("quality", "fast")):
                if (lang, a) not in self._voices and (lang, b) in self._voices:
                    self._voices[(lang, a)] = self._voices[(lang, b)]

            if not any(k[0] == lang for k in self._voices):
                print(f"нет голосов для {lang.value} — скачайте: python tools/fetch_voices.py")

        if self.available:
            self._precache()
        return self

    def _precache(self) -> None:
        """Синтезировать экстренные фразы заранее.

        Задержка синтеза «Стоп» в момент, когда надо остановиться,
        сводит на нет весь смысл предупреждения.
        """
        for lang in {k[0] for k in self._voices}:
            for text in PRECACHE_PHRASES.get(lang, []):
                try:
                    # экстренные фразы — всегда быстрым голосом
                    self._precached[(lang, text)] = self._synthesize_raw(text, lang, "fast")
                except Exception:
                    pass

    # -----------------------------------------------------------------

    def synthesize(self, utterance: Utterance) -> Optional[bytes]:
        """Реплика -> WAV. None, если синтез недоступен."""
        if not self.available:
            return None

        available = {k[0] for k in self._voices}
        lang = utterance.lang if utterance.lang in available else next(iter(available))
        role = "fast" if utterance.urgency >= FAST_VOICE_FROM else "quality"
        key = (lang, utterance.text)

        if key in self._precached:
            return self._precached[key]
        cache_k = (lang, role, utterance.text)
        if cache_k in self._cache:
            return self._cache[cache_k]

        try:
            data = self._synthesize_raw(utterance.text, lang, role)
        except Exception:
            return None

        with self._lock:
            # Ограниченный кэш: навигационные фразы повторяются,
            # но множество их не бесконечно
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[cache_k] = data
        return data

    def _synthesize_raw(self, text: str, lang: Lang, role: str = "quality") -> bytes:
        voice = self._voices[(lang, role)]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            voice.synthesize_wav(text, wav)
        self.sample_rate = getattr(voice.config, "sample_rate", self.sample_rate)
        return buf.getvalue()

    # -----------------------------------------------------------------

    def interrupt(self) -> None:
        """Оборвать текущее воспроизведение.

        Само воспроизведение идёт на телефоне, поэтому здесь только
        поднимается флаг: сервер отправляет команду остановки вместе
        со следующей репликой. Дослушивание предыдущей фразы стоит
        полутора секунд — двух шагов человека.
        """
        self._interrupt.set()

    def consume_interrupt(self) -> bool:
        was = self._interrupt.is_set()
        self._interrupt.clear()
        return was

    # -----------------------------------------------------------------

    def measure_latency(self, texts: list, lang: Lang, repeats: int = 3,
                        role: str = "quality") -> dict:
        """Замер задержки синтеза — строка таблицы задержек в статье.

        Кэш обходится намеренно: измеряется стоимость синтеза,
        а не скорость словаря.
        """
        import time
        if (lang, role) not in self._voices:
            return {}
        times = []
        for _ in range(repeats):
            for t in texts:
                t0 = time.perf_counter()
                data = self._synthesize_raw(t, lang, role)
                dt = (time.perf_counter() - t0) * 1000
                times.append((t, dt, len(data)))
        durations = [d for _, d, _ in times]
        durations.sort()
        return {
            "voice": (FAST_VOICES if role == "fast" else QUALITY_VOICES).get(lang),
            "role": role,
            "n": len(durations),
            "median_ms": round(durations[len(durations) // 2], 1),
            "p95_ms": round(durations[int(len(durations) * 0.95) - 1], 1),
            "min_ms": round(durations[0], 1),
            "max_ms": round(durations[-1], 1),
        }

    @property
    def loaded_langs(self) -> list:
        return sorted({k[0] for k in self._voices}, key=lambda l: l.value)

    def voice_for(self, urgency: Urgency, lang: Lang) -> Optional[str]:
        """Какой голос будет использован — для логов и отладки."""
        role = "fast" if urgency >= FAST_VOICE_FROM else "quality"
        return (FAST_VOICES if role == "fast" else QUALITY_VOICES).get(lang)


def cache_key(text: str, lang: Lang) -> str:
    return hashlib.sha1(f"{lang.value}:{text}".encode("utf-8")).hexdigest()[:16]

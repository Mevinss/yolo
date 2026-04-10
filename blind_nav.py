import cv2
import pyttsx3
import threading
import time
import argparse
import queue
import json
import math
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

DEFAULT_PROFILE = {
    'name': None,
    'height_m': 1.0,
    'width_m': 0.6,
    'priority': 0.35,
    'danger': False,
    'distance_scale': 1.0,
}

DEFAULT_DANGER_DATASET = {
    'default_priority': 0.35,
    'danger_priority_threshold': 0.6,
    'reference_height_px': 720,
    'classes': {
        'person': {'name': 'Человек', 'height_m': 1.68, 'width_m': 0.55, 'priority': 1.0, 'danger': True, 'distance_scale': 1.0},
        'car': {'name': 'Машина', 'height_m': 1.50, 'width_m': 1.80, 'priority': 1.0, 'danger': True, 'distance_scale': 1.0},
        'truck': {'name': 'Грузовик', 'height_m': 3.20, 'width_m': 2.50, 'priority': 1.0, 'danger': True, 'distance_scale': 1.0},
        'bus': {'name': 'Автобус', 'height_m': 3.00, 'width_m': 2.55, 'priority': 1.0, 'danger': True, 'distance_scale': 1.0},
        'motorcycle': {'name': 'Мотоцикл', 'height_m': 1.20, 'width_m': 0.8, 'priority': 0.95, 'danger': True, 'distance_scale': 1.0},
        'bicycle': {'name': 'Велосипед', 'height_m': 1.10, 'width_m': 0.55, 'priority': 0.9, 'danger': True, 'distance_scale': 1.0},
        'dog': {'name': 'Собака', 'height_m': 0.60, 'width_m': 0.35, 'priority': 0.85, 'danger': True, 'distance_scale': 1.0},
        'cat': {'name': 'Кошка', 'height_m': 0.35, 'width_m': 0.25, 'priority': 0.6, 'danger': False, 'distance_scale': 1.0},
        'traffic light': {'name': 'Светофор', 'height_m': 2.50, 'width_m': 0.35, 'priority': 0.7, 'danger': True, 'distance_scale': 1.0},
        'stop sign': {'name': 'Знак стоп', 'height_m': 2.00, 'width_m': 0.6, 'priority': 0.75, 'danger': True, 'distance_scale': 1.0},
        'bench': {'name': 'Скамейка', 'height_m': 0.90, 'width_m': 1.50, 'priority': 0.8, 'danger': True, 'distance_scale': 1.0},
        'chair': {'name': 'Стул', 'height_m': 0.90, 'width_m': 0.45, 'priority': 0.8, 'danger': True, 'distance_scale': 1.0},
        'dining table': {'name': 'Стол', 'height_m': 0.75, 'width_m': 1.20, 'priority': 0.8, 'danger': True, 'distance_scale': 1.0},
        'potted plant': {'name': 'Растение', 'height_m': 0.70, 'width_m': 0.45, 'priority': 0.6, 'danger': False, 'distance_scale': 1.0},
        'fire hydrant': {'name': 'Гидрант', 'height_m': 0.75, 'width_m': 0.35, 'priority': 0.8, 'danger': True, 'distance_scale': 1.0},
        'parking meter': {'name': 'Стойка', 'height_m': 1.40, 'width_m': 0.25, 'priority': 0.7, 'danger': True, 'distance_scale': 1.0},
        'skateboard': {'name': 'Скейтборд', 'height_m': 0.20, 'width_m': 0.2, 'priority': 0.55, 'danger': False, 'distance_scale': 1.0},
        'suitcase': {'name': 'Чемодан', 'height_m': 0.65, 'width_m': 0.45, 'priority': 0.7, 'danger': True, 'distance_scale': 1.0},
        'backpack': {'name': 'Рюкзак', 'height_m': 0.55, 'width_m': 0.35, 'priority': 0.65, 'danger': False, 'distance_scale': 1.0},
        'pothole': {'name': 'Яма', 'height_m': 0.20, 'width_m': 0.50, 'priority': 0.95, 'danger': True, 'distance_scale': 1.0},
        'obstacle': {'name': 'Препятствие', 'height_m': 0.55, 'width_m': 0.55, 'priority': 0.9, 'danger': True, 'distance_scale': 1.0},
        'stairs': {'name': 'Лестница', 'height_m': 1.80, 'width_m': 1.30, 'priority': 1.0, 'danger': True, 'distance_scale': 1.0}
    }
}

last_spoken = {}
COOLDOWN = 2.5
speech_queue = queue.Queue()
speech_error_logged = False
AUDIO_ENABLED = True


class DistanceEstimatorModel:
    """Гибридная модель дистанции: высота + ширина bbox + масштаб кадра."""

    def __init__(self, focal_px, reference_height_px):
        self.focal_px = max(float(focal_px), 1.0)
        self.reference_height_px = max(float(reference_height_px), 1.0)

    def _scaled_focal(self, frame_height_px):
        return self.focal_px * (max(float(frame_height_px), 1.0) / self.reference_height_px)

    def estimate(self, profile, bbox_w_px, bbox_h_px, frame_height_px):
        scaled_focal = self._scaled_focal(frame_height_px)
        est = []

        h_m = float(profile.get('height_m', 0.0) or 0.0)
        w_m = float(profile.get('width_m', 0.0) or 0.0)

        if bbox_h_px > 0 and h_m > 0:
            est.append(((h_m * scaled_focal) / bbox_h_px, 0.65))
        if bbox_w_px > 0 and w_m > 0:
            est.append(((w_m * scaled_focal) / bbox_w_px, 0.35))

        if not est:
            return None

        value = sum(v * w for v, w in est) / sum(w for _, w in est)
        scale = float(profile.get('distance_scale', 1.0) or 1.0)
        return max(0.2, value * scale)

    def estimate_focal(self, profile, bbox_w_px, bbox_h_px, distance_m, frame_height_px):
        if distance_m <= 0:
            return None

        h_m = float(profile.get('height_m', 0.0) or 0.0)
        w_m = float(profile.get('width_m', 0.0) or 0.0)
        focal_candidates = []

        if bbox_h_px > 0 and h_m > 0:
            focal_candidates.append(((bbox_h_px * distance_m) / h_m, 0.65))
        if bbox_w_px > 0 and w_m > 0:
            focal_candidates.append(((bbox_w_px * distance_m) / w_m, 0.35))

        if not focal_candidates:
            return None

        scaled_focal = sum(v * w for v, w in focal_candidates) / sum(w for _, w in focal_candidates)
        ref_focal = scaled_focal * (self.reference_height_px / max(float(frame_height_px), 1.0))
        return max(1.0, ref_focal)


def normalize_profile(label, data, default_priority):
    profile = dict(DEFAULT_PROFILE)
    profile['name'] = label
    profile['priority'] = float(default_priority)
    profile.update(data or {})

    profile['name'] = str(profile.get('name') or label)
    profile['height_m'] = max(float(profile.get('height_m', 1.0) or 1.0), 0.05)
    profile['width_m'] = max(float(profile.get('width_m', 0.6) or 0.6), 0.05)
    profile['priority'] = min(1.0, max(0.0, float(profile.get('priority', default_priority) or default_priority)))
    profile['danger'] = bool(profile.get('danger', False))
    profile['distance_scale'] = max(float(profile.get('distance_scale', 1.0) or 1.0), 0.1)
    return profile


def load_danger_dataset(dataset_path):
    merged = dict(DEFAULT_DANGER_DATASET)
    merged_classes = dict(DEFAULT_DANGER_DATASET['classes'])

    if dataset_path.exists():
        try:
            raw = json.loads(dataset_path.read_text(encoding='utf-8'))
            merged['default_priority'] = float(raw.get('default_priority', merged['default_priority']))
            merged['danger_priority_threshold'] = float(raw.get('danger_priority_threshold', merged['danger_priority_threshold']))
            merged['reference_height_px'] = float(raw.get('reference_height_px', merged['reference_height_px']))
            for key, value in raw.get('classes', {}).items():
                merged_classes[str(key).strip().lower()] = value
            print(f"Загружен датасет опасностей: {dataset_path}")
        except Exception as ex:
            print(f"Ошибка чтения датасета опасностей {dataset_path}: {ex}")
            print("Используется встроенный профиль опасностей")

    merged['classes'] = merged_classes
    return merged


def build_class_profiles(model_names, dataset_config):
    default_priority = float(dataset_config.get('default_priority', 0.35))
    class_map = dataset_config.get('classes', {})
    profiles = {}

    for _, label in model_names.items():
        label_key = str(label).strip().lower()
        profiles[label_key] = normalize_profile(label_key, class_map.get(label_key), default_priority)

    return profiles

def put_cyrillic_text(frame, text, position, font_scale=0.6, color=(255, 255, 255), thickness=2):
    """Выводит текст кириллицей на кадр используя PIL"""
    try:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", int(20 * font_scale))
        except:
            font = ImageFont.load_default()
        
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Ошибка вывода кириллицы: {e}")
        return frame

def speech_worker():
    global speech_error_logged
    try:
        speaker = pyttsx3.init()
        speaker.setProperty('rate', 150)
        print("✓ TTS инициализирован успешно")
    except Exception as e:
        print(f"✗ Ошибка инициализации TTS: {e}")
        speech_error_logged = True
        speaker = None
    
    while True:
        text = speech_queue.get()
        try:
            if text is None:
                break
            if speaker:
                speaker.say(text)
                speaker.runAndWait()
        except Exception as e:
            if not speech_error_logged:
                print(f"✗ Ошибка речи: {e}")
                speech_error_logged = True
        finally:
            speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def parse_source(value):
    value = value.strip()
    return int(value) if value.isdigit() else value

def format_distance(distance_m):
    if distance_m is None:
        return ''
    return f"~{distance_m:.1f}м"

def get_display_label(profile, fallback_label):
    return profile.get('name') or fallback_label


def get_risk_score(priority, distance_m, conf):
    d = max(distance_m if distance_m is not None else 4.5, 0.25)
    return (0.6 + float(priority)) * (1.0 / math.sqrt(d)) * (0.6 + float(conf))

def speak(text):
    if AUDIO_ENABLED and text:
        speech_queue.put(text)

def should_speak(key):
    now = time.time()
    if key not in last_spoken or now - last_spoken[key] > COOLDOWN:
        last_spoken[key] = now
        return True
    return False

def get_position(x1, x2, frame_width):
    center = (x1 + x2) / 2
    third = frame_width / 3
    if center < third:
        return 'слева'
    elif center < third * 2:
        return 'прямо'
    else:
        return 'справа'

def get_safe_direction(detections, frame_width):
    third = frame_width / 3
    zones = {'left': 0.0, 'center': 0.0, 'right': 0.0}

    for (x1, x2, distance_m, priority) in detections:
        center = (x1 + x2) / 2
        distance_weight = 1.0 / max(distance_m, 0.35) if distance_m is not None else 1.0
        weight = distance_weight * (0.5 + float(priority))
        if center < third:
            zones['left'] += weight
        elif center < third * 2:
            zones['center'] += weight
        else:
            zones['right'] += weight

    if zones['center'] < 0.9:
        return 'Путь свободен, идите прямо'
    elif zones['left'] + 0.2 < zones['right']:
        return 'Поверните налево'
    elif zones['right'] + 0.2 < zones['left']:
        return 'Поверните направо'
    else:
        return 'Осторожно, препятствия со всех сторон, остановитесь'

parser = argparse.ArgumentParser(description='Blind navigation assistant')
parser.add_argument(
    '--source',
    default='http://172.20.10.3:4747/video',
    help='Video source: camera index (0) or stream URL (http://IP:PORT/video)'
)
parser.add_argument('--model', default='yolov8m.pt', help='Path to YOLO model weights (default: yolov8m.pt for better accuracy)')
parser.add_argument('--conf', type=float, default=0.35, help='Detection confidence threshold (lower = more detections, default 0.35)')
parser.add_argument('--general-conf', type=float, default=0.22, help='Confidence threshold for non-danger classes (e.g. laptop, cell phone)')
parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
parser.add_argument('--alert-distance-m', type=float, default=2.0, help='Voice alert distance for dangerous objects in meters')
parser.add_argument('--danger-priority-threshold', type=float, default=None, help='Priority threshold to treat class as danger (overrides danger dataset value)')
parser.add_argument('--camera-buffer', type=int, default=1, help='OpenCV capture buffer size (lower values reduce stream delay)')
parser.add_argument('--max-voice-alerts', type=int, default=2, help='Max number of danger voice alerts per cycle')
parser.add_argument(
    '--focal-px',
    type=float,
    default=700.0,
    help='Approximate focal length in pixels for rough distance estimation'
)
parser.add_argument('--auto-calibrate', action='store_true', help='Enable automatic focal calibration at runtime')
parser.add_argument('--calib-label', default='person', help='Reference label for calibration (default: person)')
parser.add_argument('--calib-distance-m', type=float, default=2.0, help='Known distance to calibration object in meters')
parser.add_argument('--calib-alpha', type=float, default=0.1, help='Smoothing factor for focal calibration (0..1)')
parser.add_argument('--reference-height-px', type=int, default=None, help='Reference frame height for distance model scaling')
parser.add_argument('--audio', action='store_true', default=True, help='Enable voice alerts (enabled by default)')
parser.add_argument('--no-audio', dest='audio', action='store_false', help='Disable voice alerts')
parser.add_argument('--show-general', action='store_true', default=True, help='Show non-danger detections on screen')
parser.add_argument('--no-general', dest='show_general', action='store_false', help='Hide non-danger detections')
parser.add_argument('--danger-dataset', default='danger_dataset.json', help='JSON file with object priorities, dimensions and danger flags')
parser.add_argument(
    '--focal-file',
    default='focal_calibration.json',
    help='Path to focal calibration JSON file (loaded if exists and saved on exit)'
)
args = parser.parse_args()

AUDIO_ENABLED = bool(args.audio)

focal_file = Path(args.focal_file)
current_focal_px = float(args.focal_px)
danger_dataset_path = Path(args.danger_dataset)

danger_dataset = load_danger_dataset(danger_dataset_path)
danger_priority_threshold = float(danger_dataset.get('danger_priority_threshold', 0.6))
if args.danger_priority_threshold is not None:
    danger_priority_threshold = min(1.0, max(0.0, float(args.danger_priority_threshold)))

reference_height_px = float(danger_dataset.get('reference_height_px', 720.0))
if args.reference_height_px is not None:
    reference_height_px = max(1.0, float(args.reference_height_px))

if focal_file.exists():
    try:
        saved = json.loads(focal_file.read_text(encoding='utf-8'))
        loaded_focal = float(saved.get('focal_px', current_focal_px))
        if loaded_focal > 0:
            current_focal_px = loaded_focal
            print(f"Загружена калибровка focal_px={current_focal_px:.1f} из {focal_file}")
        saved_ref_h = float(saved.get('reference_height_px', reference_height_px))
        if saved_ref_h > 0:
            reference_height_px = saved_ref_h
    except Exception as ex:
        print(f"Не удалось загрузить калибровку: {ex}")

model = YOLO(args.model)
CLASS_PROFILES = build_class_profiles(model.names, danger_dataset)
distance_model = DistanceEstimatorModel(current_focal_px, reference_height_px)
print(f"Классов в danger-профиле: {len(CLASS_PROFILES)} | Порог опасности: {danger_priority_threshold:.2f}")
video_source = parse_source(args.source)
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Не удалось открыть источник видео: {video_source}")
    if AUDIO_ENABLED:
        speak("Не удалось открыть источник видео")
        speech_queue.join()
    raise SystemExit(1)

if args.camera_buffer > 0:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.camera_buffer)

print("Система запущена. Нажмите Q для выхода.")
if AUDIO_ENABLED:
    speak("Аудио активно")

last_direction_time = 0
DIRECTION_COOLDOWN = 6
last_main_alert_time = 0
MAIN_ALERT_COOLDOWN = 1.2
last_no_detection_voice_time = 0
NO_DETECTION_VOICE_COOLDOWN = 12
calib_updates = 0
calib_label = args.calib_label.strip().lower()

fps_time = time.time()
fps_counter = 0
current_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Расчет FPS
    fps_counter += 1
    elapsed = time.time() - fps_time
    if elapsed >= 1.0:
        current_fps = fps_counter / elapsed
        fps_counter = 0
        fps_time = time.time()

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

    danger_detections = []
    close_alerts = []
    person_targets = []
    best_calib = None

    for result in results:
        for box in result.boxes:
            label = str(model.names[int(box.cls)]).strip().lower()
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_h = max(y2 - y1, 1)
            box_w = max(x2 - x1, 1)

            profile = CLASS_PROFILES.get(label)
            if profile is None:
                profile = normalize_profile(label, None, danger_dataset.get('default_priority', 0.35))

            priority = float(profile.get('priority', 0.35))
            is_danger = bool(profile.get('danger', False)) or priority >= danger_priority_threshold
            distance_m = distance_model.estimate(profile, box_w, box_h, frame_height)
            position = get_position(x1, x2, frame_width)
            risk_score = get_risk_score(priority, distance_m, conf)

            if args.auto_calibrate and label == calib_label:
                measured_focal = distance_model.estimate_focal(profile, box_w, box_h, args.calib_distance_m, frame_height)
                if measured_focal is not None:
                    calib_score = conf * (0.6 * box_h + 0.4 * box_w)
                    if best_calib is None or calib_score > best_calib['score']:
                        best_calib = {'score': calib_score, 'measured_focal': measured_focal}

            if is_danger:
                danger_detections.append((x1, x2, distance_m, priority))

                # Три цвета BBox: красный (близко < 1м), оранжевый (1-4м), жёлтый (4+ м)
                if distance_m is not None:
                    if distance_m < 1.0 or risk_score > 1.8:
                        color = (0, 0, 255)  # Красный - опасно близко
                        status_text = "🔴 БЛИЗКО"
                    elif distance_m < 4.0 or risk_score > 1.2:
                        color = (0, 165, 255)  # Оранжевый - среднее расстояние
                        status_text = "🟠 СРЕДНЕЕ"
                    else:
                        color = (0, 255, 255)  # Жёлтый - далеко
                        status_text = "🟡 ДАЛЕКО"
                else:
                    color = (100, 100, 100)  # Серый - неизвестно
                    status_text = "❓"

                # Толстый прямоугольник для лучшей видимости
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                # Контур белый для контраста
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                # Информация: название + расстояние + уверенность
                dist_text = format_distance(distance_m)
                conf_text = f"{int(conf*100)}%"
                prio_text = f"P{priority:.2f}"
                info_line1 = f"{get_display_label(profile, label)} | {dist_text} | {prio_text}"
                info_line2 = f"{position.upper()} | {conf_text} | R{risk_score:.2f}"
                
                # Фон для текста
                cv2.rectangle(frame, (x1, y1-50), (x2, y1), color, -1)
                frame = put_cyrillic_text(frame, info_line1, (x1+5, y1-35), font_scale=0.55, color=(255,255,255), thickness=1)
                frame = put_cyrillic_text(frame, info_line2, (x1+5, y1-18), font_scale=0.5, color=(255,255,255), thickness=1)

                if label == 'person':
                    person_targets.append({
                        'position': position,
                        'distance_m': distance_m,
                    })

                if distance_m is not None and distance_m < args.alert_distance_m * 1.5:
                    close_alerts.append({
                        'label': label,
                        'name': get_display_label(profile, label),
                        'position': position,
                        'distance_m': distance_m,
                        'conf': conf,
                        'priority': priority,
                        'risk_score': risk_score,
                    })
            else:
                if args.show_general and conf >= args.general_conf:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
                    caption = f"{get_display_label(profile, label)} | P{priority:.2f} | {int(conf * 100)}%"
                    text_top = max(0, y1 - 22)
                    text_right = min(frame_width - 1, x1 + 260)
                    cv2.rectangle(frame, (x1, text_top), (text_right, y1), (40, 90, 40), -1)
                    frame = put_cyrillic_text(frame, caption, (x1 + 4, text_top + 4), font_scale=0.5, color=(255, 255, 255), thickness=1)

    if args.auto_calibrate and best_calib is not None:
        alpha = min(max(args.calib_alpha, 0.01), 0.5)
        current_focal_px = (1 - alpha) * distance_model.focal_px + alpha * best_calib['measured_focal']
        distance_model.focal_px = current_focal_px
        calib_updates += 1

    now = time.time()
    if close_alerts and now - last_main_alert_time > MAIN_ALERT_COOLDOWN:
        close_alerts.sort(key=lambda item: item['risk_score'], reverse=True)
        spoken_count = 0
        max_voice_alerts = max(1, args.max_voice_alerts)

        # Спец-подсказка: если перед человеком есть близкое препятствие.
        for person_info in person_targets:
            same_zone_obstacles = [
                item for item in close_alerts
                if item['label'] != 'person'
                and item['position'] == person_info['position']
                and item['distance_m'] is not None
                and person_info['distance_m'] is not None
                and item['distance_m'] <= person_info['distance_m'] + 0.6
            ]
            if same_zone_obstacles:
                top = max(same_zone_obstacles, key=lambda item: item['risk_score'])
                pair_key = f"person_block|{person_info['position']}|{top['label']}|{int(top['distance_m'] * 10)}"
                if should_speak(pair_key):
                    speak(
                        f"Внимание. Перед человеком {top['name']} {format_distance(top['distance_m'])} {top['position']}. "
                        f"Приоритет {top['priority']:.2f}"
                    )
                    spoken_count += 1
                    if spoken_count >= max_voice_alerts:
                        break

        for alert in close_alerts:
            if spoken_count >= max_voice_alerts:
                break
            dist_text = format_distance(alert['distance_m'])
            conf_text = f"{int(alert['conf']*100)}%"
            key = f"{alert['label']}|{alert['position']}|{int(alert['distance_m'] * 10)}"
            if should_speak(key):
                spoken = (
                    f"{alert['name']} {dist_text} {alert['position']}. "
                    f"Приоритет {alert['priority']:.2f}. "
                    f"Риск {alert['risk_score']:.2f}. "
                    f"Уверенность {conf_text}"
                ).strip()
                speak(spoken)
                spoken_count += 1
        if spoken_count > 0:
            last_main_alert_time = now

    if danger_detections and now - last_direction_time > DIRECTION_COOLDOWN:
        direction = get_safe_direction(danger_detections, frame_width)
        last_direction_time = now
        # Фон для text направления
        cv2.rectangle(frame, (0, 0), (600, 60), (0, 0, 0), -1)
        frame = put_cyrillic_text(frame, f"🧭 {direction}", (15, 15), font_scale=0.8, color=(0, 255, 255), thickness=2)

    left = frame_width // 3
    right = frame_width * 2 // 3
    cv2.line(frame, (left, 0), (left, frame.shape[0]), (255, 255, 255), 1)
    cv2.line(frame, (right, 0), (right, frame.shape[0]), (255, 255, 255), 1)
    
    # Статус калибровки и фокуса
    calib_state = '✓ ON' if args.auto_calibrate else '✗ OFF'
    metrics_line = (
        f"📹 {frame_width}x{frame_height} | FPS: {current_fps:.1f} | "
        f"Фокус: {current_focal_px:.0f}px | Href: {int(distance_model.reference_height_px)} | "
        f"Danger>=P{danger_priority_threshold:.2f} | Калибр: {calib_state} | Обновлений: {calib_updates}"
    )
    
    # Фон для метрик
    cv2.rectangle(frame, (0, frame_height-35), (frame_width, frame_height), (0, 0, 0), -1)
    frame = put_cyrillic_text(frame, metrics_line, (10, frame_height-20), font_scale=0.45, color=(0, 200, 100), thickness=1)

    cv2.imshow('Blind Navigation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if AUDIO_ENABLED:
    speech_queue.join()
speech_queue.put(None)

try:
    focal_file.write_text(
        json.dumps(
            {
                'focal_px': distance_model.focal_px,
                'reference_height_px': distance_model.reference_height_px,
                'danger_dataset': str(danger_dataset_path),
                'updated_at': int(time.time())
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding='utf-8'
    )
    print(f"Калибровка сохранена: {focal_file} (focal_px={distance_model.focal_px:.1f}, ref_h={distance_model.reference_height_px:.0f})")
except Exception as ex:
    print(f"Не удалось сохранить калибровку: {ex}")
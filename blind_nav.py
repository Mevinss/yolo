import cv2
import pyttsx3
import threading
import time
import argparse
import queue
import json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

DANGERS = {
    'person': {'name': 'Человек', 'height_m': 1.68, 'priority': 1.0},
    'car': {'name': 'Машина', 'height_m': 1.50, 'priority': 1.0},
    'truck': {'name': 'Грузовик', 'height_m': 3.20, 'priority': 1.0},
    'bus': {'name': 'Автобус', 'height_m': 3.00, 'priority': 1.0},
    'motorcycle': {'name': 'Мотоцикл', 'height_m': 1.20, 'priority': 0.95},
    'bicycle': {'name': 'Велосипед', 'height_m': 1.10, 'priority': 0.9},
    'dog': {'name': 'Собака', 'height_m': 0.60, 'priority': 0.85},
    'cat': {'name': 'Кошка', 'height_m': 0.35, 'priority': 0.6},
    'traffic light': {'name': 'Светофор', 'height_m': 2.50, 'priority': 0.7},
    'stop sign': {'name': 'Знак стоп', 'height_m': 2.00, 'priority': 0.75},
    'bench': {'name': 'Скамейка', 'height_m': 0.90, 'priority': 0.8},
    'chair': {'name': 'Стул', 'height_m': 0.90, 'priority': 0.8},
    'dining table': {'name': 'Стол', 'height_m': 0.75, 'priority': 0.8},
    'potted plant': {'name': 'Растение', 'height_m': 0.70, 'priority': 0.6},
    'fire hydrant': {'name': 'Гидрант', 'height_m': 0.75, 'priority': 0.8},
    'parking meter': {'name': 'Стойка', 'height_m': 1.40, 'priority': 0.7},
    'skateboard': {'name': 'Скейтборд', 'height_m': 0.20, 'priority': 0.55},
    'suitcase': {'name': 'Чемодан', 'height_m': 0.65, 'priority': 0.7},
    'backpack': {'name': 'Рюкзак', 'height_m': 0.55, 'priority': 0.65},
}

GENERAL_LABELS = {
    'laptop': 'Ноутбук',
    'cell phone': 'Телефон',
    'mouse': 'Мышь',
    'keyboard': 'Клавиатура',
    'book': 'Книга',
    'bottle': 'Бутылка',
    'cup': 'Кружка',
    'tv': 'Телевизор',
}

last_spoken = {}
COOLDOWN = 2.5
speech_queue = queue.Queue()
speech_error_logged = False
AUDIO_ENABLED = True

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

def estimate_distance_m(label, bbox_height_px, focal_px):
    if bbox_height_px <= 0 or label not in DANGERS:
        return None
    real_height = DANGERS[label]['height_m']
    return (real_height * focal_px) / bbox_height_px

def estimate_focal_px(label, bbox_height_px, distance_m):
    if bbox_height_px <= 0 or distance_m <= 0 or label not in DANGERS:
        return None
    real_height = DANGERS[label]['height_m']
    return (bbox_height_px * distance_m) / real_height

def format_distance(distance_m):
    if distance_m is None:
        return ''
    return f"~{distance_m:.1f}м"

def get_display_label(label):
    return GENERAL_LABELS.get(label, label)

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

    for (x1, x2, distance_m) in detections:
        center = (x1 + x2) / 2
        weight = 1.0 / max(distance_m, 0.3) if distance_m is not None else 1.0
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
parser.add_argument('--audio', action='store_true', default=True, help='Enable voice alerts (enabled by default)')
parser.add_argument('--no-audio', dest='audio', action='store_false', help='Disable voice alerts')
parser.add_argument('--show-general', action='store_true', default=True, help='Show non-danger detections on screen')
parser.add_argument('--no-general', dest='show_general', action='store_false', help='Hide non-danger detections')
parser.add_argument(
    '--focal-file',
    default='focal_calibration.json',
    help='Path to focal calibration JSON file (loaded if exists and saved on exit)'
)
args = parser.parse_args()

AUDIO_ENABLED = bool(args.audio)

focal_file = Path(args.focal_file)
current_focal_px = float(args.focal_px)

if focal_file.exists():
    try:
        saved = json.loads(focal_file.read_text(encoding='utf-8'))
        loaded_focal = float(saved.get('focal_px', current_focal_px))
        if loaded_focal > 0:
            current_focal_px = loaded_focal
            print(f"Загружена калибровка focal_px={current_focal_px:.1f} из {focal_file}")
    except Exception as ex:
        print(f"Не удалось загрузить калибровку: {ex}")

model = YOLO(args.model)
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
    best_calib = None

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in DANGERS:
                box_h = max(y2 - y1, 1)
                distance_m = estimate_distance_m(label, box_h, current_focal_px)
                danger_detections.append((x1, x2, distance_m))
                position = get_position(x1, x2, frame_width)

                if args.auto_calibrate and label == calib_label:
                    measured_focal = estimate_focal_px(label, box_h, args.calib_distance_m)
                    if measured_focal is not None:
                        calib_score = conf * box_h
                        if best_calib is None or calib_score > best_calib['score']:
                            best_calib = {'score': calib_score, 'measured_focal': measured_focal}

                # Три цвета BBox: красный (близко < 1м), оранжевый (1-4м), жёлтый (4+ м)
                if distance_m is not None:
                    if distance_m < 1.0:
                        color = (0, 0, 255)  # Красный - опасно близко
                        status_text = "🔴 БЛИЗКО"
                    elif distance_m < 4.0:
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
                info_line1 = f"{DANGERS[label]['name']} | {dist_text}"
                info_line2 = f"{position.upper()} | {conf_text}"
                
                # Фон для текста
                cv2.rectangle(frame, (x1, y1-50), (x2, y1), color, -1)
                frame = put_cyrillic_text(frame, info_line1, (x1+5, y1-35), font_scale=0.55, color=(255,255,255), thickness=1)
                frame = put_cyrillic_text(frame, info_line2, (x1+5, y1-18), font_scale=0.5, color=(255,255,255), thickness=1)

                if distance_m is not None and distance_m < args.alert_distance_m:
                    close_alerts.append({
                        'label': label,
                        'position': position,
                        'distance_m': distance_m,
                        'conf': conf,
                    })
            else:
                if args.show_general and conf >= args.general_conf:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
                    caption = f"{get_display_label(label)} | {int(conf * 100)}%"
                    text_top = max(0, y1 - 22)
                    text_right = min(frame_width - 1, x1 + 260)
                    cv2.rectangle(frame, (x1, text_top), (text_right, y1), (40, 90, 40), -1)
                    frame = put_cyrillic_text(frame, caption, (x1 + 4, text_top + 4), font_scale=0.5, color=(255, 255, 255), thickness=1)

    if args.auto_calibrate and best_calib is not None:
        alpha = min(max(args.calib_alpha, 0.01), 0.5)
        current_focal_px = (1 - alpha) * current_focal_px + alpha * best_calib['measured_focal']
        calib_updates += 1

    now = time.time()
    if close_alerts and now - last_main_alert_time > MAIN_ALERT_COOLDOWN:
        close_alerts.sort(key=lambda item: item['distance_m'])
        spoken_count = 0
        max_voice_alerts = max(1, args.max_voice_alerts)
        for alert in close_alerts:
            if spoken_count >= max_voice_alerts:
                break
            dist_text = format_distance(alert['distance_m'])
            conf_text = f"{int(alert['conf']*100)}%"
            key = f"{alert['label']}|{alert['position']}|{int(alert['distance_m'] * 10)}"
            if should_speak(key):
                spoken = f"{DANGERS[alert['label']]['name']} {dist_text} {alert['position']} уверенность {conf_text}".strip()
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
    metrics_line = f"📹 {frame_width}x{frame_height} | FPS: {current_fps:.1f} | Фокус: {current_focal_px:.0f}px | Калибр: {calib_state} | Обновлений: {calib_updates}"
    
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
        json.dumps({'focal_px': current_focal_px, 'updated_at': int(time.time())}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"Калибровка сохранена: {focal_file} (focal_px={current_focal_px:.1f})")
except Exception as ex:
    print(f"Не удалось сохранить калибровку: {ex}")
from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = YOLO(str(BASE_DIR / 'runs/blind_nav_model/weights/best.pt'))

results = model(str(BASE_DIR / 'YOLO-OD-main/demo/demo.jpg'), conf=0.3)

for result in results:
    print("Найдено объектов:", len(result.boxes))
    for box in result.boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"  - {label}: {conf:.0%}")
    
    output_file = BASE_DIR / 'test_result.jpg'
    result.save(filename=str(output_file))
    print(f"Результат сохранён: {output_file}")
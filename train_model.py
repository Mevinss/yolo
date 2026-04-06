import os
import shutil
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
RUNS_DIR = BASE_DIR / "runs"

parser = argparse.ArgumentParser(description='Train blind navigation detector')
parser.add_argument('--model', default='yolov8s.pt', help='Base YOLO weights (e.g. yolov8n.pt, yolov8s.pt)')
parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
parser.add_argument('--imgsz', type=int, default=640, help='Training image size')
parser.add_argument('--batch', type=int, default=8, help='Batch size')
parser.add_argument('--device', default='cpu', help='Device for training, e.g. cpu or 0')
parser.add_argument('--name', default='blind_nav_model_v2', help='Run name')
args = parser.parse_args()

datasets = {
    "pothole": str(DATASETS_DIR / "pothole detection"),
    "sidewalk": str(DATASETS_DIR / "sidewalk obstacle"),
    "stairs": str(DATASETS_DIR / "stairs detection"),
    "pedestrian": str(DATASETS_DIR / "pedestrian detection"),
}

FIXED_CLASSES = {
    "pothole": ["pothole"],
    "sidewalk": ["obstacle"],
    "stairs": ["stairs"],
    "pedestrian": ["person"],
}

output = str(DATASETS_DIR / "combined")

for split in ['train', 'valid', 'test']:
    os.makedirs(f"{output}/{split}/images", exist_ok=True)
    os.makedirs(f"{output}/{split}/labels", exist_ok=True)

all_classes = []
class_mapping = {}
current_id = 0

for ds_name, ds_path in datasets.items():
    if not os.path.exists(ds_path):
        print(f"Пропуск {ds_name}: папка не найдена -> {ds_path}")
        continue

    classes = FIXED_CLASSES[ds_name]
    print(f"{ds_name}: {classes}")

    local_to_global = {}
    for i, cls in enumerate(classes):
        if cls not in all_classes:
            all_classes.append(cls)
            class_mapping[cls] = current_id
            current_id += 1
        local_to_global[i] = class_mapping[cls]

    for split in ['train', 'valid', 'test']:
        img_dir = f"{ds_path}/{split}/images"
        lbl_dir = f"{ds_path}/{split}/labels"

        if not os.path.exists(img_dir):
            continue

        for img_file in os.listdir(img_dir):
            src_img = f"{img_dir}/{img_file}"
            dst_img = f"{output}/{split}/images/{ds_name}_{img_file}"
            shutil.copy2(src_img, dst_img)

            lbl_file = os.path.splitext(img_file)[0] + '.txt'
            src_lbl = f"{lbl_dir}/{lbl_file}"
            dst_lbl = f"{output}/{split}/labels/{ds_name}_{lbl_file}"

            if os.path.exists(src_lbl):
                with open(src_lbl, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_id = int(parts[0])
                        if old_id not in local_to_global:
                            continue
                        new_id = local_to_global[old_id]
                        new_lines.append(f"{new_id} {' '.join(parts[1:5])}\n")

                with open(dst_lbl, 'w') as f:
                    f.writelines(new_lines)

combined_yaml = {
    'path': output,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': len(all_classes),
    'names': all_classes
}

with open(f"{output}/data.yaml", 'w') as f:
    yaml.dump(combined_yaml, f)

print(f"\nВсего классов: {len(all_classes)}")
print(f"Классы: {all_classes}")
print(f"Начинаем обучение на {args.device}...")

model = YOLO(args.model)

model.train(
    data=f"{output}/data.yaml",
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    name=args.name,
    project=str(RUNS_DIR),
    device=args.device,
    workers=4,
    cache=False,
    pretrained=True,
    patience=12,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5,
    translate=0.1,
    scale=0.35,
    fliplr=0.5,
    mosaic=0.8,
    mixup=0.1,
    save=True,
    exist_ok=True,
)

print("\nОбучение завершено!")
print(f"Веса: {RUNS_DIR / args.name / 'weights/best.pt'}")
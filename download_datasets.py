import requests
import zipfile
import os
from pathlib import Path

API_KEY = "vAulDHYp5NMUESSMCL3L"
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

datasets = [
    {
        "name": "pothole",
        "workspace": "roboflow-100",
        "project": "pothole-detection-9innj",
        "version": 2,
        "path": str(DATASETS_DIR / "pothole")
    },
    {
        "name": "pedestrian",
        "workspace": "roboflow-100", 
        "project": "pedestrian-detection-bpizf",
        "version": 2,
        "path": str(DATASETS_DIR / "pedestrian")
    },
    {
        "name": "stairs",
        "workspace": "roboflow-100",
        "project": "stairs-detection-odgax",
        "version": 1,
        "path": str(DATASETS_DIR / "stairs")
    }
]

for ds in datasets:
    print(f"Скачиваем {ds['name']}...")
    url = f"https://api.roboflow.com/{ds['workspace']}/{ds['project']}/{ds['version']}/yolov8?api_key={API_KEY}"
    
    r = requests.get(url)
    data = r.json()
    
    if 'export' in data:
        download_url = data['export']['link']
        print(f"Загружаем файл...")
        
        os.makedirs(ds['path'], exist_ok=True)
        zip_path = f"{ds['path']}/dataset.zip"
        
        r2 = requests.get(download_url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r2.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Распаковываем...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(ds['path'])
        
        os.remove(zip_path)
        print(f"{ds['name']} готов!")
    else:
        print(f"Ошибка: {data}")

print("Все датасеты скачаны!")
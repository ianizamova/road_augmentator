import os
import requests
import tarfile
import zipfile
from tqdm import tqdm
import hashlib

# Конфигурация
COCO_DIR = "/media/irina/adata/data/datasets/coco"  # Папка для сохранения
TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# MD5 хеши для проверки (опционально)
FILE_HASHES = {
    "train2017.zip": "cced6f7f71b7629ddf16f17bbcfab6b2",
    "val2017.zip": "442b8da7639aecaf257c1dceb8ba8c80",
    "annotations_trainval2017.zip": "f4bbac642086de4f52a3fdda2de5fa2c"
}

def download_file(url, filename):
    """Скачивает файл с прогресс-баром"""
    os.makedirs(COCO_DIR, exist_ok=True)
    filepath = os.path.join(COCO_DIR, filename)
    
    # Проверяем, не скачан ли файл ранее
    if os.path.exists(filepath):
        print(f"Файл {filename} уже существует, пропускаем загрузку")
        return filepath
    
    print(f"Скачивание {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    # Проверка хеша (опционально)
    if filename in FILE_HASHES:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != FILE_HASHES[filename]:
            raise ValueError(f"Ошибка проверки хеша для {filename}")
    
    return filepath

def extract_zip(zip_path, target_dir):
    """Распаковывает ZIP-архив"""
    print(f"Распаковка {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def main():
    # Скачиваем основные файлы
    train_zip = download_file(TRAIN_URL, "train2017.zip")
    val_zip = download_file(VAL_URL, "val2017.zip")
    annotations_zip = download_file(ANNOTATIONS_URL, "annotations_trainval2017.zip")
    
    # Распаковываем
    extract_zip(train_zip, COCO_DIR)
    extract_zip(val_zip, COCO_DIR)
    extract_zip(annotations_zip, COCO_DIR)
    
    # Создаем симлинки для удобства (опционально)
    os.symlink(
        os.path.join(COCO_DIR, "annotations"), 
        os.path.join(COCO_DIR, "annotations_trainval2017")
    )
    
    print("Датасет успешно загружен и распакован!")
    print(f"Структура папок:\n{os.listdir(COCO_DIR)}")

if __name__ == "__main__":
    main()
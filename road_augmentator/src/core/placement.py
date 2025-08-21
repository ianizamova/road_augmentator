import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import torch
import torchvision.transforms as T

# Конфигурация
DATASET_DIR = "/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/test/testB"  # Папка с изображениями дорог
OUTPUT_DIR = "placement_annotations"  # Куда сохранять разметку
MIN_AREA = 5000  # Минимальная площадь зоны (в пикселях)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Создаем папку для результатов
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загружаем предобученную модель сегментации
model = deeplabv3_resnet50(pretrained=True).to(DEVICE)
model.eval()

# Трансформы для изображений
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def segment_road(image: np.ndarray) -> np.ndarray:
    """Сегментация дороги на изображении."""
    input_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output = output.argmax(0).cpu().numpy()
    return (output == 0).astype(np.uint8)  # Класс 0 = дорога в Cityscapes

def find_placement_zones(mask: np.ndarray) -> list:
    """Находит зоны для размещения объектов на маске дороги."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= MIN_AREA:
            zones.append({
                "bbox": [x, y, x+w, y+h],
                "area": area,
                "mask": mask[y:y+h, x:x+w].tolist()  # Опционально
            })
    return zones

def process_dataset():
    annotations = {}
    for img_name in tqdm(os.listdir(DATASET_DIR)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(DATASET_DIR, img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Сегментация дороги
        road_mask = segment_road(image)
        
        # Поиск зон размещения
        zones = find_placement_zones(road_mask)
        
        # Сохраняем результаты
        annotations[img_name] = {
            "image_size": image.shape[:2],
            "placement_zones": zones
        }

    # Сохраняем разметку в JSON
    with open(os.path.join(OUTPUT_DIR, "placement_zones.json"), "w") as f:
        json.dump(annotations, f, indent=2)

if __name__ == "__main__":
    process_dataset()
    
 
class RoadSegmenter:
    def __init__(self, model_config: dict):
        self.model = load_model(model_config)
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Возвращает маску дороги"""
        # Реализация сегментации
        return road_mask

class PlacementValidator:
    def find_optimal_placement(self, road_mask: np.ndarray) -> tuple:
        """Находит координаты (x,y) для размещения объекта"""
        # Анализ маски и геометрии
        return x, y
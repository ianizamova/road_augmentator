import cv2
import numpy as np
import torch
from torchvision import models, transforms
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from PIL import Image 
from datetime import datetime

class RoadZoneDetector:
    def __init__(self, output_dir="output"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_segmentation_model()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.output_dir = Path(output_dir)
        self.visualizations_dir = self.output_dir / "visualizations"
        self.annotations_dir = self.output_dir / "annotations"
        
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.annotations = {}
        
    def _load_segmentation_model(self):
        """Загружает предобученную модель сегментации дороги"""
        #model = deeplabv3_resnet50(pretrained=True, weights=DeepLabV3_ResNet50_Weights. ).to(self.device)
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=19)
        model.load_state_dict(torch.load("/home/irina/work/otus_cv/blackswan_generator/weights/pytorch_model.bin"), strict=False)
        model.eval()
        return model.to(self.device)

    def segment_road(self, image: np.ndarray) -> np.ndarray:
        """Сегментирует проезжую часть на изображении"""
        input_tensor = self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        road_mask = (output.argmax(0) == 12).cpu().numpy()  # Класс 0 - дорога в Cityscapes
        return road_mask.astype(np.uint8) * 255

    def find_placement_zones(self, road_mask: np.ndarray, num_zones: int = 7) -> List[Dict]:
        """
        Находит зоны размещения объектов на проезжей части
        с учетом перспективы и формы дороги
        """
        # Находим контур дороги
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        
        # Берем основной контур дороги
        main_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_contour)
        
        # Создаем маску выпуклой оболочки
        hull_mask = np.zeros_like(road_mask)
        cv2.drawContours(hull_mask, [hull], 0, 255, -1)
        
        # Разделяем дорогу на зоны по вертикали
        zones = []
        y_min, y_max = np.min(hull[:,0,1]), np.max(hull[:,0,1])
        zone_height = (y_max - y_min) // num_zones
        
        for i in range(num_zones):
            # Определяем границы зоны по вертикали
            y_start = y_min + i * zone_height
            y_end = y_min + (i + 1) * zone_height
            
            # Создаем маску текущей зоны
            zone_mask = np.zeros_like(road_mask)
            zone_mask[y_start:y_end, :] = hull_mask[y_start:y_end, :]
            
            # Находим связные компоненты в зоне
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zone_mask, 8)
            
            # Для каждой компоненты (кроме фона)
            for j in range(1, num_labels):
                x, y, w, h, area = stats[j]
                
                # Фильтруем слишком маленькие области
                if area < 1000:
                    continue
                    
                # Добавляем зону размещения
                zones.append({
                    "bbox": [x, y, x + w, y + h],
                    "area": area,
                    "zone_id": i,
                    "center": (x + w//2, y + h//2)
                })
        
        return zones

    def filter_zones_by_horizon(self, zones: List[Dict], horizon_y: int) -> List[Dict]:
        """Фильтрует зоны, оставляя только те, где нижняя граница ниже горизонта"""
        return [zone for zone in zones if zone["bbox"][3] > horizon_y]

    def process_image(self, image_path: Path):
        """Обрабатывает одно изображение и сохраняет результаты"""
        try:
            # Загрузка и предобработка изображения
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            
            # Сегментация дороги
            road_mask = self.segment_road(image)
            
            # Определение горизонта (упрощенный метод)
            horizon_y = self.estimate_horizon(road_mask)
            
            # Поиск зон размещения
            all_zones = self.find_placement_zones(road_mask)
            valid_zones = self.filter_zones_by_horizon(all_zones, horizon_y)
            
            # Сохранение результатов
            self.save_annotations(image_path.name, image.shape, horizon_y, all_zones, valid_zones)
            
            # Визуализация
            self.visualize_results(image, road_mask, horizon_y, all_zones, valid_zones, image_path.stem)
            
            return True
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            return False

    def estimate_horizon(self, road_mask: np.ndarray) -> int:
        """Оценивает положение горизонта по маске дороги"""
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return int(road_mask.shape[0] * 0.4)  # Значение по умолчанию
        
        # Находим верхнюю границу дороги
        top_points = []
        for contour in contours:
            for point in contour[:,0,:]:
                top_points.append(point[1])  # Y-координата
        
        horizon_y = min(top_points) if top_points else int(road_mask.shape[0] * 0.4)
        return max(horizon_y, int(road_mask.shape[0] * 0.2))  # Не выше 20% изображения

    def visualize_results(self, image, road_mask, horizon_y, all_zones, valid_zones, img_stem):
        """Визуализирует результаты обнаружения зон"""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Отображение изображения
        ax.imshow(image)
        
        # Маска дороги (полупрозрачная)
        ax.imshow(road_mask, alpha=0.2, cmap='Blues')
        
        # Контур дороги
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='blue', linewidth=2, label='Road Boundary')
        
        # Линия горизонта
        ax.axhline(y=horizon_y, color='yellow', linestyle='--', linewidth=2)
        ax.text(10, horizon_y + 15, "Horizon", color='yellow', fontsize=12,
               bbox=dict(facecolor='black', alpha=0.5))
        
        # Все зоны (красные)
        for zone in all_zones:
            x1, y1, x2, y2 = zone["bbox"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor='red', facecolor='red', alpha=0.2
            )
            ax.add_patch(rect)
        
        # Валидные зоны (зеленые)
        for zone in valid_zones:
            x1, y1, x2, y2 = zone["bbox"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.3
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"Zone {zone['zone_id']}", 
                   color='white', fontsize=10,
                   bbox=dict(facecolor='green', alpha=0.7))
        
        # Легенда
        handles = [
            patches.Patch(facecolor='blue', edgecolor='blue', label='Road Boundary'),
            patches.Patch(facecolor='red', alpha=0.2, label='All Zones'),
            patches.Patch(facecolor='lime', alpha=0.3, label='Valid Zones'),
            plt.Line2D([0], [0], color='yellow', linestyle='--', label='Horizon')
        ]
        ax.legend(handles=handles, loc='upper right')
        
        plt.title(f"Road Placement Zones - {img_stem}")
        output_path = self.visualizations_dir / f"{img_stem}_zones.jpg"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

    def save_annotations(self, img_name, img_shape, horizon_y, all_zones, valid_zones):
        """Сохраняет аннотации в JSON файл"""
        self.annotations[img_name] = {
            "image_size": list(img_shape[:2]),
            "horizon_y": int(horizon_y),
            "all_zones": [{
                "bbox": list(map(int, zone["bbox"])),
                "area": int(zone["area"]),
                "zone_id": int(zone["zone_id"]),
                "center": list(map(int, zone["center"]))
            } for zone in all_zones],
            "valid_zones": [{
                "bbox": list(map(int, zone["bbox"])),
                "area": int(zone["area"]),
                "zone_id": int(zone["zone_id"]),
                "center": list(map(int, zone["center"]))
            } for zone in valid_zones],
            "timestamp": datetime.now().isoformat()
        }

    def process_directory(self, input_dir: str):
        """Обрабатывает все изображения в директории"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        print(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            self.process_image(img_path)
        
        # Сохраняем аннотации
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.annotations_dir / f"zones_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"\nResults saved to:\n"
              f"- Annotations: {output_file}\n"
              f"- Visualizations: {self.visualizations_dir}")


# Пример использования
if __name__ == "__main__":
    # Инициализация с указанием папки для результатов
    detector = RoadZoneDetector(output_dir="road_augmentation_results")
    
    # Обработка всех изображений в папке
    detector.process_directory("/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/test/testB")
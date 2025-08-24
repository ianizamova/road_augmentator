import os
import json
import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

class ObjectExtractor:
    def __init__(self, config_path=None):
        """
        Инициализация экстрактора объектов
        
        Args:
            config_path (str): Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self.coco = None
        self.class_ids = []
        
    def _load_config(self, config_path):
        """Загрузка конфигурации из файла или использование значений по умолчанию"""
        default_config = {
            "dataset_path": "/media/irina/ADATA HD330/data/datasets/coco",
            "annotation_file": "annotations/instances_train2017.json",
            "output_dir": "extracted_objects_elgi",
            "target_classes": ["elephant", 'giraffe'],
            "image_folder": "train2017",
            "save_format": "png",
            "crop_to_bbox": True,
            "preserve_aspect_ratio": True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Обновляем дефолтную конфигурацию пользовательскими настройками
                default_config.update(user_config)
                print(f"Конфигурация загружена из {config_path}")
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}. Использую значения по умолчанию")
        
        return default_config
    
    def save_config(self, config_path):
        """Сохранение текущей конфигурации в файл"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Конфигурация сохранена в {config_path}")
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
    
    def initialize(self):
        """Инициализация COCO и подготовка к извлечению"""
        try:
            annotation_path = os.path.join(
                self.config["dataset_path"], 
                self.config["annotation_file"]
            )
            self.coco = COCO(annotation_path)
            
            # Получаем ID целевых классов
            self.class_ids = self.coco.getCatIds(catNms=self.config["target_classes"])
            
            # Создаем выходную директорию
            os.makedirs(self.config["output_dir"], exist_ok=True)
            
            print(f"Инициализация завершена. Найдено классов: {len(self.class_ids)}")
            return True
            
        except Exception as e:
            print(f"Ошибка инициализации: {e}")
            return False
    
    def get_image_ids(self):
        """Получение ID изображений, содержащих целевые классы"""
        if not self.coco:
            print("COCO не инициализирован")
            return []
        
        image_ids = []
        for class_id in self.class_ids:
            image_ids.extend(self.coco.getImgIds(catIds=class_id))
        
        return list(set(image_ids))  # Убираем дубликаты
    
    def extract_objects(self):
        """Основной метод для извлечения объектов"""
        if not self.coco:
            print("Сначала выполните initialize()")
            return
        
        image_ids = self.get_image_ids()
        print(f"Найдено {len(image_ids)} изображений для обработки")
        
        total_objects = 0
        
        for img_id in tqdm(image_ids, desc="Извлечение объектов"):
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                img_path = os.path.join(
                    self.config["dataset_path"],
                    self.config["image_folder"],
                    img_info["file_name"]
                )
                
                if not os.path.exists(img_path):
                    print(f"Изображение не найдено: {img_path}")
                    continue
                
                # Загружаем изображение
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Не удалось загрузить изображение: {img_path}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Загружаем аннотации
                ann_ids = self.coco.getAnnIds(
                    imgIds=img_info["id"], 
                    catIds=self.class_ids
                )
                annotations = self.coco.loadAnns(ann_ids)
                
                # Обрабатываем каждую аннотацию
                for i, ann in enumerate(annotations):
                    if self._process_annotation(img, img_info, ann, i):
                        total_objects += 1
                        
            except Exception as e:
                print(f"Ошибка обработки изображения {img_id}: {e}")
                continue
        
        print(f"Сохранено {total_objects} объектов в {self.config['output_dir']}")
        return total_objects
    
    def _process_annotation(self, img, img_info, ann, index):
        """Обработка отдельной аннотации"""
        try:
            # Создаем маску
            mask = self.coco.annToMask(ann)
            
            # Создаем изображение с прозрачным фоном (RGBA)
            object_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            object_img[:, :, :3] = img
            object_img[:, :, 3] = mask * 255
            
            # Обрезаем по bounding box если нужно
            if self.config["crop_to_bbox"]:
                x, y, w, h = ann["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                # Проверяем границы
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                
                if w > 0 and h > 0:
                    cropped = object_img[y:y+h, x:x+w]
                else:
                    print(f"Некорректный bbox для аннотации {ann['id']}")
                    return False
            else:
                cropped = object_img
            
            # Получаем имя класса
            class_name = self.coco.loadCats(ann["category_id"])[0]["name"]
            
            # Сохраняем объект
            filename = f"{class_name}_{img_info['id']}_{index}.{self.config['save_format']}"
            output_path = os.path.join(self.config["output_dir"], filename)
            
            Image.fromarray(cropped).save(output_path)
            return True
            
        except Exception as e:
            print(f"Ошибка обработки аннотации {ann['id']}: {e}")
            return False
        
if __name__ == "__main__":
    # Создаем экземпляр экстрактора
    extractor = ObjectExtractor("configs/extract_config.json")
    
    # Инициализируем
    if extractor.initialize():
        # Запускаем извлечение
        extractor.extract_objects()
    
    # Можно также сохранить текущую конфигурацию
    extractor.save_config("configs/extract_coco_config.json")
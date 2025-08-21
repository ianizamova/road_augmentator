import os
import json
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

class BicycleRiderExtractor:
    def __init__(self, coco_path, output_dir='output_riders'):
        """
        Инициализация класса для извлечения велосипедистов (человек + велосипед) из COCO датасета
        
        :param coco_path: Путь к директории COCO датасета
        :param output_dir: Директория для сохранения результатов
        """
        self.coco_path = coco_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Пути к аннотациям
        self.annot_path = os.path.join(coco_path, 'annotations', 'instances_train2017.json')
        self.img_dir = os.path.join(coco_path, 'train2017')
        
        # Загрузка COCO аннотаций
        self.coco = COCO(self.annot_path)
        
    def extract_bicycle_riders(self):
        """
        Основной метод для извлечения и сохранения велосипедистов (человек + велосипед)
        """
        # Получаем ID категорий
        bicycle_cat_id = self.coco.getCatIds(catNms=['bicycle'])
        person_cat_id = self.coco.getCatIds(catNms=['person'])
        
        # Получаем все изображения с велосипедами и людьми
        img_ids = set(self.coco.getImgIds(catIds=bicycle_cat_id)) & \
                 set(self.coco.getImgIds(catIds=person_cat_id))
        
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_file = os.path.join(self.img_dir, img_info['file_name'])
            
            # Загружаем изображение
            img = Image.open(img_file).convert('RGBA')
            img_array = np.array(img)
            
            # Получаем аннотации
            annot_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(annot_ids)
            
            # Фильтруем аннотации с сегментацией
            bicycles = [a for a in annotations if a['category_id'] in bicycle_cat_id and 'segmentation' in a]
            people = [a for a in annotations if a['category_id'] in person_cat_id and 'segmentation' in a]
            
            # Для каждого велосипеда ищем связанных людей
            for bike in bicycles:
                bike_mask = self.coco.annToMask(bike)
                
                for person in people:
                    person_mask = self.coco.annToMask(person)
                    
                    if self._is_rider(bike_mask, person_mask):
                        # Комбинированная маска (велосипед + человек)
                        combined_mask = np.logical_or(bike_mask, person_mask)
                        self._save_combined_rider(img_array, combined_mask, 
                                                img_info['file_name'], 
                                                bike['id'], person['id'])
    
    def _is_rider(self, bike_mask, person_mask, min_overlap=0.2):
        """
        Проверяет связь между велосипедом и человеком
        """
        intersection = np.logical_and(bike_mask, person_mask)
        return np.sum(intersection) > 0  # Простое пересечение
    
    def _save_combined_rider(self, img_array, combined_mask, 
                           original_filename, bike_id, person_id):
        """
        Сохраняет комбинированный объект (велосипед + человек) на прозрачном фоне
        """
        # Создаем прозрачное изображение
        result = np.zeros_like(img_array)
        result[..., 3] = 0  # Альфа-канал = 0 (прозрачный)
        
        # Копируем только пиксели под маской
        result[combined_mask == 1] = img_array[combined_mask == 1]
        
        # Находим границы объекта
        rows = np.any(combined_mask, axis=1)
        cols = np.any(combined_mask, axis=0)
        if len(np.where(rows)[0]) > 0 and len(np.where(cols)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # Обрезаем по границам
            cropped = result[ymin:ymax+1, xmin:xmax+1]
            
            # Сохраняем
            rider_img = Image.fromarray(cropped)
            base_name = os.path.splitext(original_filename)[0]
            output_path = os.path.join(
                self.output_dir, 
                f'{base_name}_bike_{bike_id}_rider_{person_id}.png'
            )
            rider_img.save(output_path)
            print(f'Saved combined rider to {output_path}')

if __name__ == '__main__':
    coco_path = '/media/irina/ADATA HD330/data/datasets/coco'  # Укажите ваш путь
    extractor = BicycleRiderExtractor(coco_path)
    extractor.extract_bicycle_riders()
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class ObjectInserter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def insert_object(self, background_path, object_path, positions, enhance=True):
        """
        Вставляет объект на фон с улучшением качества
        
        :param background_path: путь к фоновому изображению
        :param object_path: путь к объекту с прозрачностью (PNG)
        :param x, y: координаты для вставки
        :param enhance: улучшать ли интеграцию (True/False)
        :return: результирующее изображение
        """
        # Загрузка изображений
        bg = cv2.imread(background_path)
        if bg is None:
            raise ValueError(f"Не удалось загрузить фоновое изображение: {background_path}")
        
        obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
        if obj is None:
            raise ValueError(f"Не удалось загрузить объект: {object_path}")

        # Конвертация в RGB для работы с PIL
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        
        # it is need to resize object and then use only resized one
        obj_height_list = np.ones(len(positions)) * obj.shape[1]
        best_position = positions[np.argmin(np.abs([p['height'] for p in positions] - obj_height_list))]
        
        position = best_position
        h_ins = position['height']
        ar = h_ins/obj.shape[1]
        w_ins = int(obj.shape[0] * ar)
        
        obj_resized =  obj #cv2.resize(obj, dsize=(w_ins, h_ins), interpolation=cv2.INTER_CUBIC)
        
        # Базовое наложение объекта
        result = self._basic_blend(bg, obj_resized, position['x'], position['y'])
        
        # Улучшение интеграции
        if enhance:
            result = self._enhance_integration(result, obj_resized, position['x'], position['y'])
        
        #cv2.imwrite("blended.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        insertion_mask = self._create_insertion_mask(bg.shape, obj_resized.shape, position)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR), insertion_mask
    
    def _basic_blend(self, bg, obj, x, y):
        """Базовое наложение объекта с альфа-каналом"""
        h, w = obj.shape[:2]
        bg_h, bg_w = bg.shape[:2]
        
        #ar = h_ins/h
        #h_ins = int(h_ins)
        #w_ins = int(w * ar)
        
        # Проверка границ
        #if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        #    raise ValueError("Объект выходит за границы фона")
        #obj_resized = cv2.resize(obj, dsize=(w_ins, h_ins), interpolation=cv2.INTER_CUBIC)
        # Создаем маску из альфа-канала
        if obj.shape[2] == 4:
            obj_img = obj[:, :, :3]
            obj_mask = obj[:, :, 3] / 255.0
        else:
            obj_img = obj
            obj_mask = np.ones((h, w))
        
        # Копируем фон
        result = bg.copy()
        
        # Наложение с учетом прозрачности
        for c in range(3):
            result[y:y+h, x:x+w, c] = (
                obj_img[:, :, c] * obj_mask +
                result[y:y+h, x:x+w, c] * (1 - obj_mask))
        #cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.rectangle(result, (x, y), (x + w_ins, y + h_ins), (0, 255, 0), 2)
        #cv2.imwrite("insert_img.jpg", result)
        return result
    
    def _enhance_integration(self, image, obj, x, y):
        """Улучшение интеграции объекта с фоном"""
        # 1. Сглаживание границ
        image = self._blend_edges(image, obj, x, y)
        
        # 2. Коррекция цвета (гистограммная согласованность)
        image = self._color_correction(image, obj, x, y)
        
        # 3. Добавление теней
        image = self._add_shadow(image, obj, x, y)
        
        return image
    
    def _blend_edges(self, image, obj, x, y):
        """Сглаживание границ с помощью Gaussian Blur"""
        h, w = obj.shape[:2]
        
        # Создаем маску
        if obj.shape[2] == 4:
            mask = (obj[:, :, 3] > 0).astype(np.uint8) * 255
        else:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Размываем границы маски
        kernel_size = max(3, int(min(h, w) * 0.05) | 1)  # Не менее 3 и нечетное
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0) / 255.0
        
        # Область интереса
        roi = image[y:y+h, x:x+w]
        
        # Пересчитываем альфа-смешение с размытой маской
        for c in range(3):
            roi[:, :, c] = (
                obj[:, :, c] * blurred_mask +
                roi[:, :, c] * (1 - blurred_mask))
                
        return image
    
    def _color_correction(self, image, obj, x, y):
        """Коррекция цвета объекта под фон"""
        h, w = obj.shape[:2]
        
        if obj.shape[2] == 4:
            obj_area = obj[:, :, :3]
            mask = obj[:, :, 3] > 0
        else:
            obj_area = obj
            mask = np.ones((h, w), dtype=bool)
        
        bg_area = image[y:y+h, x:x+w]
        
        # Вычисляем средние цвета
        obj_mean = obj_area[mask].mean(axis=0)
        bg_mean = bg_area[mask].mean(axis=0)
        
        # Коррекция
        correction = bg_mean - obj_mean
        obj_corrected = np.clip(obj_area + correction, 0, 255).astype(np.uint8)
        
        # Применяем коррекцию
        image[y:y+h, x:x+w][mask] = obj_corrected[mask]
        
        return image
    
    def _add_shadow(self, image, obj, x, y):
        """Добавление простой тени"""
        h, w = obj.shape[:2]
        
        if obj.shape[2] == 4:
            mask = (obj[:, :, 3] > 0).astype(np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8)
        
        # Создаем тень (смещение + размытие)
        shadow_mask = np.zeros_like(mask)
        offset_x, offset_y = max(1, int(w*0.03)), max(1, int(h*0.03))
        shadow_mask[offset_y:, offset_x:] = mask[:-offset_y, :-offset_x]
        
        # Размытие тени
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        # Добавляем тень
        shadow_strength = 0.6
        for c in range(3):
            image[y:y+h, x:x+w, c] = (
                image[y:y+h, x:x+w, c] * (1 - shadow_mask * shadow_strength))
                
        return image
    
    def _create_insertion_mask(self, bg_shape, fg_shape, position):
        mask = np.zeros(bg_shape[:2], dtype=np.uint8)
        fg_height, fg_width = fg_shape[:2]
        
        x = max(0, min(position['x'] - fg_width // 2, bg_shape[1] - fg_width))
        y = max(0, min(position['y'] - fg_height // 2, bg_shape[0] - fg_height))
        
        mask[y:y+fg_height, x:x+fg_width] = 255
        return mask
    
    def enhance_quality(self, image):
        """Улучшение качества изображения с помощью PyTorch"""
        try:
            # Преобразуем в PIL Image
            pil_img = Image.fromarray(image)
            
            # Простое улучшение резкости (можно заменить на более сложные методы)
            enhancer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Обратное преобразование
            tensor_img = enhancer(pil_img).unsqueeze(0).to(self.device)
            output = tensor_img.squeeze().cpu().numpy().transpose(1, 2, 0)
            output = (output * 255).astype(np.uint8)
            
            return output
        except Exception as e:
            print(f"Ошибка при улучшении качества: {e}")
            return image
# Пример использования
if __name__ == '__main__':
    inserter = ObjectInserter()
    
    
    # Вставляем объект
    result = inserter.insert_object(
        background_path='/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val/7d15b18b-1e0d6e3f.jpg',
        object_path='/home/irina/work/otus_cv/blackswan_generator/extracted_objects/horse_100599_0.png',
        x=300, y=200,
        enhance=True
    )
    
    
    # Сохраняем результат
    cv2.imwrite('result.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
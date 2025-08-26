import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import datetime

from src.utils.config_loader import ConfigLoader

class ObjectInserter:
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация ObjectInserter с конфигурацией
        
        Args:
            config_path: Путь к JSON конфигу. Если None, используется дефолтный конфиг.
        """
        # Загрузка конфигурации
        self.config = ConfigLoader.get_inserter_config(config_path)
        
        # Настройка устройства
        self.device = self._setup_device()
        
        # Логгер
        self.logger = self._setup_logger()
        
        self.logger.info(f"ObjectInserter initialized on device: {self.device}")
        # Аннотации
        self.annotation_config = self.config["annotation_settings"]
        os.makedirs(self.annotation_config["output_directory"], exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Настройка устройства выполнения на основе конфига"""
        device_config = self.config["device_settings"]
        
        if device_config["device"] == "auto":
            if torch.cuda.is_available():
                device = torch.device('cuda')
                if device_config["verbose_device_info"]:
                    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif device_config["fallback_to_cpu"]:
                device = torch.device('cpu')
                if device_config["verbose_device_info"]:
                    print("Using CPU (CUDA not available)")
            else:
                raise RuntimeError("CUDA not available and fallback to CPU disabled")
        else:
            device = torch.device(device_config["device"])
        
        return device
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def insert_object(self, background_path: str, object_path: str, 
                     positions: List[Dict], enhance: Optional[bool] = None) -> tuple:
        """
        Вставляет объект на фон с улучшением качества
        
        Args:
            background_path: путь к фоновому изображению
            object_path: путь к объекту с прозрачностью (PNG)
            positions: список позиций для размещения
            enhance: улучшать ли интеграцию (переопределяет конфиг)
        
        Returns:
            tuple: (результирующее изображение, маска вставки)
        """
        try:
            # Загрузка изображений
            bg = cv2.imread(background_path)
            if bg is None:
                raise ValueError(f"Не удалось загрузить фоновое изображение: {background_path}")
            
            obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
            if obj is None:
                raise ValueError(f"Не удалось загрузить объект: {object_path}")

            # Выбор лучшей позиции
            best_position = self._select_best_position(positions, obj.shape)
            if best_position is None:
                raise ValueError("Не найдено подходящей позиции для вставки")
            
            # Определяем, нужно ли улучшение
            should_enhance = enhance if enhance is not None else self.config["blending_settings"]["default_enhance"]
            
            # Базовое наложение объекта
            result = self._basic_blend(bg, obj, best_position['x'], best_position['y'])
            
            # Улучшение интеграции
            if should_enhance:
                result = self._enhance_integration(result, obj, best_position['x'], best_position['y'])

            insertion_mask = self._create_insertion_mask(bg.shape, obj.shape, best_position)
            
            # Улучшение качества если включено в конфиге
            if self.config["quality_enhancement"]["enabled"]:
                result = self.enhance_quality(result)
                cv2.imwrite("inserter_examples/quality_enhancement.png", result)
                
            #return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), insertion_mask
            return result, insertion_mask
            
        except Exception as e:
            error_config = self.config["error_handling"]
            if error_config["log_errors"]:
                self.logger.error(f"Error inserting object: {e}")
            if error_config["raise_exceptions"]:
                raise
            return None, None
    
    def _select_best_position(self, positions: List[Dict], obj_shape: tuple) -> Optional[Dict]:
        """Выбор лучшей позиции на основе конфигурации"""
        if not positions:
            return None
        
        selection_config = self.config["position_selection"]
        strategy = selection_config["selection_strategy"]
        
        if strategy == "min_height_difference":
            return self._select_by_min_height_diff(positions, obj_shape)
        else:
            # По умолчанию используем минимальную разницу высот
            return self._select_by_min_height_diff(positions, obj_shape)
    
    def _select_by_min_height_diff(self, positions: List[Dict], obj_shape: tuple) -> Optional[Dict]:
        """Выбор позиции с минимальной разницей высот"""
        obj_height = obj_shape[0]
        max_diff = self.config["position_selection"]["max_relative_height_diff"] * obj_height
        
        valid_positions = []
        for pos in positions:
            if 'height' in pos:
                height_diff = abs(pos['height'] - obj_height)
                if height_diff <= max_diff:
                    valid_positions.append((pos, height_diff))
        
        if not valid_positions:
            if self.config["error_handling"]["skip_invalid_positions"]:
                self.logger.warning("No valid positions found within allowed height difference")
                return None
            else:
                # Возвращаем наименее неподходящую позицию
                return min(positions, key=lambda p: abs(p.get('height', 0) - obj_height))
        
        # Сортируем по минимальной разнице
        valid_positions.sort(key=lambda x: x[1])
        return valid_positions[0][0]
    
    def _basic_blend(self, bg: np.ndarray, obj: np.ndarray, x: int, y: int) -> np.ndarray:
        """Базовое наложение объекта с альфа-каналом"""
        h, w = obj.shape[:2]
        bg_h, bg_w = bg.shape[:2]
        
        # Проверка границ
        if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
            self.logger.warning(f"Object at ({x},{y}) exceeds background boundaries. Clipping.")
            x = max(0, min(x, bg_w - w))
            y = max(0, min(y, bg_h - h))
        
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
        
        return result
    
    def _enhance_integration(self, image: np.ndarray, obj: np.ndarray, x: int, y: int) -> np.ndarray:
        """Улучшение интеграции объекта с фоном на основе конфига"""
        blend_config = self.config["blending_settings"]
        
        if blend_config["edge_smoothing"]:
            image = self._blend_edges(image, obj, x, y)

        if blend_config["color_correction"]:
            image = self._color_correction(image, obj, x, y)

        if blend_config["shadow_effects"]:
            image = self._add_shadow(image, obj, x, y)

        return image
    
    def _blend_edges(self, image: np.ndarray, obj: np.ndarray, x: int, y: int) -> np.ndarray:
        """Сглаживание границ с помощью Gaussian Blur"""
        if not self.config["edge_blending"]["enabled"]:
            return image
        
        h, w = obj.shape[:2]
        
        # Создаем маску
        if obj.shape[2] == 4:
            mask = (obj[:, :, 3] > 0).astype(np.uint8) * 255
        else:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Размываем границы маски
        edge_config = self.config["edge_blending"]
        kernel_size = max(edge_config["min_kernel_size"], 
                         int(min(h, w) * edge_config["kernel_size_ratio"]))
        
        if edge_config["kernel_always_odd"] and kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 
                                      edge_config["gaussian_sigma"]) / 255.0
        
        # Область интереса
        roi = image[y:y+h, x:x+w]
        
        # Пересчитываем альфа-смешение с размытой маской
        for c in range(3):
            roi[:, :, c] = (
                obj[:, :, c] * blurred_mask +
                roi[:, :, c] * (1 - blurred_mask))
                
        return image
    
    def _color_correction(self, image: np.ndarray, obj: np.ndarray, x: int, y: int) -> np.ndarray:
        """Коррекция цвета объекта под фон"""
        if not self.config["color_correction"]["enabled"]:
            return image
        
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
        
        # Применяем веса каналов из конфига
        weights = np.array(self.config["color_correction"]["channel_weights"])
        correction = (bg_mean - obj_mean) * weights * self.config["color_correction"]["correction_strength"]
        
        obj_corrected = np.clip(obj_area + correction, 0, 255).astype(np.uint8)
        
        # Применяем коррекцию
        image[y:y+h, x:x+w][mask] = obj_corrected[mask]
        
        return image
    
    def _add_shadow(self, image: np.ndarray, obj: np.ndarray, x: int, y: int) -> np.ndarray:
        """Добавление простой тени"""
        if not self.config["shadow_effects"]["enabled"]:
            return image
        
        h, w = obj.shape[:2]
        shadow_config = self.config["shadow_effects"]
        
        if obj.shape[2] == 4:
            mask = (obj[:, :, 3] > 0).astype(np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8)
        
        # Создаем тень (смещение + размытие)
        shadow_mask = np.zeros_like(mask)
        
        # Определяем направление тени
        offset_x = max(shadow_config["min_offset"], int(w * shadow_config["offset_ratio_x"]))
        offset_y = max(shadow_config["min_offset"], int(h * shadow_config["offset_ratio_y"]))
        
        if shadow_config["shadow_direction"] == "bottom_right":
            shadow_mask[offset_y:, offset_x:] = mask[:-offset_y, :-offset_x]
        elif shadow_config["shadow_direction"] == "bottom_left":
            shadow_mask[offset_y:, :-offset_x] = mask[:-offset_y, offset_x:]
        else:  # bottom_right по умолчанию
            shadow_mask[offset_y:, offset_x:] = mask[:-offset_y, :-offset_x]
        
        # Размытие тени
        shadow_mask = cv2.GaussianBlur(shadow_mask, 
                                     (shadow_config["blur_kernel_size"], 
                                      shadow_config["blur_kernel_size"]), 
                                     shadow_config["blur_sigma"])
        
        # Добавляем тень
        shadow_strength = shadow_config["shadow_strength"]
        shadow_color = np.array(shadow_config["shadow_color"]) / 255.0
        
        for c in range(3):
            image[y:y+h, x:x+w, c] = (
                image[y:y+h, x:x+w, c] * (1 - shadow_mask * shadow_strength) +
                shadow_color[c] * shadow_mask * shadow_strength * 255)
                
        return image
    
    def _create_insertion_mask(self, bg_shape: tuple, fg_shape: tuple, position: Dict) -> np.ndarray:
        """Создание маски области вставки"""
        mask = np.zeros(bg_shape[:2], dtype=np.uint8)
        fg_height, fg_width = fg_shape[:2]
        
        x = max(0, min(position['x'] - fg_width // 2, bg_shape[1] - fg_width))
        y = max(0, min(position['y'] - fg_height // 2, bg_shape[0] - fg_height))
        
        mask[y:y+fg_height, x:x+fg_width] = 255
        
        if self.config["debug_settings"]["visualize_mask"]:
            debug_path = self.config["debug_settings"]["debug_output_path"]
            os.makedirs(debug_path, exist_ok=True)
            cv2.imwrite(os.path.join(debug_path, "insertion_mask.png"), mask)
        
        return mask
    
    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """Улучшение качества изображения с помощью PyTorch"""
        if not self.config["quality_enhancement"]["enabled"]:
            return image
        
        try:
            # Преобразуем в PIL Image
            pil_img = Image.fromarray(image)
            
            # Простое улучшение резкости
            enhancer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config["quality_enhancement"]["normalize_mean"],
                    std=self.config["quality_enhancement"]["normalize_std"]
                )
            ])
            
            # Обратное преобразование
            tensor_img = enhancer(pil_img).unsqueeze(0).to(self.device)
            output = tensor_img.squeeze().cpu().numpy().transpose(1, 2, 0)
            output = (output * 255).astype(np.uint8)
            
            return output
        except Exception as e:
            self.logger.error(f"Ошибка при улучшении качества: {e}")
            return image
        
        
        
    def insert_object_with_annotation(self, background_path: str, object_path: str, 
                                    positions: List[Dict], object_class: str,
                                    enhance: Optional[bool] = None) -> tuple:
        """
        Вставляет объект и создает аннотации
        
        Args:
            background_path: путь к фоновому изображению
            object_path: путь к объекту
            positions: список позиций
            object_class: класс объекта для аннотации
            enhance: улучшать ли интеграцию
        
        Returns:
            tuple: (изображение, маска, путь к аннотации)
        """
        # Вставляем объект
        result_image, insertion_mask = self.insert_object(
            background_path, object_path, positions, enhance
        )
        
        if result_image is None:
            return None, None, None
        
        # Создаем аннотации
        best_position = self._select_best_position(positions, cv2.imread(object_path, cv2.IMREAD_UNCHANGED).shape)
        annotation_path = self.create_annotation(
            background_path, result_image, best_position, object_class, object_path
        )
        
        return result_image, insertion_mask, annotation_path
    
    def create_annotation(self, original_image_path: str, result_image: np.ndarray,
                         position: Dict, object_class: str, object_path: str) -> str:
        """
        Создает файл аннотации для вставленного объекта
        
        Args:
            original_image_path: путь к исходному фоновому изображению
            result_image: результирующее изображение с объектом
            position: позиция и размеры объекта
            object_class: класс объекта
            object_path: путь к файлу объекта (для дополнительной информации)
        
        Returns:
            Путь к созданному файлу аннотации
        """
        if not self.annotation_config["enabled"]:
            return None
        
        try:
            # Получаем базовое имя файла
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            
            # Создаем bbox координаты
            bbox = self._calculate_bbox(position, result_image.shape)
            
            # Получаем ID класса
            class_id = self._get_class_id(object_class)
            
            # Создаем аннотацию в выбранном формате
            annotation_format = self.annotation_config["default_format"]
            annotation_path = self._create_annotation_file(
                base_name, bbox, class_id, object_class, annotation_format,
                result_image.shape, original_image_path
            )
            
            # Создаем визуализацию если нужно
            if self.annotation_config["save_visualization"]:
                self._create_annotation_visualization(
                    result_image, bbox, class_id, object_class, base_name
                )
            
            self.logger.info(f"Annotation created: {annotation_path}")
            return annotation_path
            
        except Exception as e:
            self.logger.error(f"Error creating annotation: {e}")
            return None
    
    def _calculate_bbox(self, position: Dict, image_shape: tuple) -> Dict[str, int]:
        """
        Вычисляет координаты bounding box
        """
        img_height, img_width = image_shape[:2]
        
        # Центральные координаты
        center_x = position['x']
        center_y = position['y']
        
        # Размеры объекта
        obj_width = position.get('width', 50)
        obj_height = position.get('height', 50)
        
        # Вычисляем координаты bbox
        x_min = max(0, center_x - obj_width // 2)
        y_min = max(0, center_y - obj_height // 2)
        x_max = min(img_width, center_x + obj_width // 2)
        y_max = min(img_height, center_y + obj_height // 2)
        
        # Гарантируем валидность размеров
        bbox_width = max(1, x_max - x_min)
        bbox_height = max(1, y_max - y_min)
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': bbox_width,
            'height': bbox_height,
            'center_x': center_x,
            'center_y': center_y
        }
    
    def _get_class_id(self, object_class: str) -> int:
        """
        Получает ID класса из маппинга
        """
        class_mapping = self.annotation_config["class_mapping"]
        return class_mapping.get(object_class.lower(), class_mapping.get("unknown", 0))
    
    def _create_annotation_file(self, base_name: str, bbox: Dict, class_id: int,
                               class_name: str, format_type: str,
                               image_shape: tuple, image_path: str) -> str:
        """
        Создает файл аннотации в указанном формате
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.annotation_config["include_timestamp"] else ""
        annotation_name = f"{base_name}_{class_name}_{timestamp}" if timestamp else f"{base_name}_{class_name}"
        
        if format_type == "yolo":
            return self._create_yolo_annotation(annotation_name, bbox, class_id, image_shape)
        elif format_type == "coco":
            return self._create_coco_annotation(annotation_name, bbox, class_id, class_name, image_path, image_shape)
        elif format_type == "pascal_voc":
            return self._create_pascal_voc_annotation(annotation_name, bbox, class_id, class_name, image_shape)
        elif format_type == "csv":
            return self._create_csv_annotation(annotation_name, bbox, class_id, class_name, image_path)
        else:
            return self._create_yolo_annotation(annotation_name, bbox, class_id, image_shape)
    
    def _create_yolo_annotation(self, annotation_name: str, bbox: Dict, 
                               class_id: int, image_shape: tuple) -> str:
        """
        Создает аннотацию в формате YOLO
        """
        img_height, img_width = image_shape[:2]
        
        # Нормализованные координаты центра и размеров
        x_center = (bbox['center_x'] / img_width)
        y_center = (bbox['center_y'] / img_height)
        width_norm = (bbox['width'] / img_width)
        height_norm = (bbox['height'] / img_height)
        
        # confidence
        confidence = self.annotation_config["default_confidence"] if self.annotation_config["include_confidence"] else ""
        
        # Создаем содержимое файла
        content = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
        if confidence:
            content += f" {confidence:.2f}"
        
        # Сохраняем файл
        annotation_path = os.path.join(self.annotation_config["output_directory"], f"{annotation_name}.txt")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return annotation_path
    
    def _create_coco_annotation(self, annotation_name: str, bbox: Dict,
                               class_id: int, class_name: str,
                               image_path: str, image_shape: tuple) -> str:
        """
        Создает аннотацию в формате COCO
        """
        img_height, img_width = image_shape[:2]
        
        annotation_data = {
            "image": {
                "file_name": os.path.basename(image_path),
                "height": img_height,
                "width": img_width,
                "id": annotation_name
            },
            "annotations": [{
                "id": 1,
                "image_id": annotation_name,
                "category_id": class_id,
                "category_name": class_name,
                "bbox": [bbox['x_min'], bbox['y_min'], bbox['width'], bbox['height']],
                "area": bbox['width'] * bbox['height'],
                "iscrowd": 0,
                "confidence": self.annotation_config["default_confidence"] if self.annotation_config["include_confidence"] else 1.0
            }],
            "categories": [{
                "id": class_id,
                "name": class_name,
                "supercategory": "object"
            }]
        }
        
        annotation_path = os.path.join(self.annotation_config["output_directory"], f"{annotation_name}.json")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        return annotation_path
    
    def _create_pascal_voc_annotation(self, annotation_name: str, bbox: Dict,
                                    class_id: int, class_name: str,
                                    image_shape: tuple) -> str:
        """
        Создает аннотацию в формате Pascal VOC
        """
        img_height, img_width = image_shape[:2]
        
        # Создаем XML структуру
        root = ET.Element("annotation")
        
        # Информация об изображении
        folder = ET.SubElement(root, "folder")
        folder.text = "generated"
        
        filename = ET.SubElement(root, "filename")
        filename.text = f"{annotation_name}.jpg"
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = "3"
        
        # Объект
        object_elem = ET.SubElement(root, "object")
        ET.SubElement(object_elem, "name").text = class_name
        ET.SubElement(object_elem, "pose").text = "Unspecified"
        ET.SubElement(object_elem, "truncated").text = "0"
        ET.SubElement(object_elem, "difficult").text = "0"
        
        if self.annotation_config["include_confidence"]:
            ET.SubElement(object_elem, "confidence").text = str(self.annotation_config["default_confidence"])
        
        # Bounding box
        bndbox = ET.SubElement(object_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox['x_min'])
        ET.SubElement(bndbox, "ymin").text = str(bbox['y_min'])
        ET.SubElement(bndbox, "xmax").text = str(bbox['x_max'])
        ET.SubElement(bndbox, "ymax").text = str(bbox['y_max'])
        
        # Сохраняем XML
        annotation_path = os.path.join(self.annotation_config["output_directory"], f"{annotation_name}.xml")
        tree = ET.ElementTree(root)
        tree.write(annotation_path, encoding='utf-8', xml_declaration=True)
        
        return annotation_path
    
    def _create_csv_annotation(self, annotation_name: str, bbox: Dict,
                              class_id: int, class_name: str,
                              image_path: str) -> str:
        """
        Создает аннотацию в CSV формате
        """
        confidence = self.annotation_config["default_confidence"] if self.annotation_config["include_confidence"] else ""
        
        csv_data = [
            "image_path,class_id,class_name,x_min,y_min,x_max,y_max,width,height,confidence",
            f"{image_path},{class_id},{class_name},{bbox['x_min']},{bbox['y_min']},"
            f"{bbox['x_max']},{bbox['y_max']},{bbox['width']},{bbox['height']},{confidence}"
        ]
        
        annotation_path = os.path.join(self.annotation_config["output_directory"], f"{annotation_name}.csv")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(csv_data))
        
        return annotation_path
    
    def _create_annotation_visualization(self, image: np.ndarray, bbox: Dict,
                                       class_id: int, class_name: str,
                                       base_name: str) -> None:
        """
        Создает визуализацию аннотации с bounding box
        """
        try:
            # Копируем изображение для визуализации
            vis_image = image.copy()
            
            # Рисуем bounding box
            color = (0, 255, 0)  # Зеленый
            thickness = 2
            
            cv2.rectangle(vis_image, 
                         (bbox['x_min'], bbox['y_min']),
                         (bbox['x_max'], bbox['y_max']),
                         color, thickness)
            
            # Добавляем подпись
            label = f"{class_name} ({class_id})"
            font_scale = 0.6
            font_thickness = 1
            
            # Фон для текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            cv2.rectangle(vis_image,
                         (bbox['x_min'], bbox['y_min'] - text_height - 5),
                         (bbox['x_min'] + text_width, bbox['y_min']),
                         color, -1)
            
            # Текст
            cv2.putText(vis_image, label,
                       (bbox['x_min'], bbox['y_min'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 0, 0), font_thickness)
            
            # Сохраняем визуализацию
            vis_dir = os.path.join(self.annotation_config["output_directory"], "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path = os.path.join(vis_dir, f"{base_name}_{class_name}_visualization.jpg")
            cv2.imwrite(vis_path, vis_image, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.annotation_config["visualization_quality"]])
            
        except Exception as e:
            self.logger.warning(f"Could not create annotation visualization: {e}")
    
    def batch_create_annotations(self, results: List[Tuple], output_dir: str = None) -> List[str]:
        """
        Создает аннотации для пакетной обработки
        
        Args:
            results: список кортежей (image_path, image, positions, object_class, object_path)
            output_dir: целевая директория (опционально)
        
        Returns:
            Список путей к созданным аннотациям
        """
        annotation_paths = []
        
        for result in results:
            image_path, image, positions, object_class, object_path = result
            best_position = self._select_best_position(positions, cv2.imread(object_path, cv2.IMREAD_UNCHANGED).shape)
            
            if best_position:
                annotation_path = self.create_annotation(
                    image_path, image, best_position, object_class, object_path
                )
                annotation_paths.append(annotation_path)
        
        return annotation_paths
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
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
from torchvision import transforms
from typing import List, Dict, Tuple, Optional
import os

from src.utils.config_loader import ConfigLoader

class ObjectPlacer:
    def __init__(self, config_path: Optional[str] = None):
        # Загрузка конфигурации
        self.config = ConfigLoader.get_object_placer_config(config_path)
        
        # Инициализация моделей
        self._initialize_models()
        
    def _initialize_models(self):
        """Инициализация моделей на основе конфигурации"""
        model_config = self.config["model_settings"]
        
        self.segmenter = pipeline(
            "image-segmentation", 
            model=model_config["segmentation_model"],
            device=model_config["device"]
        )
        
        self.depth_estimator = pipeline(
            "depth-estimation", 
            model=model_config["depth_model"],
            device=model_config["device"]
        )
    
    def calc_object_width_from_depth(self, depth: float, base_size: int) -> float:
        """Рассчитывает ширину объекта на основе глубины"""
        depth_config = self.config["depth_estimation"]["depth_calibration"]
        
        if depth <= depth_config["very_close_threshold"]:
            return base_size * depth_config["very_close_scale"]
        elif depth <= depth_config["close_threshold"]:
            return base_size * depth_config["close_scale"]
        else:
            return base_size * depth_config["far_scale"]
    
    def predict_size_and_position(self, image_path: str, object_type: str) -> Optional[List[Dict]]:
        """
        Предсказывает позицию и размер для вставки объекта
        """
        # Загрузка изображения
        image = Image.open(image_path)
        image_np = np.array(image)
        
        # Находим оптимальные позиции
        positions = self._find_optimal_position(image_np, object_type)
        if not positions:
            return None
        
        # Оцениваем глубину
        depth_map = self._estimate_depth(image)
        
        # Рассчитываем размеры объектов
        h, w = image_np.shape[:2]
        road_config = self.config["road_positioning"]
        
        for position in positions:
            x, y = position['x'], position['y']
            
            # Оценка глубины в точке
            depth_value = depth_map[y, x] if y < depth_map.shape[0] and x < depth_map.shape[1] else 0.5
            
            # Расчет размера объекта
            base_size = self.config["reference_sizes"].get(object_type, 100)
            object_width = self.calc_object_width_from_depth(depth_value, base_size)
            
            # Применяем ограничения
            object_width = max(
                road_config["min_object_width"], 
                min(object_width, w * road_config["max_object_width_ratio"])
            )
            
            # Обновляем позицию
            position['x'] = min(int(x - object_width/2), image.width - int(object_width) - 1)
            position['y'] = min(int(y - object_width), image.height - int(object_width))
            position['depth'] = depth_value
            position['object_width'] = int(object_width)
            position['object_type'] = object_type
        
        return positions
    
    def _find_optimal_position(self, image_np: np.ndarray, object_type: str) -> List[Dict]:
        """Находит оптимальные позиции для размещения объекта"""
        segments = self.segmenter(Image.fromarray(image_np))
        valid_surfaces = []
        
        # Проверяем совместимость поверхностей
        compatibility = self.config["object_surface_compatibility"]
        valid_labels = compatibility.get(object_type, compatibility.get("default", []))
        
        for segment in segments:
            if segment['label'].lower() in [label.lower() for label in valid_labels]:
                valid_surfaces.append(np.array(segment['mask']))
        
        if valid_surfaces:
            combined_mask = np.zeros_like(valid_surfaces[0], dtype=np.uint8)
            for mask in valid_surfaces:
                combined_mask = cv2.bitwise_or(combined_mask, mask.astype(np.uint8))
            
            # Получаем позиции на дороге
            road_config = self.config["road_positioning"]
            positions = self.calculate_road_positions(
                combined_mask,
                road_config["positions_per_section"]["lower"],
                road_config["positions_per_section"]["middle"],
                road_config["positions_per_section"]["upper"]
            )
            
            # Визуализация для отладки
            if self.config["visualization"]["save_visualization"]:
                self.visualize_positions(image_np, positions, combined_mask)
            
            return positions
     
        return None
    
    def calculate_road_positions(self, road_mask: np.ndarray, 
                               num_lower_positions: int = 3,
                               num_middle_positions: int = 2,
                               num_upper_positions: int = 1) -> List[Dict]:
        """
        Рассчитывает позиции для размещения объектов на дороге
        """
        contours, _ = cv2.findContours(road_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Берем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.reshape(-1, 2)
        
        # Находим границы контура
        min_y = np.min(contour_points[:, 1])
        max_y = np.max(contour_points[:, 1])
        
        # Вычисляем высоты для разделения на 3 части
        total_height = max_y - min_y
        section_height = (total_height + 150) / 6
        
        # Определяем границы секций
        lower_section_y = max_y - 3 * section_height
        middle_section_y = max_y - 5 * section_height
        upper_section_y = min_y
        
        # Находим позиции для каждой секции
        positions = []
        road_config = self.config["road_positioning"]
        
        # Нижняя часть
        lower_positions = self.find_positions_in_section(
            contour_points, lower_section_y, max_y, num_lower_positions, "lower"
        )
        positions.extend(lower_positions)
        
        # Средняя часть
        middle_positions = self.find_positions_in_section(
            contour_points, middle_section_y, lower_section_y, num_middle_positions, "middle"
        )
        positions.extend(middle_positions)
        
        # Верхняя часть
        upper_positions = self.find_positions_in_section(
            contour_points, upper_section_y, middle_section_y, num_upper_positions, "upper"
        )
        positions.extend(upper_positions)
        
        return positions
    
    def find_positions_in_section(self, contour_points: np.ndarray, 
                                 y_start: float, y_end: float,
                                 num_positions: int, 
                                 section_type: str) -> List[Dict]:
        """
        Находит позиции в заданной вертикальной секции дороги
        """
        y_tolerance = self.config["road_positioning"]["y_tolerance"]
        mask = (contour_points[:, 1] >= y_start) & (contour_points[:, 1] <= y_end)
        section_points = contour_points[mask]
        
        if len(section_points) == 0:
            return []
        
        # Находим среднюю Y координату
        avg_y = np.mean(section_points[:, 1])
        
        # Находим границы дороги
        left_bound, right_bound = self.find_road_bounds_at_y(contour_points, avg_y, y_tolerance)
        if left_bound is None or right_bound is None:
            return []
        
        road_width = right_bound - left_bound
        positions = []
        position_width = road_width / (num_positions + 1)
        
        road_config = self.config["road_positioning"]
        base_scales = road_config["base_scale_factors"]
        adjustment_factor = road_config["position_adjustment_factor"]
        
        for i in range(num_positions):
            # Вычисляем позицию
            x_position = left_bound + (i + 1) * position_width
            
            # Вычисляем размер объекта
            scale_factor = base_scales[section_type] * (1.0 - (i * adjustment_factor))
            object_width = road_width * scale_factor
            object_height = y_end - y_start
            
            positions.append({
                'x': int(x_position),
                'y': int(avg_y),
                'width': int(object_width),
                'height': int(object_height),
                'section': section_type,
                'position_index': i,
                'road_width_at_position': int(road_width),
                'confidence': self.calculate_position_confidence(section_type, i)
            })
        
        return positions
    
    def find_road_bounds_at_y(self, contour_points: np.ndarray, 
                             target_y: float, y_tolerance: float) -> Tuple[float, float]:
        """
        Находит левую и правую границы дороги на заданной высоте
        """
        mask = (contour_points[:, 1] >= target_y - y_tolerance) & \
               (contour_points[:, 1] <= target_y + y_tolerance)
        
        nearby_points = contour_points[mask]
        if len(nearby_points) == 0:
            return None, None
        
        return np.min(nearby_points[:, 0]), np.max(nearby_points[:, 0])
    
    def calculate_position_confidence(self, section_type: str, position_index: int) -> float:
        """
        Вычисляет уверенность в позиции на основе конфигурации
        """
        # Можно добавить в конфиг, но для простоты оставим статичным
        confidences = {
            'lower': [0.9, 0.85, 0.8],
            'middle': [0.75, 0.7],
            'upper': [0.65]
        }
        
        if section_type in confidences and position_index < len(confidences[section_type]):
            return confidences[section_type][position_index]
        return 0.5
    
    def visualize_positions(self, image: np.ndarray, positions: List[Dict], 
                           road_mask: np.ndarray = None) -> None:
        """
        Визуализирует позиции и сохраняет изображение
        """
        vis_config = self.config["visualization"]
        vis_image = image.copy()
        
        # Рисуем контур дороги
        if vis_config["draw_contours"] and road_mask is not None:
            contours, _ = cv2.findContours(road_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(vis_image, contours, -1, 
                                vis_config["contour_color"], 
                                vis_config["contour_thickness"])
        
        # Рисуем позиции
        section_colors = vis_config["section_colors"]
        
        for pos in positions:
            color = section_colors.get(pos['section'], [255, 255, 255])
            
            # Рисуем bounding box
            x, y = pos['x'], pos['y']
            width, height = pos['width'], pos['height']
            
            x1 = x - width // 2
            y1 = y - height
            x2 = x + width // 2
            y2 = y
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_image, (x, y), vis_config["point_radius"], color, -1)
            
            # Подписываем позицию
            label = f"{pos['section']}_{pos['position_index']}"
            cv2.putText(vis_image, label, (x - 20, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, vis_config["label_font_scale"], 
                       color, vis_config["label_thickness"])
        
        # Сохраняем визуализацию
        os.makedirs(vis_config["visualization_path"], exist_ok=True)
        output_path = os.path.join(vis_config["visualization_path"], "road_positions.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    def _estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Оценивает карту глубины"""
        results = self.depth_estimator(image)
        depth_map = np.array(results["depth"])
        
        if self.config["depth_estimation"]["normalize"]:
            depth_map = depth_map / depth_map.max()
        
        return depth_map
    
    def visualize_depth_map(self, image_path, output_path="depth_visualization.jpg", alpha=0.5):
        """
        Визуализирует карту глубины с наложением на исходное изображение
        :param image_path: путь к исходному изображению
        :param output_path: путь для сохранения результата
        :param alpha: прозрачность наложения (0-1)
        :return: Путь к сохранённому изображению
        """
        try:
            # Загрузка изображения
            image = Image.open(image_path)
            
            # Получаем карту глубины
            depth_map = self._estimate_depth(image)
            
            # Нормализуем для визуализации
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            
            # Применяем цветовую карту (JET для классического вида)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Конвертируем оригинал в BGR
            original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Наложение с прозрачностью
            blended = cv2.addWeighted(original_bgr, 1-alpha, depth_colored, alpha, 0)
            
            # Сохраняем результат
            cv2.imwrite(output_path, blended)
            return output_path
            
        except Exception as e:
            print(f"Ошибка при визуализации глубины: {e}")
            return None
    
    def visualize_segmentation(self, image_path, object_type, output_path="segmentation_visualization.jpg"):
        """
        Визуализирует маски сегментации для заданного типа объекта
        :param image_path: путь к исходному изображению
        :param object_type: тип объекта ('car', 'person' и т.д.)
        :param output_path: путь для сохранения результата
        :return: Путь к сохранённому изображению или None при ошибке
        """
        try:
            # Загрузка изображения
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Получаем сегментацию
            segments = self.segmenter(image)
            
            # Создаём пустую маску для визуализации
            visualization = image_np.copy()
            
            # Цвет для выделения (BGR)
            highlight_color = (0, 255, 0)  # Зелёный
            
            # Находим все подходящие поверхности
            for segment in segments:
                if self._is_valid_surface(segment['label'], object_type):
                    mask = np.array(segment['mask'])
                    # Накладываем маску на изображение
                    visualization[mask > 0] = (
                        0.6 * np.array(highlight_color) + 
                        0.4 * visualization[mask > 0]
                    ).astype(np.uint8)
            
            # Сохраняем результат
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            return output_path
            
        except Exception as e:
            print(f"Ошибка при визуализации сегментации: {e}")
            return None

# Пример использования
if __name__ == "__main__":
    placer = ObjectPlacer()
    
     
    image_path = '/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val/7eb7131c-420e1747.jpg'
    # Визуализируем сегментацию для автомобилей
    vis_path = placer.visualize_segmentation(
        image_path = image_path,    
        object_type="horse",
        output_path="horse_segmentation.jpg"
    )
    
    # Визуализация глубины
    depth_vis_path = placer.visualize_depth_map(
        image_path=image_path,
        output_path="depth_overlay.jpg",
        alpha=0.6  # Уровень прозрачности
    )
    
    if depth_vis_path:
        print(f"Визуализация глубины сохранена в {depth_vis_path}")
        
        # # Дополнительная визуализация сегментации
        # seg_vis_path = placer.visualize_segmentation(
        #     image_path="street_scene.jpg",
        #     object_type="car",
        #     output_path="car_segmentation.jpg"
        # )
        
        
    
        if vis_path:
            print(f"Визуализация сохранена в {vis_path}")
            
            # Дополнительно получаем позицию и размер
            results = placer.predict_size_and_position(image_path, "horse")
            if results:
                # Визуализация
                image = cv2.imread("depth_overlay.jpg")
                for result in results:
                    x, y, width = result
                    print(f"Оптимальная позиция: ({x}, {y}), Ширина: {width}px")
                
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
                
                    # Рисуем bounding box предполагаемого размера
                    cv2.rectangle(image, 
                                (int(x - width//2), int(y - width)),
                                (int(x + width//2), int(y)),
                                (0, 255, 0), 2)
                
                cv2.imwrite("result_with_size_and_depth.jpg", image)
            else:
                print("Не удалось найти подходящее место для вставки")  

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
from torchvision import transforms
from typing import List, Dict, Tuple

def calculate_road_positions(road_mask: np.ndarray, 
                           num_lower_positions: int = 3,
                           num_middle_positions: int = 2,
                           num_upper_positions: int = 1) -> List[Dict]:
    """
    Рассчитывает позиции для размещения объектов на дороге, разделяя контур на 3 части по вертикали.
    
    Args:
        road_mask: Маска сегментации дороги (бинарная)
        num_lower_positions: Количество позиций в нижней части
        num_middle_positions: Количество позиций в средней части
        num_upper_positions: Количество позиций в верхней части
    
    Returns:
        Список словарей с информацией о позициях
    """
    # Находим контуры дороги
    contours, _ = cv2.findContours(road_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Берем самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    
    # Находим границы контура
    min_y = np.min(contour_points[:, 1])  # Верхняя граница
    max_y = np.max(contour_points[:, 1])  # Нижняя граница
    
    # Вычисляем высоты для разделения на 3 части
    total_height = max_y - min_y
    section_height = total_height / 3
    
    lower_section_y = max_y - section_height  # Нижняя треть
    middle_section_y = max_y - 2 * section_height  # Средняя треть
    upper_section_y = min_y  # Верхняя треть
    
    # Находим позиции для каждой секции
    positions = []
    
    # 1. Нижняя часть (3 позиции)
    lower_positions = find_positions_in_section(contour_points, 
                                               lower_section_y, max_y, 
                                               num_lower_positions, "lower")
    positions.extend(lower_positions)
    
    # 2. Средняя часть (2 позиции)
    middle_positions = find_positions_in_section(contour_points,
                                                middle_section_y, lower_section_y,
                                                num_middle_positions, "middle")
    positions.extend(middle_positions)
    
    # 3. Верхняя часть (1 позиция)
    upper_positions = find_positions_in_section(contour_points,
                                               upper_section_y, middle_section_y,
                                               num_upper_positions, "upper")
    positions.extend(upper_positions)
    
    return positions

def find_positions_in_section(contour_points: np.ndarray, 
                             y_start: float, y_end: float,
                             num_positions: int, 
                             section_type: str) -> List[Dict]:
    """
    Находит позиции в заданной вертикальной секции дороги.
    """
    # Фильтруем точки в заданном диапазоне Y
    mask = (contour_points[:, 1] >= y_start) & (contour_points[:, 1] <= y_end)
    section_points = contour_points[mask]
    
    if len(section_points) == 0:
        return []
    
    # Находим среднюю Y координату для секции
    avg_y = np.mean(section_points[:, 1])
    
    # Находим левую и правую границы дороги на этой высоте
    left_bound, right_bound = find_road_bounds_at_y(contour_points, avg_y)
    
    if left_bound is None or right_bound is None:
        return []
    
    road_width = right_bound - left_bound
    
    # Вычисляем позиции вдоль ширины дороги
    positions = []
    position_width = road_width / (num_positions + 1)  # Равномерное распределение
    
    for i in range(num_positions):
        # Вычисляем X координату
        x_position = left_bound + (i + 1) * position_width
        
        # Вычисляем рекомендуемый размер объекта на основе положения
        scale_factor = calculate_scale_factor(section_type, i, num_positions)
        object_width = road_width * scale_factor
        object_height = object_width * 0.6  # Сохранение пропорций
        
        positions.append({
            'x': int(x_position),
            'y': int(avg_y),
            'width': int(object_width),
            'height': int(object_height),
            'section': section_type,
            'position_index': i,
            'road_width_at_position': int(road_width),
            'confidence': calculate_position_confidence(section_type, i)
        })
    
    return positions

def find_road_bounds_at_y(contour_points: np.ndarray, target_y: float) -> Tuple[float, float]:
    """
    Находит левую и правую границы дороги на заданной высоте Y.
    """
    # Находим точки, близкие к целевой Y координате
    y_tolerance = 50  # Допуск по Y
    mask = (contour_points[:, 1] >= target_y - y_tolerance) & \
           (contour_points[:, 1] <= target_y + y_tolerance)
    
    nearby_points = contour_points[mask]
    
    if len(nearby_points) == 0:
        return None, None
    
    left_bound = np.min(nearby_points[:, 0])
    right_bound = np.max(nearby_points[:, 0])
    
    return left_bound, right_bound

def calculate_scale_factor(section_type: str, position_index: int, total_positions: int) -> float:
    """
    Вычисляет масштабный коэффициент для размера объекта.
    Объекты становятся меньше по мере удаления (перспектива).
    """
    base_scales = {
        'lower': 0.25,  # Ближние объекты - самые большие
        'middle': 0.18,  # Средние объекты
        'upper': 0.12   # Дальние объекты - самые маленькие
    }
    
    # Дополнительная регулировка в пределах секции
    position_adjustment = 1.0 - (position_index * 0.05)
    
    return base_scales[section_type] * position_adjustment

def calculate_position_confidence(section_type: str, position_index: int) -> float:
    """
    Вычисляет уверенность в позиции (для последующего выбора).
    """
    confidences = {
        'lower': [0.9, 0.85, 0.8],    # Нижняя секция
        'middle': [0.75, 0.7],         # Средняя секция  
        'upper': [0.65]                # Верхняя секция
    }
    
    return confidences[section_type][position_index]

def visualize_positions(image: np.ndarray, positions: List[Dict], 
                       road_mask: np.ndarray = None) -> np.ndarray:
    """
    Визуализирует рассчитанные позиции на изображении.
    """
    vis_image = image.copy()
    
    # Рисуем контур дороги если предоставлена маска
    if road_mask is not None:
        contours, _ = cv2.findContours(road_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
    
    # Цвета для разных секций
    section_colors = {
        'lower': (0, 0, 255),    # Красный - нижняя
        'middle': (0, 165, 255), # Оранжевый - средняя
        'upper': (0, 255, 255)   # Желтый - верхняя
    }
    
    # Рисуем позиции
    for pos in positions:
        color = section_colors[pos['section']]
        
        # Рисуем bounding box
        x, y = pos['x'], pos['y']
        width, height = pos['width'], pos['height']
        
        # Центрируем bounding box относительно позиции
        x1 = x - width // 2
        y1 = y - height
        x2 = x + width // 2
        y2 = y
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Рисуем центральную точку
        cv2.circle(vis_image, (x, y), 5, color, -1)
        
        # Подписываем позицию
        label = f"{pos['section']}_{pos['position_index']}"
        cv2.putText(vis_image, label, (x - 20, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis_image

# Пример использования в вашем пайплайне
def get_road_positions_for_augmentation(road_mask: np.ndarray, 
                                      original_image: np.ndarray = None) -> List[Dict]:
    """
    Основная функция для получения позиций на дороге.
    """
    positions = calculate_road_positions(road_mask)
    
    # Визуализация для отладки (опционально)
    if original_image is not None:
        visualization = visualize_positions(original_image, positions, road_mask)
        cv2.imwrite("visualize_positions.jpg",visualization)
        #cv2.imshow("Road Positions", visualization)
        #cv2.waitKey(1000)  # Показать на 1 секунду
        #cv2.destroyAllWindows()
    
    # Сортируем позиции по уверенности (для приоритетного выбора)
    positions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return positions

# Интеграция с вашим пайплайном
class RoadPositionPredictor:
    def __init__(self, config: Dict):
        self.config = config
    
    def predict_positions(self, background_image: np.ndarray, road_mask: np.ndarray) -> List[Dict]:
        """
        Предсказывает позиции для размещения объектов на дороге.
        """
        try:
            positions = get_road_positions_for_augmentation(road_mask, background_image)
            
            # Логирование
            print(f"Найдено {len(positions)} позиций на дороге:")
            for pos in positions:
                print(f"  {pos['section']}_{pos['position_index']}: "
                      f"({pos['x']}, {pos['y']}), size: {pos['width']}x{pos['height']}, "
                      f"conf: {pos['confidence']:.2f}")
            
            return positions
            
        except Exception as e:
            print(f"Ошибка при предсказании позиций: {e}")
            return []
        

class ObjectPlacer:
    def __init__(self):
        # Инициализация моделей
        self.detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        self.segmenter = pipeline("image-segmentation", model="facebook/maskformer-swin-large-coco")
        
        # Модель для оценки глубины
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
        
        # Эталонные размеры объектов (ширина в пикселях для расстояния 5м)
        self.reference_sizes = {
            'car': 200,
            'person': 80,
            'bicycle': 120,
            'vase': 40,
            'horse': 300,
            'bird': 50
        }
        
    def calc_object_width_from_depth(self, depth, base_size):
        #width = 0
        if depth <= 0.2:
            return base_size * 0.15
        elif depth <= 0.55 and depth > 0.2:
             return base_size* 0.4
        else:
            return base_size * depth
    
    def predict_size_and_position(self, image_path, object_type):
        """
        Предсказывает позицию и размер для вставки объекта
        
        :param image_path: путь к изображению
        :param object_type: тип объекта ('car', 'person' и т.д.)
        :return: (x, y, width) или None, если место не найдено
        """
        # Загрузка изображения
        image = Image.open(image_path)
        image_np = np.array(image)
        
        # Шаг 1: Находим оптимальную позицию
        positions = self._find_optimal_position(image_np, object_type)
        if not positions:
            return None
        
        out_list = []
        for position in positions:   
            x, y = position['x'], position['y']
            
            # Шаг 2: Оцениваем глубину в точке вставки
            depth_map = self._estimate_depth(image)
            depth_value = depth_map[y, x]
            
            # Шаг 3: Рассчитываем размер объекта
            base_size = self.reference_sizes.get(object_type, 100)
            #obj ect_width = int(base_size * (1.0 / (depth_value + 0.1)))  # Простая обратная зависимость
            #object_width = int(base_size * (depth_value + 0.1))  # Прямая зависимость - чем ближе, тем больше число

            object_width = self.calc_object_width_from_depth(depth_value, base_size)

            # Корректировка по минимальному/максимальному размеру
            h, w = image_np.shape[:2]
            object_width = max(30, min(object_width, w//3))
            out_list.append((x, y, object_width))
        
        return out_list
    
    def find_contour_center(self, contour):
        """Находит геометрический центр контура через среднее X и Y"""
        # Преобразуем контур в массив координат точек
        points = contour.reshape(-1, 2)
        
        # Вычисляем среднее по X и Y
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        
        return (center_x, center_y)
    
    def _find_optimal_position(self, image_np, object_type):
        """Находит оптимальную позицию (как в предыдущем примере)"""
        segments = self.segmenter(Image.fromarray(image_np))
        valid_surfaces = []
        positions = []
        for segment in segments:
            if self._is_valid_surface(segment['label'], object_type):
                valid_surfaces.append(np.array(segment['mask']))
        
        if valid_surfaces:
            combined_mask = np.zeros_like(valid_surfaces[0], dtype=np.uint8)
            for mask in valid_surfaces:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            positions = get_road_positions_for_augmentation(combined_mask, image_np)
            #contours, _ = cv2.findContours( combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return positions
            # if contours:
            #     largest_contour = max(contours, key=cv2.contourArea)
            #     #M = cv2.moments(largest_contour)
            #     #if M['m00'] != 0:
            #     #    return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            #     return self.find_contour_center(largest_contour)
                
        return None
    
    def _estimate_depth(self, image):
        """Оценивает карту глубины изображения"""
        results = self.depth_estimator(image)
        depth_map = np.array(results["depth"])
        return depth_map / depth_map.max()  # Нормализация к [0, 1]
    
    def _is_valid_surface(self, label, object_type):
        # Правила совместимости объектов и поверхностей
        compatibility = {
            'car': ['road', 'parking lot', 'street'],
            'vase': ['table', 'shelf', 'desk'],
            'person': ['sidewalk', 'floor', 'grass', 'pavement-merged'],
            'bicycle': ['road', 'sidewalk', 'path'],
            'horse': ['road', 'path', 'street', 'driving-area'],
            'bird': ['sky']
        }
        return label.lower() in [s.lower() for s in compatibility.get(object_type, [])]
    
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

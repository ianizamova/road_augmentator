import cv2
import numpy as np

def find_vanishing_point(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    
    # Фильтруем линии, которые могут быть границами дороги
    road_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2-y1, x2-x1) * 180/np.pi
        if abs(angle) > 10 and abs(angle) < 80:  # Отбираем наклонные линии
            road_lines.append(line[0])
    
    # Находим пересечения всех пар линий
    vanishing_points = []
    for i in range(len(road_lines)):
        for j in range(i+1, len(road_lines)):
            x1, y1, x2, y2 = road_lines[i]
            x3, y3, x4, y4 = road_lines[j]
            
            # Вычисляем точку пересечения
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if denom != 0:
                px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                vanishing_points.append([px, py])
    
    # Усредняем все точки пересечения
    if vanishing_points:
        vp = np.mean(vanishing_points, axis=0)
        return tuple(map(int, vp))
    return None

def get_road_placement_zone(img, vanishing_point, horizon_line):
    height, width = img.shape[:2]
    
    # Создаем маску возможных зон размещения
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Если есть точка схода, строим перспективные зоны
    if vanishing_point:
        vx, vy = vanishing_point
        
        # Определяем основные зоны дороги (треугольник от низа к точке схода)
        pts = np.array([[0, height-1], [width-1, height-1], [vx, vy]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Учитываем линию горизонта (не размещаем объекты выше)
        if horizon_line:
            mask[:horizon_line, :] = 0
    else:
        # Просто нижняя часть изображения
        mask[height//2:, :] = 255
    
    return mask

def calculate_object_size(img, placement_point, vanishing_point, base_size=100):
    height, width = img.shape[:2]
    vx, vy = vanishing_point
    
    # Расчет расстояния до точки схода
    distance_to_vp = np.sqrt((placement_point[0]-vx)**2 + (placement_point[1]-vy)**2)
    max_distance = np.sqrt((width/2-vx)**2 + (height-vy)**2)
    
    # Коэффициент масштабирования (чем ближе к точке схода - тем меньше объект)
    scale_factor = 0.5 + 0.5*(distance_to_vp/max_distance)
    
    # Размер объекта зависит от положения и базового размера
    object_width = int(base_size * scale_factor)
    object_height = int(base_size * scale_factor)
    
    # Корректировка по вертикали (объекты у горизонта должны быть меньше)
    if vanishing_point[1] > 0:
        vertical_scale = 1 - (placement_point[1]/vanishing_point[1])*0.5
        object_width *= vertical_scale
        object_height *= vertical_scale
    
    return max(10, int(object_width)), max(10, int(object_height))

def place_object_on_road(background_img, object_img, object_mask):
    # Находим зону размещения
    vp = find_vanishing_point(background_img)
    horizon = find_horizon_line(background_img)  # Реализацию см. ниже
    road_mask = get_road_placement_zone(background_img, vp, horizon)
    
    # Находим случайную точку на дороге
    road_pixels = np.where(road_mask == 255)
    if len(road_pixels[0]) == 0:
        return background_img
    
    idx = np.random.randint(len(road_pixels[0]))
    center_y, center_x = road_pixels[0][idx], road_pixels[1][idx]
    
    # Рассчитываем размер объекта
    obj_w, obj_h = calculate_object_size(background_img, (center_x, center_y), vp)
    object_img = cv2.resize(object_img, (obj_w, obj_h))
    object_mask = cv2.resize(object_mask, (obj_w, obj_h))
    
    # Вставляем объект с учетом смешивания
    result = background_img.copy()
    y1 = max(0, center_y - obj_h//2)
    y2 = min(background_img.shape[0], center_y + obj_h//2)
    x1 = max(0, center_x - obj_w//2)
    x2 = min(background_img.shape[1], center_x + obj_w//2)
    
    # Обрезаем если выходит за границы
    obj = object_img[:y2-y1, :x2-x1]
    mask = object_mask[:y2-y1, :x2-x1]
    
    # Наложение с учетом маски
    result[y1:y2, x1:x2][mask > 0] = obj[mask > 0]
    
    return result

def find_horizon_line(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    horizon_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > img.shape[1]*0.5:  # Длинные горизонтальные линии
            horizon_candidates.append(max(y1, y2))
    
    if horizon_candidates:
        return int(np.median(horizon_candidates))
    return img.shape[0]//2  # По умолчанию - середина изображения


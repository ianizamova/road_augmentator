import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time

class DepthMapComparator:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.results = {}
        
    def geometric_depth(self):
        """Геометрический метод на основе точки схода"""
        start_time = time()
        
        # Находим точку схода
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        vanishing_point = [self.image.shape[1]//2, self.image.shape[0]//2]  # По умолчанию центр
        if lines is not None:
            intersections = []
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    x3, y3, x4, y4 = lines[j][0]
                    
                    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                    if denom != 0:
                        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                        if 0 <= px < self.image.shape[1] and 0 <= py < self.image.shape[0]:
                            intersections.append([px, py])
            
            if intersections:
                vanishing_point = np.mean(intersections, axis=0)
        
        # Строим карту глубины
        height, width = self.image.shape[:2]
        depth_map = np.zeros((height, width))
        
        vx, vy = vanishing_point
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x-vx)**2 + (y-vy)**2)
                depth_map[y,x] = dist
        
        # Нормализация
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        self.results['Geometric'] = {
            'depth_map': depth_map,
            'time': time() - start_time
        }
    
    def midas_depth(self, model_type='DPT_Large'):
        """Использование модели MiDaS"""
        start_time = time()
        
        # Загрузка модели
        midas = torch.hub.load('intel-isl/MiDaS', model_type)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        midas.to(device)
        midas.eval()
        
        # Трансформации
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms') #224x384
        transform = midas_transforms.default_transform
        
        # Подготовка изображения
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        input_tensor = transform(img).to(device)
        
        # Предсказание
        with torch.no_grad():
            prediction = midas(input_tensor)
            depth_map = prediction.cpu().numpy()
        
        depth_map = cv2.resize(depth_map, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
        # Нормализация
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        
        self.results['MiDaS'] = {
            'depth_map': depth_map.squeeze(),
            'time': time() - start_time
        }
    
    def stereo_depth(self):
        """Стерео метод (упрощенный)"""
        start_time = time()
        
        # Для демонстрации используем упрощенный подход
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        
        # Создаем "фейковое" правое изображение (сдвинутое)
        h, w = gray.shape
        right_img = np.roll(gray, shift=-20, axis=1)
        
        # Вычисляем карту глубины
        disparity = stereo.compute(gray, right_img)
        depth_map = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX)
        
        self.results['Stereo'] = {
            'depth_map': depth_map,
            'time': time() - start_time
        }
    
    def laplacian_depth(self):
        """Метод на основе лапласиана (для текстур)"""
        start_time = time()
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        depth_map = cv2.normalize(np.abs(laplacian), None, 0, 1, cv2.NORM_MINMAX)
        
        self.results['Laplacian'] = {
            'depth_map': depth_map,
            'time': time() - start_time
        }
    
    def run_all(self):
        """Запуск всех методов"""
        self.geometric_depth()
        #self.midas_depth()
        self.stereo_depth()
        self.laplacian_depth()
    
    def visualize_results(self, output_dir='depth_comparison'):
        """Визуализация и сохранение результатов"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(20, 10))
        
        # Сохраняем оригинальное изображение
        plt.subplot(2, 3, 1)
        plt.imshow(self.image)
        plt.title('Original Image')
        plt.axis('off')
        plt.savefig(f'{output_dir}/original.jpg', bbox_inches='tight')
        
        # Визуализируем каждый метод
        for i, (method, data) in enumerate(self.results.items(), 2):
            depth_map = data['depth_map']
            time_taken = data['time']
            
            # Применяем цветовую карту
            depth_colormap = cv2.applyColorMap(
                np.uint8(depth_map * 255), 
                cv2.COLORMAP_JET
            )
            depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            
            # Наложение на исходное изображение
            overlay = cv2.addWeighted(
                self.image, 0.7, 
                depth_colormap, 0.3, 0
            )
            
            # Сохраняем результаты
            plt.subplot(2, 3, i)
            plt.imshow(overlay)
            plt.title(f'{method}\nTime: {time_taken:.2f}s')
            plt.axis('off')
            
            cv2.imwrite(f'{output_dir}/{method}_depth.jpg', cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{output_dir}/{method}_overlay.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison.jpg', dpi=300, bbox_inches='tight')
        plt.close()

# Пример использования
if __name__ == '__main__':
    path_to_image = "/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/test/d43b5a98-00000000.jpg"  # Замените на свой путь
    comparator = DepthMapComparator(path_to_image)
    comparator.run_all()
    comparator.visualize_results()
import cv2
import numpy as np
import torch
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

def load_efficientvit_sam(model_name="xl0", device="cuda"):
    """Загрузка EfficientViT-SAM"""
    model = create_sam_model(name=model_name, weight_url="auto").to(device)
    predictor = EfficientViTSamPredictor(model)
    return predictor

def segment_road(image_path, output_path="road_mask.png"):
    # Загрузка изображения
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    predictor = load_efficientvit_sam()

    # Генерация масок (автоматический режим)
    predictor.set_image(image)
    masks, _, _ = predictor.predict()  # Без промптов — сегментация всего

    # Фильтрация маски дороги (эвристика: нижняя часть + большая площадь)
    h, w = image.shape[:2]
    road_mask = None
    for mask in masks:
        y_coords = np.where(mask)[0]
        if y_coords.mean() > h * 0.6 and mask.sum() > w * h * 0.3:  # Дорога обычно внизу
            road_mask = mask
            break

    if road_mask is None:
        print("Дорога не найдена. Попробуйте ручные промпты (bbox/points).")
        return

    # Визуализация
    overlay = image.copy()
    overlay[road_mask] = [255, 0, 0]  # Красный цвет для маски
    blended = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    # Сохранение результата
    cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Результат сохранён в {output_path}")

    # Отображение
    cv2.imshow("Road Segmentation", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

if __name__ == "__main__":
    image_path = "/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/test/d6a288b4-5ec72a0e.jpg"  # Замените на свой путь
    segment_road(image_path)  # Укажите путь к изображению
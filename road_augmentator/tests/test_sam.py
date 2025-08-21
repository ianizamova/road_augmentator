import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def load_sam_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    
    # Загрузка предобученной модели (версия vit_b, vit_l или vit_h)
    model_type = "vit_b"
    checkpoint_path = "/home/irina/work/otus_cv/blackswan_generator/sam/checkpoints/sam_vit_b_01ec64.pth"  # Скачайте веса с https://github.com/facebookresearch/segment-anything#model-checkpoints
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam

def segment_road(image_path, output_path="output_masked.png"):
    # Загрузка изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Инициализация SAM
    sam = load_sam_model()
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Генерация масок
    masks = mask_generator.generate(image)

    # Фильтрация масок для поиска проезжей части (эвристика)
    road_mask = None
    for idx_mask,mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        # Эвристика: выбираем маску с большой площадью и низким отношением высоты к ширине
        # y, x = np.where(mask)
        # h, w = y.max() - y.min(), x.max() - x.min()
        # aspect_ratio = h / w
        # if aspect_ratio < 0.5 and mask.sum() > 10000:  # Подберите параметры под ваш случай
        #     road_mask = mask
        #     break

    # if road_mask is None:
    #     print("Не удалось найти проезжую часть.")
    #     return

        # Наложение маски на изображение
        overlay = image.copy()
        #overlay[mask] = (max(255-2*idx_mask, 0), max(0, 128 - 4*idx_mask), min(255, 10*idx_mask))  # Красный цвет для маски
        overlay[mask] = (0, 255, 0)
        alpha = 0.4  # Прозрачность наложения
        masked_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Сохранение результата
        masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        output_path = f"masked_img_{str(idx_mask)}.png"
        cv2.imwrite(output_path, masked_image_bgr)
        print(f"Результат сохранён в {output_path}")

    # # Визуализация
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title("Исходное изображение")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(masked_image)
    # plt.title("Маска проезжей части")
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    image_path = "/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/test/d6a288b4-5ec72a0e.jpg"  # Замените на свой путь
    segment_road(image_path)  # Укажите путь к вашему изображению
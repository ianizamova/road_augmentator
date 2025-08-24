import cv2
import numpy as np

def load_image(image_path, with_alpha=False):
    """Загрузка изображения с возможностью загрузки альфа-канала"""
    if with_alpha:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image_path, image):
    """Сохранение изображения"""
    #if len(image.shape) == 3 and image.shape[2] == 4:
    #    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    #else:
    #    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)
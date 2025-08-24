import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

class ImageEnhancer:
    def __init__(self):
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.pipeline = None
        
    def load_model(self):
        """Загрузка модели Stable Diffusion для улучшения качества"""
        if self.pipeline is None:
            model_id = "stabilityai/stable-diffusion-2-1-base"
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
    def enhance_image(self, input_image, prompt="high quality, sharp details, realistic, 4k, color alignment", strength=0.3, guidance_scale=7.5):
        """
        Улучшение качества изображения с помощью Stable Diffusion
        
        :param input_image: входное изображение (numpy array)
        :param prompt: текст-описание желаемого результата
        :param strength: сила воздействия (0-1)
        :param guidance_scale: уровень соответствия prompt
        :return: улучшенное изображение (numpy array)
        """

        self.load_model()
        
        # Конвертация в PIL Image
        init_image = Image.fromarray(input_image)
        
        # Улучшение качества
        with torch.no_grad():
            result = self.pipeline(
                prompt="high quality, sharp details, realistic, 4k, color alignment",
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale
            ).images[0]
        
        return np.array(result)
    
class ObjectInserter:
        
    def __init__(self):
        self.enhancer = ImageEnhancer()
        
    def insert_and_enhance(self, background_path, object_path, x, y):
        """Основной метод вставки и улучшения"""
        # Вставка объекта (из предыдущего кода)
        result = self._basic_insert(background_path, object_path, x, y)
        
        # Улучшение качества всей сцены
        enhanced = self.enhancer.enhance_image(result)
        
        return enhanced
    
    def _basic_insert(self, background_path, object_path, x, y):
        """Базовая вставка объекта"""
        bg = cv2.imread(background_path)
        obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)
        
        # Альфа-смешение
        if obj.shape[2] == 4:
            alpha = obj[:,:,3] / 255.0
            for c in range(3):
                bg[y:y+obj.shape[0], x:x+obj.shape[1], c] = \
                    alpha * obj[:,:,c] + (1-alpha) * bg[y:y+obj.shape[0], x:x+obj.shape[1], c]
        else:
            bg[y:y+obj.shape[0], x:x+obj.shape[1]] = obj
            
        return cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

# Пример использования
if __name__ == "__main__":
    inserter = ObjectInserter()
    
    # Вставка и улучшение
    # result = inserter.insert_and_enhance(
    #     background_path="background.jpg",
    #     object_path="object.png",
    #     x=300,
    #     y=200
    # )
     # Вставляем объект
    result = inserter.insert_and_enhance(
        background_path='/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val/7d15b18b-1e0d6e3f.jpg',
        object_path='/home/irina/work/otus_cv/blackswan_generator/extracted_objects/horse_100599_0.png',
        x=300, y=400,
    #    enhance=True
    )
    
    # Сохранение результата
    Image.fromarray(result).save("enhanced_result.jpg")
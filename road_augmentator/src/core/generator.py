# from diffusers import StableDiffusionPipeline
# import torch
# from PIL import Image

# # Инициализация пайплайна
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
# ).to("cuda" if torch.cuda.is_available() else "cpu")

# # Генерация объекта (например, велосипеда)
# prompt = "A bicycle isolated on white background, high detail, 4k"
# negative_prompt = "low quality, blurry, text, watermark"

# generated_image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     width=512,
#     height=512,
#     num_inference_steps=30
# ).images[0]

# generated_image.save("generated_bicycle.png")

import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import cv2
import torch

class ObjectGenerator:
    def __init__(self, config_path=None):
        """
        Инициализация генератора изображений с сегментацией
        
        Args:
            config_path (str): Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self.pipe = None
        self.segmentation_processor = None
        self.segmentation_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_config(self, config_path):
        """Загрузка конфигурации из файла или использование значений по умолчанию"""
        default_config = {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "segmentation_model": "nvidia/segformer-b0-finetuned-ade-512-512",
            "output_dir": "generated_objects",
            "configs_dir": "configs",
            "default_prompt": "A bicycle isolated on white background, high detail, 4k",
            "default_negative_prompt": "low quality, blurry, text, watermark, people, multiple objects",
            "width": 512,
            "height": 512,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": None,
            "save_original": True,
            "save_segmented": True,
            "target_classes": ["bicycle", "motorcycle", "car"],
            "confidence_threshold": 0.5,
            "use_fast_processor": True,
            "enable_safety_checker": True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"Конфигурация загружена из {config_path}")
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}. Использую значения по умолчанию")
        
        return default_config
    
    def save_config(self, config_path=None):
        """Сохранение текущей конфигурации в файл"""
        if config_path is None:
            # Создаем директорию configs если ее нет
            os.makedirs(self.config["configs_dir"], exist_ok=True)
            config_path = os.path.join(self.config["configs_dir"], "generator_config.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"Конфигурация сохранена в {config_path}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            return False
    
    def initialize_models(self):
        """Инициализация моделей генерации и сегментации"""
        try:
            # Инициализация модели генерации
            print("Инициализация Stable Diffusion...")
            
            # Параметры для пайплайна
            pipe_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # Добавляем safety_checker только если включен
            if not self.config["enable_safety_checker"]:
                pipe_kwargs["safety_checker"] = None
                print("Safety checker отключен")
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config["model_name"],
                **pipe_kwargs
            ).to(self.device)
            
            # Инициализация модели сегментации (рабочая модель SegFormer)
            print("Инициализация модели сегментации...")
            self.segmentation_processor = AutoImageProcessor.from_pretrained(
                self.config["segmentation_model"],
                use_fast=self.config["use_fast_processor"]
            )
            self.segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(
                self.config["segmentation_model"]
            ).to(self.device)
            
            # Создаем выходную директорию
            os.makedirs(self.config["output_dir"], exist_ok=True)
            
            print("Модели успешно инициализированы")
            return True
            
        except Exception as e:
            print(f"Ошибка инициализации моделей: {e}")
            return False
    
    def generate_image(self, prompt=None, negative_prompt=None, **kwargs):
        """Генерация изображения по промпту"""
        if not self.pipe:
            print("Модель не инициализирована")
            return None
        
        # Используем промпты из конфига, если не указаны
        prompt = prompt or self.config["default_prompt"]
        negative_prompt = negative_prompt or self.config["default_negative_prompt"]
        
        # Обновляем параметры из kwargs
        generation_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": kwargs.get("width", self.config["width"]),
            "height": kwargs.get("height", self.config["height"]),
            "num_inference_steps": kwargs.get("num_inference_steps", self.config["num_inference_steps"]),
            "guidance_scale": kwargs.get("guidance_scale", self.config["guidance_scale"]),
        }
        
        # Устанавливаем seed если указан
        seed = kwargs.get("seed", self.config["seed"])
        if seed is not None:
            torch.manual_seed(seed)
        
        print(f"Генерация изображения: {prompt}")
        result = self.pipe(**generation_params)
        return result.images[0]
    
    def segment_image(self, image):
        """Сегментация изображения"""
        if not self.segmentation_model:
            print("Модель сегментации не инициализирована")
            return None
        
        try:
            # Преобразуем PIL Image в numpy array
            image_np = np.array(image)
            
            # Подготовка изображения для модели
            inputs = self.segmentation_processor(images=image_np, return_tensors="pt").to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.segmentation_model(**inputs)
            
            # Получаем семантическую сегментацию
            segmentation = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
            
            return segmentation
            
        except Exception as e:
            print(f"Ошибка сегментации: {e}")
            return None
    
    def extract_object_with_transparency(self, image, segmentation_mask, target_class):
        """Извлечение объекта с прозрачным фоном"""
        try:
            # Получаем ID класса для сегментации
            class_id = self._get_class_id(target_class)
            if class_id is None:
                print(f"Класс '{target_class}' не найден в модели сегментации")
                return None
            
            # Создаем маску для целевого класса
            object_mask = (segmentation_mask == class_id).astype(np.uint8)
            
            if np.sum(object_mask) == 0:
                print(f"Объект класса '{target_class}' не найден на изображении")
                return None
            
            # Улучшаем маску с помощью морфологических операций
            kernel = np.ones((5, 5), np.uint8)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
            object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
            
            # Создаем изображение с прозрачным фоном
            image_rgba = image.convert("RGBA")
            image_array = np.array(image_rgba)
            
            # Создаем альфа-канал
            alpha_channel = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
            alpha_channel[object_mask == 1] = 255
            
            # Размываем границы для сглаживания
            alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
            
            # Обновляем альфа-канал
            image_array[:, :, 3] = alpha_channel
            
            # Находим bounding box объекта
            y_indices, x_indices = np.where(object_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return None
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Добавляем отступы
            padding = 15
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image_array.shape[1], x_max + padding)
            y_max = min(image_array.shape[0], y_max + padding)
            
            # Обрезаем изображение
            cropped_image = image_array[y_min:y_max, x_min:x_max]
            
            return Image.fromarray(cropped_image, 'RGBA')
            
        except Exception as e:
            print(f"Ошибка извлечения объекта: {e}")
            return None
    
    def _get_class_id(self, class_name):
        """Получение ID класса из модели сегментации ADE20K"""
        # ADE20K classes mapping (SegFormer)
        class_mapping = {
            "bicycle": 116,  # bicycle
            "motorcycle": 117,  # motorcycle, motorbike
            "car": 114,  # car, auto, automobile, machine, motorcar
            "person": 12,  # person, individual, someone, somebody, mortal, soul
            "bus": 115,  # bus, autobus, coach, charabanc, double-decker
            "truck": 118,  # truck, motortruck
            "vehicle": 114,  # общее для транспортных средств
        }
        
        class_name_lower = class_name.lower()
        for key, value in class_mapping.items():
            if key in class_name_lower:
                return value
        
        # Если точного совпадения нет, пробуем частичные совпадения
        if "bike" in class_name_lower or "cycle" in class_name_lower:
            return 116
        elif "moto" in class_name_lower:
            return 117
        elif "vehicle" in class_name_lower or "transport" in class_name_lower:
            return 114
        
        return None
    
    def generate_and_extract_object(self, prompt=None, negative_prompt=None, class_name=None, **kwargs):
        """Полный процесс: генерация + сегментация + извлечение"""
        # Генерация изображения
        generated_image = self.generate_image(prompt, negative_prompt, **kwargs)
        if generated_image is None:
            return None
        
        # Определяем класс объекта из промпта если не указан
        if class_name is None:
            class_name = self._extract_class_from_prompt(
                prompt or self.config["default_prompt"]
            )
        
        # Сегментация
        segmentation_mask = self.segment_image(generated_image)
        if segmentation_mask is None:
            return None
        
        # Извлечение объекта с прозрачным фоном
        extracted_object = self.extract_object_with_transparency(
            generated_image, segmentation_mask, class_name
        )
        
        # Сохранение результатов
        base_filename = class_name.lower().replace(" ", "_")
        timestamp = str(int(torch.randint(1000, 9999, (1,)).item()))
        
        if self.config["save_original"] and generated_image:
            original_path = os.path.join(
                self.config["output_dir"], 
                f"{base_filename}_original_{timestamp}.png"
            )
            generated_image.save(original_path)
            print(f"Оригинальное изображение сохранено: {original_path}")
        
        if self.config["save_segmented"] and extracted_object:
            segmented_path = os.path.join(
                self.config["output_dir"], 
                f"{base_filename}_transparent_{timestamp}.png"
            )
            extracted_object.save(segmented_path, "PNG")
            print(f"Объект с прозрачным фоном сохранен: {segmented_path}")
            print(f"Размер: {extracted_object.size}")
        
        return {
            "original": generated_image,
            "segmented": extracted_object,
            "class_name": class_name
        }
    
    def _extract_class_from_prompt(self, prompt):
        """Извлечение названия класса из промпта"""
        prompt_lower = prompt.lower()
        for target_class in self.config["target_classes"]:
            if target_class.lower() in prompt_lower:
                return target_class
        return "object"
    
    def batch_generate(self, prompts_list):
        """Пакетная генерация для нескольких промптов"""
        results = []
        for i, prompt_config in enumerate(prompts_list):
            try:
                if isinstance(prompt_config, str):
                    result = self.generate_and_extract_object(prompt=prompt_config)
                elif isinstance(prompt_config, dict):
                    result = self.generate_and_extract_object(**prompt_config)
                else:
                    print(f"Неверный формат промпта {i}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                print(f"Ошибка при обработке промпта {i}: {e}")
                continue
        
        return results

# Пример использования
if __name__ == "__main__":
    # Создаем генератор
    generator = ObjectGenerator()
    
    # Инициализируем модели
    if generator.initialize_models():
        # Сохраняем конфигурацию
        generator.save_config()
        
        # Генерация одного объекта
        result = generator.generate_and_extract_object(
            prompt="A red bicycle isolated on white background, professional photo",
            class_name="bicycle"
        )
        
        if result and result["segmented"]:
            print("Успешно сгенерирован объект с прозрачным фоном!")
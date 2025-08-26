# import cv2
# import numpy as np
# import torch
# from diffusers import StableDiffusionImg2ImgPipeline
# from PIL import Image

# class ImageEnhancer:
#     def __init__(self):
#         #self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device = "cpu"
#         self.pipeline = None
        
#     def load_model(self):
#         """Загрузка модели Stable Diffusion для улучшения качества"""
#         if self.pipeline is None:
#             model_id = "stabilityai/stable-diffusion-2-1-base"
#             self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
#                 model_id,
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
#             ).to(self.device)
            
#     def enhance_image(self, input_image, prompt="high quality, sharp details, realistic, 4k, color alignment", strength=0.3, guidance_scale=7.5):
#         """
#         Улучшение качества изображения с помощью Stable Diffusion
        
#         :param input_image: входное изображение (numpy array)
#         :param prompt: текст-описание желаемого результата
#         :param strength: сила воздействия (0-1)
#         :param guidance_scale: уровень соответствия prompt
#         :return: улучшенное изображение (numpy array)
#         """

#         self.load_model()
        
#         # Конвертация в PIL Image
#         init_image = Image.fromarray(input_image)
        
#         # Улучшение качества
#         with torch.no_grad():
#             result = self.pipeline(
#                 prompt="high quality, sharp details, realistic, 4k, color alignment",
#                 image=init_image,
#                 strength=strength,
#                 guidance_scale=guidance_scale
#             ).images[0]
        
#         return np.array(result)

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import time
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging

from src.utils.config_loader import ConfigLoader

@dataclass
class EnhancementResult:
    image: np.ndarray
    metadata: Dict[str, Any]
    processing_time: float

class ImageEnhancer:
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация улучшателя изображений
        
        Args:
            config_path: Путь к JSON конфигу. Если None, используется дефолтный конфиг.
        """
        # Загрузка конфигурации
        self.config = ConfigLoader.get_enhancer_config(config_path)
        
        self.device = self._setup_device()
        self.pipeline = None
        self.inpainting_pipeline = None
        self.logger = self._setup_logger()
        
    def _setup_device(self) -> str:
        """Настройка устройства выполнения"""
        perf_config = self.config["performance_settings"]
        default_config = self.config["default_parameters"]
        
        if perf_config["device"] == "auto":
            if default_config["auto_detect_device"] and torch.cuda.is_available():
                return "cuda"
            elif default_config["fallback_to_cpu"]:
                return "cpu"
            else:
                raise RuntimeError("CUDA not available and fallback to CPU disabled")
        else:
            return perf_config["device"]
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            if self.config["default_parameters"]["verbose"]:
                logger.setLevel(logging.INFO)
            else:
                logger.setLevel(logging.WARNING)
        
        return logger
    
    def load_model(self):
        """Загрузка модели улучшения качества"""
        if self.pipeline is None:
            model_config = self.config["model_settings"]
            perf_config = self.config["performance_settings"]
            
            self.logger.info(f"Loading enhancement model: {model_config['model_id']}")
            
            # Определяем тип данных
            if model_config["torch_dtype"] == "float16" and self.device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            # Загрузка основной модели
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_config["model_id"],
                #revision=model_config["revision"],
                torch_dtype=torch_dtype,
                #cache_dir=model_config["cache_dir"],
                local_files_only=model_config["local_files_only"],
                safety_checker=None,
                requires_safety_checker=model_config["requires_safety_checker"]
            ).to(self.device)
            
            # Применяем оптимизации производительности
            self._apply_performance_optimizations()
            
            self.logger.info("Enhancement model loaded successfully")
    
    def load_inpainting_model(self):
        """Загрузка модели для inpainting"""
        if self.inpainting_pipeline is None and self.config["inpainting_settings"]["use_inpainting"]:
            inpainting_config = self.config["inpainting_settings"]
            
            self.logger.info(f"Loading inpainting model: {inpainting_config['inpainting_model']}")
            
            self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                inpainting_config["inpainting_model"],
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                #cache_dir=self.config["model_settings"]["cache_dir"]
            ).to(self.device)
            
            self.logger.info("Inpainting model loaded successfully")
    
    def _apply_performance_optimizations(self):
        """Применение оптимизаций производительности"""
        perf_config = self.config["performance_settings"]
        
        if perf_config["enable_attention_slicing"]:
            self.pipeline.enable_attention_slicing()
        
        if perf_config["enable_xformers"] and torch.cuda.is_available():
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except ImportError:
                self.logger.warning("xformers not available, skipping optimization")
        
        if perf_config["enable_vae_slicing"]:
            self.pipeline.enable_vae_slicing()
        
        if perf_config["enable_vae_tiling"]:
            self.pipeline.enable_vae_tiling()
    
    def enhance_image(self, input_image: np.ndarray, 
                     prompt: Optional[str] = None,
                     negative_prompt: Optional[str] = None,
                     strength: Optional[float] = None,
                     guidance_scale: Optional[float] = None,
                     num_inference_steps: Optional[int] = None,
                     seed: Optional[int] = None,
                     object_type: Optional[str] = None,
                     scene_type: Optional[str] = None,
                     mask: Optional[np.ndarray] = None) -> EnhancementResult:
        """
        Улучшение качества изображения
        
        Args:
            input_image: Входное изображение (numpy array BGR или RGB)
            prompt: Кастомный промпт (опционально)
            negative_prompt: Кастомный негативный промпт (опционально)
            strength: Сила улучшения (0-1)
            guidance_scale: Уровень соответствия промпту
            num_inference_steps: Количество шагов инференса
            seed: Seed для воспроизводимости
            object_type: Тип объекта для специфичного промпта
            scene_type: Тип сцены для специфичного промпта
            mask: Маска для inpainting (опционально)
        
        Returns:
            EnhancementResult с улучшенным изображением и метаданными
        """
        start_time = time.time()
        
        # Загружаем модель если нужно
        self.load_model()
        
        # Получаем параметры из конфига или аргументов
        params = self._get_enhancement_parameters(
            prompt, negative_prompt, strength, guidance_scale, 
            num_inference_steps, object_type, scene_type
        )
        
        # Конвертация в PIL Image
        # if len(input_image.shape) == 3 and input_image.shape[2] == 3:
        #     # Конвертируем BGR to RGB если нужно
        #     pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        # else:
        #     pil_image = Image.fromarray(input_image)
        pil_image = Image.fromarray(input_image)
        
        # Ресайз если нужно
        max_size = self.config["default_parameters"]["max_size"]
        if max(pil_image.size) > max_size:
            pil_image = self._resize_image(pil_image, max_size)
        
        # Установка seed для воспроизводимости
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        elif self.config["default_parameters"]["seed"] is not None:
            generator = torch.Generator(device=self.device).manual_seed(
                self.config["default_parameters"]["seed"]
            )
        
        # Выполнение улучшения
        with torch.no_grad():
            result = self.pipeline(
                prompt=params["prompt"],
                negative_prompt=params["negative_prompt"],
                image=pil_image,
                strength=params["strength"],
                guidance_scale=params["guidance_scale"],
                num_inference_steps=params["num_inference_steps"],
                generator=generator,
                eta=params.get("eta", 0.0)
            ).images[0]
        
        # Дополнительный inpainting если нужно и есть маска
        if mask is not None and self.config["inpainting_settings"]["use_inpainting"]:
            result = self._apply_inpainting(result, mask, params)
        
        # Конвертация обратно в numpy array
        enhanced_np = np.array(result)
        #if enhanced_np.shape[2] == 3:  # RGB to BGR
        #    enhanced_np = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
        
        processing_time = time.time() - start_time
        
        # Сбор метаданных
        metadata = {
            "parameters": params,
            "device": self.device,
            "original_size": input_image.shape[:2],
            "enhanced_size": enhanced_np.shape[:2],
            "processing_time_seconds": processing_time,
            "seed_used": seed if seed else self.config["default_parameters"]["seed"]
        }
        
        self.logger.info(f"Image enhanced in {processing_time:.2f} seconds")
        
        return EnhancementResult(image=enhanced_np, metadata=metadata, processing_time=processing_time)
    
    def _get_enhancement_parameters(self, prompt: Optional[str], 
                                   negative_prompt: Optional[str],
                                   strength: Optional[float],
                                   guidance_scale: Optional[float],
                                   num_inference_steps: Optional[int],
                                   object_type: Optional[str],
                                   scene_type: Optional[str]) -> Dict[str, Any]:
        """Получение параметров улучшения с учетом приоритетов"""
        # Базовые параметры из пресета
        preset_name = self.config["default_parameters"]["preset"]
        params = self.config["enhancement_presets"][preset_name].copy()
        
        # Обогащаем промпт если указаны тип объекта или сцены
        final_prompt = prompt or params["prompt"]
        if object_type or scene_type:
            final_prompt = self._enrich_prompt(final_prompt, object_type, scene_type)
        
        # Переопределяем параметры если переданы явно
        if prompt is not None:
            params["prompt"] = prompt
        else:
            params["prompt"] = final_prompt
        
        if negative_prompt is not None:
            params["negative_prompt"] = negative_prompt
        
        if strength is not None:
            params["strength"] = strength
        
        if guidance_scale is not None:
            params["guidance_scale"] = guidance_scale
        
        if num_inference_steps is not None:
            params["num_inference_steps"] = num_inference_steps
        
        return params
    
    def _enrich_prompt(self, base_prompt: str, 
                      object_type: Optional[str], 
                      scene_type: Optional[str]) -> str:
        """Обогащение промпта специфичными шаблонами"""
        prompt_templates = self.config["prompt_templates"]
        enriched_prompt = base_prompt
        
        if object_type and object_type.lower() in prompt_templates["object_specific"]:
            object_prompt = prompt_templates["object_specific"][object_type.lower()]
            enriched_prompt = f"{base_prompt}, {object_prompt}"
        
        if scene_type and scene_type.lower() in prompt_templates["scene_specific"]:
            scene_prompt = prompt_templates["scene_specific"][scene_type.lower()]
            enriched_prompt = f"{enriched_prompt}, {scene_prompt}"
        
        return enriched_prompt
    
    def _resize_image(self, image: Image.Image, max_size: int) -> Image.Image:
        """Ресайз изображения с сохранением пропорций"""
        resize_mode = self.config["default_parameters"]["resize_mode"]
        width, height = image.size
        
        if max(width, height) <= max_size:
            return image
        
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        if resize_mode == "crop_center":
            # Ресайз + кроп центра
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            # Здесь можно добавить логику кропа если нужно
            return resized
        else:
            # Простой ресайз
            return image.resize((new_width, new_height), Image.LANCZOS)
    
    def _apply_inpainting(self, image: Image.Image, 
                         mask: np.ndarray, 
                         params: Dict[str, Any]) -> Image.Image:
        """Применение inpainting к изображению"""
        self.load_inpainting_model()
        
        if self.inpainting_pipeline is None:
            return image
        
        # Подготовка маски
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        inpainting_config = self.config["inpainting_settings"]
        
        # Дилатация и блюр маски
        if inpainting_config["mask_dilation"] > 0:
            mask_np = np.array(mask_pil)
            kernel = np.ones((inpainting_config["mask_dilation"], 
                             inpainting_config["mask_dilation"]), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            if inpainting_config["mask_blur"] > 0:
                mask_np = cv2.GaussianBlur(mask_np, 
                                          (inpainting_config["mask_blur"], 
                                           inpainting_config["mask_blur"]), 0)
            mask_pil = Image.fromarray(mask_np)
        
        # Применение inpainting
        result = self.inpainting_pipeline(
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            image=image,
            mask_image=mask_pil,
            strength=inpainting_config["inpainting_strength"],
            guidance_scale=inpainting_config["inpainting_guidance_scale"],
            num_inference_steps=params["num_inference_steps"]
        ).images[0]
        
        return result
    
    def save_result(self, result: EnhancementResult, original_path: Optional[str] = None):
        """Сохранение результата улучшения"""
        output_config = self.config["output_settings"]
        os.makedirs(output_config["output_directory"], exist_ok=True)
        
        # Генерация имени файла
        if original_path:
            original_name = os.path.splitext(os.path.basename(original_path))[0]
        else:
            original_name = "image"
        
        timestamp = int(time.time())
        filename = output_config["filename_pattern"].format(
            original_name=original_name,
            timestamp=timestamp,
            format=output_config["output_format"]
        )
        
        output_path = os.path.join(output_config["output_directory"], filename)
         # Конвертация из BGR в RGB если нужно
        # if len(result.image.shape) == 3 and result.image.shape[2] == 3:
        #     # Проверяем, является ли изображение BGR (обычный выход из OpenCV)
        #     # Конвертируем BGR to RGB
        #     image_to_save = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
        # else:
        #     # Если уже RGB или grayscale, используем как есть
        #image_to_save = result.image
        image_to_save = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
            
        # Сохранение изображения
        if output_config["output_format"].lower() == "png":
            pil_image = Image.fromarray(image_to_save)
            pil_image.save(output_path, 
                        format='PNG', 
                        optimize=True, 
                        quality=output_config["output_quality"])
        else:
            # Для JPEG и других форматов используем OpenCV
            # Но сначала конвертируем обратно в BGR для OpenCV
            #if len(image_to_save.shape) == 3 and image_to_save.shape[2] == 3:
            #    bgr_image = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
            #else:
            #    bgr_image = image_to_save
                
            cv2.imwrite(output_path, image_to_save, 
                    [cv2.IMWRITE_JPEG_QUALITY, output_config["output_quality"]])
        
        # Сохранение метаданных
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(result.metadata, f, indent=2)
        
        self.logger.info(f"Result saved to {output_path}")
        
        return output_path
    
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
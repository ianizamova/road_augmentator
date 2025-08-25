import json
import yaml
from omegaconf import OmegaConf
from typing import Dict, Any, List, Optional
import os

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> OmegaConf:
        """Загрузка конфигурации из YAML файла"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return OmegaConf.create(config)
    
    @staticmethod
    def load_json_config(config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из JSON файла"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def get_object_placer_config(config_path: str = "configs/object_placer_config.json") -> Dict[str, Any]:
        """
        Загрузка конфигурации для ObjectPlacer
        
        Args:
            config_path: Путь к JSON файлу конфигурации
            
        Returns:
            Словарь с конфигурацией ObjectPlacer
        """
        try:
            config = ConfigLoader.load_json_config(config_path)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load ObjectPlacer config from {config_path}: {e}")
            print("Using default ObjectPlacer configuration")
            return ConfigLoader.get_default_object_placer_config()
    
    @staticmethod
    def get_default_object_placer_config() -> Dict[str, Any]:
        """Возвращает дефолтную конфигурацию ObjectPlacer"""
        return {
            "model_settings": {
                "detection_model": "facebook/detr-resnet-50",
                "segmentation_model": "facebook/maskformer-swin-large-coco",
                "depth_model": "Intel/dpt-large",
                "device": "cuda"
            },
            "reference_sizes": {
                'car': 200,
                'person': 80,
                'bicycle': 120,
                'vase': 40,
                'horse': 300,
                'bird': 50,
                'elephant': 400,
                'giraffe': 500
            },
            "object_surface_compatibility": {
                'car': ['road', 'parking lot', 'street'],
                'vase': ['table', 'shelf', 'desk'],
                'person': ['sidewalk', 'floor', 'grass', 'pavement-merged'],
                'bicycle': ['road', 'sidewalk', 'path'],
                'horse': ['road', 'path', 'street', 'driving-area'],
                'elephant': ['road', 'path', 'street', 'driving-area'],
                'giraffe': ['road', 'path', 'street', 'driving-area'],
                'bird': ['sky']
            }
        }
        
    @staticmethod
    def get_enhancer_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Загрузка конфигурации для ImageEnhancer
        
        Args:
            config_path: Путь к JSON файлу конфигурации
            
        Returns:
            Словарь с конфигурацией ImageEnhancer
        """
        if config_path is None:
            config_path = "configs/enhancer_config.json"
        
        try:
            config = ConfigLoader.load_json_config(config_path)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load Enhancer config from {config_path}: {e}")
            print("Using default Enhancer configuration")
            return ConfigLoader.get_default_enhancer_config()
    
    @staticmethod
    def get_default_enhancer_config() -> Dict[str, Any]:
        """Возвращает дефолтную конфигурацию ImageEnhancer"""
        return {
            "model_settings": {
                "model_id": "stabilityai/stable-diffusion-2-1-base",
                "torch_dtype": "float32",
                "cache_dir": "./models_cache"
            },
            "enhancement_presets": {
                "default": {
                    "prompt": "high quality, sharp details, realistic, 4k",
                    "negative_prompt": "blurry, low quality, distorted",
                    "strength": 0.3,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 20
                }
            },
            "performance_settings": {
                "device": "auto",
                "enable_attention_slicing": True
            },
            "default_parameters": {
                "preset": "default",
                "max_size": 1024
            }
        }
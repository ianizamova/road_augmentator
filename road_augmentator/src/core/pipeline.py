import json
import os
from typing import Dict, Any, List, Tuple
from PIL import Image
import glob

from generator import ObjectGenerator
from object_extractor import ObjectExtractor
from rider_extractor import BicycleRiderExtractor
from inserter import ObjectInserter
from position_predictor import ObjectPlacer
from enhancer import ImageEnhancer

def load_config(config_path: str) -> Dict[str, Any]:
    """Загрузка конфигурации из JSON файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Конфигурация загружена из {config_path}")
        return config
    except FileNotFoundError:
        print(f"Конфигурационный файл {config_path} не найден")
        return {}
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON в файле {config_path}: {e}")
        return {}
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return {}

class AugmentationPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.modules = {}
        
        # Инициализация модулей
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Инициализация всех модулей на основе конфигурации"""
        # Object Extractor
        if self.config.get('object_extractor', {}).get('enabled', False):
            extractor_config_path = self.config['object_extractor'].get('config_path')
            if extractor_config_path:
                extractor_config = load_config(extractor_config_path)
                self.object_extractor = ObjectExtractor(extractor_config)
                self.modules['object_extractor'] = self.object_extractor
                print("ObjectExtractor инициализирован")
        
        # Rider Extractor
        if self.config.get('rider_extractor', {}).get('enabled', False):
            rider_config_path = self.config['rider_extractor'].get('config_path')
            if rider_config_path:
                rider_config = load_config(rider_config_path)
                self.rider_extractor = BicycleRiderExtractor(rider_config)
                self.modules['rider_extractor'] = self.rider_extractor
                print("BicycleRiderExtractor инициализирован")
        
        # Object Generator
        if self.config.get('generator', {}).get('enabled', False):
            generator_config_path = self.config['generator'].get('config_path')
            if generator_config_path:
                generator_config = load_config(generator_config_path)
                self.generator = ObjectGenerator(generator_config)
                self.modules['generator'] = self.generator
                print("ObjectGenerator инициализирован")
        
        # Обязательные модули
        predictor_config_path = self.config['position_predictor'].get('config_path')
        if predictor_config_path:
            predictor_config = load_config(predictor_config_path)
            self.position_predictor = ObjectPlacer(predictor_config)
            self.modules['position_predictor'] = self.position_predictor
            print("ObjectPlacer инициализирован")
        
        inserter_config_path = self.config['inserter'].get('config_path')
        if inserter_config_path:
            inserter_config = load_config(inserter_config_path)
            self.inserter = ObjectInserter(inserter_config)
            self.modules['inserter'] = self.inserter
            print("ObjectInserter инициализирован")
        
        enhancer_config_path = self.config['enhancer'].get('config_path')
        if enhancer_config_path:
            enhancer_config = load_config(enhancer_config_path)
            self.enhancer = ImageEnhancer(enhancer_config)
            self.modules['enhancer'] = self.enhancer
            print("ImageEnhancer инициализирован")
    
    def _get_object_sources(self) -> List[str]:
        """Получение путей к источникам объектов"""
        sources_config = self.config.get('object_sources', {})
        sources = []
        
        # Директории с изображениями объектов
        directories = sources_config.get('directories', [])
        for directory in directories:
            if os.path.exists(directory):
                sources.extend(glob.glob(os.path.join(directory, '*.png')) +
                              glob.glob(os.path.join(directory, '*.jpg')) +
                              glob.glob(os.path.join(directory, '*.jpeg')))
        
        # Отдельные файлы
        files = sources_config.get('files', [])
        sources.extend([f for f in files if os.path.exists(f)])
        
        return list(set(sources))  # Убираем дубликаты
    
    def _get_background_sources(self) -> List[str]:
        """Получение путей к фоновым изображениям"""
        backgrounds_config = self.config.get('background_sources', {})
        sources = []
        
        # Директории с фонами
        directories = backgrounds_config.get('directories', [])
        for directory in directories:
            if os.path.exists(directory):
                sources.extend(glob.glob(os.path.join(directory, '*.png')) +
                              glob.glob(os.path.join(directory, '*.jpg')) +
                              glob.glob(os.path.join(directory, '*.jpeg')))
        
        # Отдельные файлы
        files = backgrounds_config.get('files', [])
        sources.extend([f for f in files if os.path.exists(f)])
        
        return list(set(sources))
    
    def extract_objects_batch(self) -> List[Dict]:
        """Пакетное извлечение объектов из всех источников"""
        object_sources = self._get_object_sources()
        all_objects = []
        
        print(f"Найдено {len(object_sources)} источников объектов")
        
        for source in object_sources:
            try:
                # Извлечение объектов
                if 'object_extractor' in self.modules:
                    objects = self.object_extractor.extract_objects(source)
                    all_objects.extend(objects)
                    print(f"Из {source} извлечено {len(objects)} объектов")
                
                # Извлечение велосипедистов
                if 'rider_extractor' in self.modules:
                    riders = self.rider_extractor.extract_riders(source)
                    all_objects.extend(riders)
                    print(f"Из {source} извлечено {len(riders)} велосипедистов")
                    
            except Exception as e:
                print(f"Ошибка при обработке {source}: {e}")
                continue
        
        # Генерация объектов
        if 'generator' in self.modules:
            generation_prompts = self.config['generator'].get('prompts', [])
            generated_objects = self.generator.generate_objects(generation_prompts)
            all_objects.extend(generated_objects)
            print(f"Сгенерировано {len(generated_objects)} объектов")
        
        print(f"Всего объектов: {len(all_objects)}")
        return all_objects
    
    def process_single_background(self, background_path: str, objects: List[Dict]) -> Dict:
        """Обработка одного фонового изображения с несколькими объектами"""
        try:
            background_image = Image.open(background_path)
            print(f"Обработка фона: {os.path.basename(background_path)}")
            
            positioned_objects = []
            
            # Предсказание позиций для каждого объекта
            for obj in objects:
                position = self.position_predictor.predict_position(obj, background_image)
                if position:  # Если найдена валидная позиция
                    positioned_objects.append({
                        'object': obj,
                        'position': position
                    })
            
            if not positioned_objects:
                print("Не найдено подходящих позиций для объектов")
                return None
            
            # Вставка объектов
            augmented_image = background_image.copy()
            for obj_data in positioned_objects:
                augmented_image = self.inserter.insert_object(
                    augmented_image, 
                    obj_data['object'], 
                    obj_data['position']
                )
            
            # Улучшение изображения
            final_image = self.enhancer.enhance(augmented_image)
            
            # Сохранение результата
            output_dir = self.config.get('output_directory', 'results')
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(background_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_augmented.png")
            
            final_image.save(output_path)
            
            return {
                'background_path': background_path,
                'output_path': output_path,
                'objects_placed': len(positioned_objects),
                'total_objects': len(objects)
            }
            
        except Exception as e:
            print(f"Ошибка при обработке фона {background_path}: {e}")
            return None
    
    def process_multiple_backgrounds(self, background_paths: List[str], objects: List[Dict]) -> List[Dict]:
        """Обработка нескольких фоновых изображений"""
        results = []
        
        for background_path in background_paths:
            result = self.process_single_background(background_path, objects)
            if result:
                results.append(result)
        
        return results
    
    def run(self):
        """Основной метод запуска пайплайна"""
        print("Запуск пайплайна аугментации...")
        
        # Шаг 1: Пакетное извлечение/генерация объектов
        print("=== ЭТАП 1: Подготовка объектов ===")
        all_objects = self.extract_objects_batch()
        
        if not all_objects:
            print("Не найдено объектов для обработки")
            return None
        
        # Шаг 2: Получение фоновых изображений
        print("\n=== ЭТАП 2: Подготовка фонов ===")
        background_paths = self._get_background_sources()
        
        if not background_paths:
            print("Не найдено фоновых изображений")
            return None
        
        print(f"Найдено {len(background_paths)} фоновых изображений")
        
        # Шаг 3: Обработка фонов с объектами
        print("\n=== ЭТАП 3: Аугментация ===")
        results = self.process_multiple_backgrounds(background_paths, all_objects)
        
        # Статистика
        total_processed = len(results)
        total_objects_placed = sum(result['objects_placed'] for result in results)
        
        print(f"\n=== РЕЗУЛЬТАТЫ ===")
        print(f"Обработано фонов: {total_processed}/{len(background_paths)}")
        print(f"Всего объектов доступно: {len(all_objects)}")
        print(f"Всего объектов размещено: {total_objects_placed}")
        print(f"Среднее объектов на фон: {total_objects_placed/total_processed if total_processed > 0 else 0:.2f}")
        
        return {
            'results': results,
            'total_backgrounds': len(background_paths),
            'total_objects': len(all_objects),
            'objects_placed': total_objects_placed,
            'success_rate': total_processed / len(background_paths) if background_paths else 0
        }
    
    def run_with_custom_objects(self, custom_objects: List[Dict]):
        """Запуск пайплайна с пользовательскими объектами"""
        print("Запуск пайплайна с пользовательскими объектами...")
        
        background_paths = self._get_background_sources()
        
        if not background_paths:
            print("Не найдено фоновых изображений")
            return None
        
        results = self.process_multiple_backgrounds(background_paths, custom_objects)
        
        return {
            'results': results,
            'total_objects': len(custom_objects)
        }

# Пример главного конфигурационного файла (configs/main_config.json)
"""
{
    "object_sources": {
        "directories": [
            "data/objects/extracted",
            "data/objects/generated"
        ],
        "files": [
            "data/custom_objects/special_bicycle.png"
        ]
    },
    "background_sources": {
        "directories": [
            "data/backgrounds/road_scenes",
            "data/backgrounds/urban"
        ],
        "files": [
            "data/backgrounds/special_scene.jpg"
        ]
    },
    "object_extractor": {
        "enabled": true,
        "config_path": "configs/object_extractor_config.json"
    },
    "rider_extractor": {
        "enabled": true,
        "config_path": "configs/rider_extractor_config.json"
    },
    "generator": {
        "enabled": true,
        "config_path": "configs/generator_config.json",
        "prompts": [
            "A blue bicycle on white background",
            "A red motorcycle isolated on white background",
            "A person riding a bicycle on white background"
        ]
    },
    "position_predictor": {
        "config_path": "configs/position_predictor_config.json"
    },
    "inserter": {
        "config_path": "configs/inserter_config.json"
    },
    "enhancer": {
        "config_path": "configs/enhancer_config.json"
    },
    "output_directory": "results/augmented_images",
    "max_objects_per_background": 5,
    "min_object_confidence": 0.6
}
"""

# Пример использования
if __name__ == "__main__":
    # Инициализация пайплайна
    pipeline = AugmentationPipeline("configs/main_config.json")
    
    # Стандартный запуск
    results = pipeline.run()
    
    # Или запуск с пользовательскими объектами
    # custom_objects = [{'image': Image.open('custom_obj.png'), 'type': 'bicycle'}]
    # results = pipeline.run_with_custom_objects(custom_objects)
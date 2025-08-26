from glob import glob
from tqdm import tqdm
import random
import os
import argparse
import numpy as np

#from src.utils.config_loader import load_config
from src.utils.image_utils import load_image, save_image
from src.core.position_predictor import ObjectPlacer
from src.core.inserter import ObjectInserter
from src.core.enhancer import ImageEnhancer

def main():
    parser = argparse.ArgumentParser(description='Object Placement and Enhancement Pipeline')
    parser.add_argument('--backgrounds_dir', type=str, required=True, default="/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val", help='Path to background images')
    parser.add_argument('--objects_dir', type=str, required=True, default='output_riders', help='Path to foreground objects with transparency')
    parser.add_argument('--output_dir', type=str, required=True, default='experiment1', help='Output directory')
    parser.add_argument('--placer_config', type=str, default='configs/placer_config.yaml')
    parser.add_argument('--inserter_config', type=str, default='configs/inserter_config.yaml')
    parser.add_argument('--enhancer_config', type=str, default='configs/enhancer_config.yaml')
  
    args = parser.parse_args()
    
    # Создаем выходные директории
    os.makedirs(args.output_dir, exist_ok=True)
    blended_dir = os.path.join(args.output_dir, 'blended')
    enhanced_dir = os.path.join(args.output_dir, 'enhanced')
    os.makedirs(blended_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Инициализируем компоненты
    placer = ObjectPlacer(config_path="road_augmentator/configs/object_placer_config.json")
    inserter = ObjectInserter(config_path="road_augmentator/configs/inserter_config.json")
    enhancer = ImageEnhancer(config_path="road_augmentator/configs/enhancer_config.json")
    
    # Получаем список изображений
    background_paths = glob(os.path.join(args.backgrounds_dir, '*.jpg')) + \
                      glob(os.path.join(args.backgrounds_dir, '*.png'))
    object_paths = glob(os.path.join(args.objects_dir, '*.png'))  # Ожидаем PNG с прозрачностью
    
    print(f"Found {len(background_paths)} backgrounds and {len(object_paths)} objects")
    
    # Обрабатываем каждое фоновое изображение
    for bg_path in tqdm(background_paths, desc="Processing backgrounds"):
        try:
            # Загружаем фон
            background = load_image(bg_path)
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
            
            # Предсказываем позиции для фона
            positions = placer.predict_size_and_position(bg_path, 'horse')
            if not positions:
                print(f"No valid positions found for {bg_name}")
                continue
            
            # Выбираем случайный объект и позицию
            obj_path = np.random.choice(object_paths)
            
             # Загружаем объект с альфа-каналом
            foreground = load_image(obj_path, with_alpha=True)
            obj_name = os.path.splitext(os.path.basename(obj_path))[0]
            
            # Вставляем объект
            blended_image, insertion_mask, annotation_path = inserter.insert_object_with_annotation(
                bg_path, obj_path, positions, "horse"
            )
            
            # Сохраняем промежуточный результат
            blended_output = os.path.join(blended_dir, f"{bg_name}_{obj_name}_blended.png")
            save_image(blended_output, blended_image)
            
            # Улучшаем изображение
            # Простое улучшение
            result_simple = enhancer.enhance_image(blended_image)
            enhanced_output = os.path.join(enhanced_dir, f"{bg_name}_{obj_name}_subtle.png")
            save_image(enhanced_output, result_simple.image)
            
            # Улучшение с кастомными параметрами
            result_custom = enhancer.enhance_image(
                blended_image,
                prompt="professional photography, sharp details",
                strength=0.5,
                object_type="horse",
                scene_type="road"
            )
            enhanced_output = os.path.join(enhanced_dir, f"{bg_name}_{obj_name}_subtle_custom.png")
            save_image(enhanced_output, result_custom.image)
            
            # Улучшение с inpainting маской
            result_inpainting = enhancer.enhance_image(
                blended_image,
                mask=insertion_mask,
                object_type="bicycle rider"
            )
            enhanced_output = os.path.join(enhanced_dir, f"{bg_name}_{obj_name}_subtle_inpaint.png")
            save_image(enhanced_output, result_inpainting.image)
            
            
        except Exception as e:
            print(f"Error processing {bg_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
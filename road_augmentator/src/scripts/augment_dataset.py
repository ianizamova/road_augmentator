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
    #parser.add_argument('--backgrounds_dir', type=str, required=True, default="/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val", help='Path to background images')
    #parser.add_argument('--objects_dir', type=str, required=True, default='output_riders', help='Path to foreground objects with transparency')
    #parser.add_argument('--output_dir', type=str, required=True, default='experiment1', help='Output directory')
    #parser.add_argument('--placer_config', type=str, default='configs/placer_config.yaml')
    #parser.add_argument('--inserter_config', type=str, default='configs/inserter_config.yaml')
    #parser.add_argument('--enhancer_config', type=str, default='configs/enhancer_config.yaml')
 
 
    parser.add_argument('--backgrounds_dir', type=str, required=False, default="/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val", help='Path to background images')
    parser.add_argument('--objects_dir', type=str, required=False, default='output_riders', help='Path to foreground objects with transparency')
    parser.add_argument('--output_dir', type=str, required=False, default='experiment1', help='Output directory')
       
    args = parser.parse_args()
    
    # Создаем выходные директории
    os.makedirs(args.output_dir, exist_ok=True)
    blended_dir = os.path.join(args.output_dir, 'blended')
    enhanced_dir = os.path.join(args.output_dir, 'enhanced')
    os.makedirs(blended_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Загружаем конфиги
    #placer_config = load_config(args.placer_config)
    #inserter_config = load_config(args.inserter_config)
    #enhancer_config = load_config(args.enhancer_config)
    
    # Инициализируем компоненты
    placer = ObjectPlacer()
    inserter = ObjectInserter()
    enhancer = ImageEnhancer()
    
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
            positions = placer.predict_size_and_position(bg_path, 'bicycle')
            if not positions:
                print(f"No valid positions found for {bg_name}")
                continue
            
            # Выбираем случайный объект и позицию
            obj_path = np.random.choice(object_paths)
            
            position_num =  random.randint(0, 5)
            position = positions[position_num]  # Берем случайную позицию
            
            # Загружаем объект с альфа-каналом
            foreground = load_image(obj_path, with_alpha=True)
            obj_name = os.path.splitext(os.path.basename(obj_path))[0]
            
            # # Вставляем объект
            # blended_image, insertion_mask = inserter.insert_object(
            #     background, foreground, position, position['depth']
            # )
            
            # Вставляем объект
            blended_image, insertion_mask = inserter.insert_object(
                bg_path, obj_path, position, position['depth']
            )
            
            # Сохраняем промежуточный результат
            blended_output = os.path.join(blended_dir, f"{bg_name}_{obj_name}_blended.png")
            save_image(blended_output, blended_image)
            
            # Улучшаем изображение
            enhanced_image = enhancer.enhance_image(blended_image, insertion_mask)
            
            # Сохраняем финальный результат
            enhanced_output = os.path.join(enhanced_dir, f"{bg_name}_{obj_name}_enhanced.png")
            save_image(enhanced_output, enhanced_image)
            
        except Exception as e:
            print(f"Error processing {bg_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
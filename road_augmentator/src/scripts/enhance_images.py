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
    parser.add_argument('--images_dir', type=str, required=False, default="/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val", help='Path to background images')
    parser.add_argument('--output_dir', type=str, required=False, default='experiment1', help='Output directory')
    parser.add_argument('--enhancer_config', type=str, default='configs/enhancer_config.yaml')
 
 
    #parser.add_argument('--images_dir', type=str, required=False, default="/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/val", help='Path to background images')
    #parser.add_argument('--output_dir', type=str, required=False, default='experiment1', help='Output directory')
       
    args = parser.parse_args()
    
    # Создаем выходные директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    enhancer = ImageEnhancer(config_path="road_augmentator/configs/enhancer_config.json")
    
    # Получаем список изображений
    images_paths = glob(os.path.join(args.images_dir, '*.jpg')) + \
                      glob(os.path.join(args.images_dir, '*.png'))

    print(f"Found {len(images_paths)} images to enhance ")
    
    # Обрабатываем каждое фоновое изображение
    for img_path in tqdm(images_paths, desc="Processing images"):
        try:
            # Загружаем фон
            image2e = load_image(img_path)
            bg_name = os.path.splitext(os.path.basename(img_path))[0]
            
            
            # Улучшаем изображение
            # Простое улучшение
            result_simple = enhancer.enhance_image(image2e)
            enhancer.save_result(result_simple, img_path)
            # Улучшение с кастомными параметрами
            result_custom = enhancer.enhance_image(
                image2e,
                prompt="professional photography, sharp details",
                strength=0.4,
                object_type="bicycle",
                scene_type="road"
            )
            enhancer.save_result(result_custom, img_path)
            # Улучшение с inpainting маской
            # result_inpainting = enhancer.enhance_image(
            #     image2e,
            #     mask=insertion_mask,
            #     object_type="bicycle"
            # )
            # enhancer.save_result(result_inpainting, img_path)
            
            #enhanced_image = enhancer.enhance_image(blended_image, insertion_mask)
            
          
            #save_image(enhanced_output, enhanced_image)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
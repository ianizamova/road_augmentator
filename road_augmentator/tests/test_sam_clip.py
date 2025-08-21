import cv2
import torch
import clip
import numpy as np
from PIL import Image
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Загрузка SAM
sam = sam_model_registry["vit_l"](checkpoint="/home/irina/work/otus_cv/blackswan_generator/sam/checkpoints/sam_vit_l_0b3195.pth").to("cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

# Загрузка CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

# Классы для классификации
classes = ["road", "car", "building", "greenery", "driving area", "ground", "driving_lane"]
text_inputs = torch.cat([clip.tokenize(f"{c}") for c in classes]).to("cuda")

def classify_segment(image, mask):
    """Классифицирует сегмент с помощью CLIP"""
    masked_image = image * mask[:, :, np.newaxis]
    patch = cv2.resize(masked_image, (224, 224))
    patch = clip_preprocess(Image.fromarray(patch)).unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        image_features = clip_model.encode_image(patch)
        text_features = clip_model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
    
    best_class_idx = similarity.argmax().item()
    confidence = similarity.max().item()
    return classes[best_class_idx], confidence


def segment_and_clip(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    
    # Классификация каждого сегмента
    road_segments_list = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        class_name, confidence = classify_segment(image, mask)
        if class_name in ["road", "driving area", "car"] and confidence > 0.8:
            road_segments_list.append(mask)
        print(f"Segment {i}: {class_name} (confidence: {confidence:.2f})")
        

    for idx_mask,road_segment in enumerate(road_segments_list):
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
    
    return road_segments_list


if __name__ == "__main__":
    path_to_images = "/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/test"  # Замените на свой путь

    imgs_list = []
    for _, _, files in os.walk(path_to_images):
        imgs_list.extend(files)
        
    road_segments_dict = {}
    
    for image in imgs_list:
        image_path = os.path.join(path_to_images, image)
        
        road_segments_dict[image] = segment_and_clip(image_path)
    
    print(road_segments_dict)        
    
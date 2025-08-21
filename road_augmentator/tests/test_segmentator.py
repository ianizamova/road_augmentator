import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches

# Загрузка модели с весами Cityscapes (19 классов)
#model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=19)
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# Загрузите веса вручную (пример пути)
weights_path = '/home/irina/work/otus_cv/blackswan_generator/weights/pytorch_model.bin'  # Замените на реальный путь
#model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
model.eval()

# Палитра Cityscapes (19 классов + фон)
palette = np.array([
    [128, 64, 128],    # 0: 'road'
    [244, 35, 232],    # 1: 'sidewalk'
    [70, 70, 70],      # 2: 'building'
    [102, 102, 156],   # 3: 'wall'
    [190, 153, 153],   # 4: 'fence'
    [153, 153, 153],   # 5: 'pole'
    [250, 170, 30],    # 6: 'traffic light'
    [220, 220, 0],     # 7: 'traffic sign'
    [107, 142, 35],    # 8: 'vegetation'
    [152, 251, 152],   # 9: 'terrain'
    [70, 130, 180],    # 10: 'sky'
    [220, 20, 60],     # 11: 'person'
    [255, 0, 0],       # 12: 'rider'
    [0, 0, 142],       # 13: 'car'
    [0, 0, 70],        # 14: 'truck'
    [0, 60, 100],      # 15: 'bus'
    [0, 80, 100],      # 16: 'train'
    [0, 0, 230],       # 17: 'motorcycle'
    [119, 11, 32],     # 18: 'bicycle'
    [0, 0, 0]          # 19: 'background' (не используется в Cityscapes)
], dtype=np.uint8)

# Названия классов
class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]

# Трансформации
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка изображения
image_path = "/media/irina/ADATA HD330/data/datasets/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/10k/test/df34b4be-c313d03a.jpg"  # Замените на свой путь
image = Image.open(image_path)
original_width, original_height = image.size
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Предсказание
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0).cpu().numpy()

# Раскрашивание маски
segmentation_colored = palette[output_predictions]
segmentation_colored = cv2.resize(segmentation_colored, 
                                (original_width, original_height), 
                                interpolation=cv2.INTER_NEAREST)

# Сохранение результата
output_path = "segmentation_result_coco.png"
cv2.imwrite(output_path, cv2.cvtColor(segmentation_colored, cv2.COLOR_RGB2BGR))

# Создание легенды
legend_patches = [mpatches.Patch(color=np.array(palette[i])/255., 
                                label=class_names[i]) for i in range(19)]

# Визуализация
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentation Result")
plt.imshow(segmentation_colored)
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axis('off')

plt.tight_layout()
plt.savefig("segmentation_with_legend_coco.png", bbox_inches='tight')
plt.show()
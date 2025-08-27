# Road Scene Augmentation Pipeline

Проект для аугментации дорожных сцен с помощью добавления редких объектов с использованием современных методов компьютерного зрения и генеративного ИИ.

## Возможности

- **Извлечение объектов**: Автоматическое извлечение редких объектов из датасетов (COCO)
- **Генерация объектов**: Создание новых объектов с помощью генеративных моделей (GANs, Diffusion)
- **Предсказание позиций**: Интеллектуальное размещение объектов на сцене с учетом сегментации и глубины
- **Реалистичное встраивание**: Правдоподобное встраивание объектов с учетом освещения и перспективы
- **Улучшение качества**: Постобработка с помощью диффузионных моделей для бесшовного интеграции

## Установка

### Предварительные требования

- Python 3.9+
- GPU (рекомендуется) для работы с моделями

## Датасеты
Для работы проекта используются датасеты 
- https://github.com/bdd100k/bdd100k - для фонов 
- https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset - для объектов



## Структура проекта

```
road_augmentator/
├── configs/                 # Конфигурационные файлы
│   ├── placer_config.json    # Предсказатель позиций
│   ├── inserter_config.json   # Вставка объектов
│   └── enhancer_config.json   # Улучшение качества
├── src/                       # Исходный код
│   ├── core/                  # Классы 
│   │   ├── object_extractor.py    # Извлечение объектов
│   │   ├── object_generator.py    # Генерация объектов
│   │   ├── position_predictor.py  # Предсказание позиций
│   │   ├── inserter.py            # Вставка объектов
│   │   └── enhancer.py            # Улучшение качества
│   ├── scripts/
│   │   └── augment_dataset.py    #  Запуск пайплайна
│   └── utils/                    # Вспомогательные утилиты
│       ├── image_utils.py            # работа с изображениями
│       ├── bdd100k_load.py           # загрузка датасета с фонами
│       └── coco_load.py              # загрузка датасета с объектами
├── tests/                 # тесты (тут не те тесты)
│   ├── test_efficientvit.py     # тест сегментатора efficientvit
│   ├── test_sam_clip.py         # тест сегментатора SAM + классификатора CLIP
│   └── test_segmentator.py      # тест сегментатора
└── docs/                        # Документация (пока нет)
```


## 🧪 Примеры использования

### Вырезание объектов

```python
from src.object_extractor import ObjectExtractor

extractor = ObjectExtractor("configs/extract_config.json")

if extractor.initialize():
    # Запускаем извлечение
    extractor.extract_objects()
```

### Только предсказание позиций

```python
from src.position_predictor import ObjectPlacer
from src.utils.config_loader import load_config

config = load_config("configs/object_placer_config.json")
placer = ObjectPlacer(config)

positions = placer.predict_position(background_image)
print(f"Найдено позиций: {len(positions)}")
```

### Кастомная вставка объектов

```python
from src.object_inserter import ObjectInserter

inserter = ObjectInserter(config)
result, mask = inserter.insert_object(
    background, 
    foreground_object, 
    position, 
    depth_value
)
```
### Основной пайплайн 

Основной пайплайн сейчас описан в файле scripts/augment_dataset.py

```bash
cd road_augmentator
python3 src/scripts/augment_dataset.py --backgrounds_dir="/path/to/backgrounds" --objects_dir="/path/to/objects" --output_dir="/path/to/output"
```

## 📊 Поддерживаемые модели

### Сегментация
- **facebook/maskformer-swin-large-coco** - семантическая сегментация
- **Intel/dpt-large** - оценка глубины

### Генеративные модели
- **stabilityai/stable-diffusion-2-1-base** - улучшение качества изображения
- **stabilityai/stable-diffusion-2-inpainting** - inpainting модель для добавления объектов на изображение


- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers/index)

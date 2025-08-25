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
- PDM (менеджер зависимостей)
- GPU (рекомендуется) для работы с моделями

### Установка с помощью PDM

1. **Установите PDM** (если не установлен):
```bash
pip install pdm
```

2. **Клонируйте репозиторий**:
```bash
git clone https://github.com/your-username/road-augmentation-project.git
cd road-augmentation-project
```

3. **Установите зависимости**:
```bash
pdm install
```

4. **Для разработки** установите дополнительные зависимости:
```bash
pdm install -d dev
```

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
├── tests/                   # Тесты
└── docs/                    # Документация
```

## 🚀 Быстрый старт

### 1. Подготовка данных

Поместите фоновые изображения и объекты в соответствующие директории:

```bash
mkdir -p data/raw/backgrounds
mkdir -p data/raw/objects

# Поместите фоновые изображения в data/raw/backgrounds/
# Поместите объекты с прозрачным фоном в data/raw/objects/
```

### 2. Запуск полного пайплайна

```bash
# Запуск с параметрами по умолчанию
pdm run run-pipeline \
  --backgrounds_dir data/raw/backgrounds \
  --objects_dir data/raw/objects \
  --output_dir data/output
```

### 3. Запуск с кастомными настройками

```bash
pdm run run-pipeline \
  --backgrounds_dir data/raw/backgrounds \
  --objects_dir data/raw/objects \
  --output_dir data/output \
  --placer_config configs/custom_placer.yaml \
  --enhancer_config configs/custom_enhancer.yaml
```

## ⚙️ Конфигурация

### Основные параметры конфигурации

Создайте кастомные конфиги в директории `configs/`:

**`configs/custom_placer.yaml`**:
```yaml
model:
  depth_model: "dpt_hybrid"
  seg_model: "mask2former"

params:
  target_classes: ["road", "sidewalk", "ground"]
  min_depth: 0.1
  max_depth: 0.8
  num_candidates: 5
```

**`configs/custom_enhancer.yaml`**:
```yaml
model:
  name: "stabilityai/stable-diffusion-2-inpainting"
  device: "cuda"

params:
  prompt: "realistic, seamless integration, natural shadows"
  strength: 0.8
```

## 🧪 Примеры использования

### Только предсказание позиций

```python
from src.position_predictor import ObjectPlacer
from src.utils.config_loader import load_config

config = load_config("configs/placer_config.yaml")
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

## 🛠️ Команды разработки

```bash
# Запуск тестов
pdm run test

# Форматирование кода
pdm run format

# Линтинг
pdm run lint

# Активация виртуального окружения
pdm shell

# Просмотр зависимостей
pdm list
```

## 📊 Поддерживаемые модели

### Детекция и сегментация
- **Detectron2** - детекция объектов
- **Mask2Former** - семантическая сегментация
- **MiDaS** - оценка глубины

### Генеративные модели
- **Stable Diffusion** - генерация и улучшение
- **ControlNet** - контролируемая генерация
- **GANs** - генерация объектов

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте feature branch: `git checkout -b feature/new-feature`
3. Сделайте коммит: `git commit -am 'Add new feature'`
4. Запушьте ветку: `git push origin feature/new-feature`
5. Создайте Pull Request

## 📝 Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для подробностей.

## 🙋‍♂️ Поддержка

Если у вас возникли вопросы или проблемы:

1. Проверьте [Issues](https://github.com/your-username/road-augmentation-project/issues)
2. Создайте новое Issue с описанием проблемы
3. Напишите на email: your.email@example.com

## 📚 Дополнительные материалы

- [Документация PDM](https://pdm.fming.dev/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers/index)

---

**Примечание**: Для работы с большими моделями рекомендуется использовать GPU с至少 8GB памяти.
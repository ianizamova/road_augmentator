# road_augmentor/src/cli.py
import argparse
from core.pipeline import AugmentationPipeline
from road_augmentator.src.utils.logger import setup_logger

def main():
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description="Генератор редких объектов для датасетов")
    
    # Добавляем аргументы
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Путь к конфигурационному файлу"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/outputs",
        help="Директория для сохранения результатов"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",  # Флаг (без значения)
        help="Включить режим отладки"
    )

    # Парсим аргументы
    args = parser.parse_args()

    # Настройка логирования
    logger = setup_logger(debug=args.debug)

    # Запуск пайплайна
    pipeline = AugmentationPipeline(config_path=args.config)
    pipeline.run(output_dir=args.output_dir)

if __name__ == "__main__":
    main()
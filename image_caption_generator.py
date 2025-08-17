import os
import csv
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model():
    """Инициализация модели BLIP"""
    logger.info("Загрузка модели BLIP...")

    # Используем MPS если доступен (для Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")

    # Загрузка процессора и модели
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = model.to(device)

    return processor, model, device


def setup_translation_model():
    """Инициализация модели перевода"""
    logger.info("Загрузка модели перевода...")

    # Модель для перевода с английского на русский
    translation_model_name = "Helsinki-NLP/opus-mt-en-ru"
    tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    model = MarianMTModel.from_pretrained(translation_model_name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    return tokenizer, model, device


def translate_text(text, tokenizer, model, device):
    """Перевод текста с английского на русский"""
    try:
        # Форматирование текста для модели перевода
        formatted_text = f">>ru<< {text}" if ">>ru<<" not in text else text

        # Кодирование текста
        inputs = tokenizer(formatted_text, return_tensors="pt", padding=True).to(device)

        # Генерация перевода
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=100)

        # Декодирование перевода
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        return result

    except Exception as e:
        logger.error(f"Ошибка при переводе текста '{text}': {e}")
        return text  # Возвращаем оригинальный текст в случае ошибки


def get_image_files(folder_path):
    """Получение списка изображений из папки"""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = []

    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)

    return sorted(image_files)


def generate_caption(image_path, processor, model, device):
    """Генерация описания для одного изображения"""
    try:
        # Открываем изображение
        image = Image.open(image_path).convert("RGB")

        # Подготовка изображения для модели
        inputs = processor(image, return_tensors="pt").to(device)

        # Генерация описания (на английском)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_length=50, num_beams=3, early_stopping=True
            )

        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    except Exception as e:
        logger.error(f"Ошибка при обработке {image_path}: {e}")
        return "Ошибка генерации описания"


def process_images(input_folder, output_csv, translate_to_russian=True):
    """Основная функция обработки изображений"""

    # Создание папки для CSV если её нет
    os.makedirs(
        os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".",
        exist_ok=True,
    )

    # Инициализация модели BLIP
    processor, blip_model, blip_device = setup_model()

    # Инициализация модели перевода (если нужно)
    if translate_to_russian:
        translation_tokenizer, translation_model, translation_device = (
            setup_translation_model()
        )
    else:
        translation_tokenizer, translation_model, translation_device = None, None, None

    # Получение списка изображений
    image_files = get_image_files(input_folder)

    if not image_files:
        logger.warning("В папке не найдено изображений")
        return

    logger.info(f"Найдено {len(image_files)} изображений для обработки")

    # Открытие CSV файла для записи
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Запись заголовков
        if translate_to_russian:
            writer.writerow(["Путь к фото", "Описание (англ)", "Описание (рус)"])
        else:
            writer.writerow(["Путь к фото", "Описание"])

        # Обработка каждого изображения
        for filename in tqdm(image_files, desc="Обработка изображений"):
            image_path = os.path.join(input_folder, filename)

            # Генерация описания на английском
            caption_en = generate_caption(
                image_path, processor, blip_model, blip_device
            )

            # Перевод на русский (если нужно)
            if translate_to_russian and translation_model is not None:
                caption_ru = translate_text(
                    caption_en,
                    translation_tokenizer,
                    translation_model,
                    translation_device,
                )
                # Запись в CSV
                writer.writerow([image_path, caption_en, caption_ru])
            else:
                # Запись в CSV
                writer.writerow([image_path, caption_en])

            # Очистка GPU памяти (если используется MPS)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    logger.info(f"Результаты сохранены в {output_csv}")


def main():
    # Настройки путей
    input_folder = "images"  # Папка с изображениями
    output_csv = "image_captions.csv"  # Выходной CSV файл

    # Проверка существования папки
    if not os.path.exists(input_folder):
        logger.error(f"Папка {input_folder} не существует")
        return

    # Запуск обработки с переводом на русский
    process_images(input_folder, output_csv, translate_to_russian=True)


if __name__ == "__main__":
    main()

```markdown
# Image Caption Generator

This project automatically generates descriptions for images using AI models and translates them from English to Russian.

## 📋 Features

- Generates English descriptions for images using BLIP model
- Translates descriptions from English to Russian
- Saves results to CSV file
- Supports multiple image formats (JPG, PNG, BMP, TIFF, etc.)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image_caption_generator
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install transformers torch Pillow tqdm sentencepiece
```

## 📁 Project Structure

```
project/
├── images/                 # Put your images here
├── image_caption_generator.py  # Main script
├── image_captions.csv      # Output file with descriptions
└── README.md
```

## ▶️ Usage

1. Place your images in the `images/` folder
2. Run the script:
```bash
python image_caption_generator.py
```
3. Check results in `image_captions.csv`

## ⚙️ How It Works

1. **Image Analysis**: Uses Salesforce BLIP model to generate English descriptions
2. **Translation**: Translates English descriptions to Russian using translation model
3. **Output**: Saves both original and translated descriptions to CSV file

## 📄 Output Format

The CSV file contains three columns:
- **Path to photo**: Full path to the image file
- **Description (eng)**: Original English description
- **Description (rus)**: Translated Russian description

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Transformers library
- Pillow
- tqdm
- sentencepiece

## 📝 Example

Input image: `cat.jpg`
Output in CSV:
```
Path to photo,Description (eng),Description (rus)
images/cat.jpg,A cat sitting on a windowsill,Кошка, сидящая на подоконнике
```

## 🤖 Models Used

- **BLIP**: `Salesforce/blip-image-captioning-base` for image description
- **Translation**: `Helsinki-NLP/opus-mt-en-ru` for English-Russian translation

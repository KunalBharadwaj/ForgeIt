# Data Directory

## Overview

This directory contains datasets for the fake news detection project.

## Structure

- `raw/`: Original, unprocessed datasets
- `processed/`: Preprocessed and cleaned data ready for training

## Dataset Sources

### Recommended Datasets

1. **Kaggle Fake News Dataset**
   - URL: https://www.kaggle.com/c/fake-news/data
   - Format: CSV with `id`, `title`, `author`, `text`, `label`
   - Size: ~20,000 articles

2. **LIAR Dataset**
   - URL: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
   - Format: TSV with political statements and truth ratings
   - Size: ~12,800 statements

3. **FakeNewsNet**
   - URL: https://github.com/KaiDMML/FakeNewsNet
   - Format: JSON with social media data
   - Size: Varies by subset

4. **ISOT Fake News Dataset**
   - URL: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
   - Format: CSV files (separate for True and Fake)
   - Size: ~44,000 articles

## Required Format

Your dataset should be in CSV format with at minimum these columns:

```csv
text,label
"News article content here...",0
"Another article content...",1
```

Where:
- `text`: The news article content (can also be named `title`, `content`, or `article`)
- `label`: Binary classification
  - 0 = Real/True news
  - 1 = Fake/False news
  - Or: "Real"/"Fake", "True"/"False"

## Downloading Data

### Option 1: Kaggle CLI

```bash
# Install kaggle CLI
pip install kaggle

# Place your kaggle.json in ~/.kaggle/
# Download dataset
kaggle competitions download -c fake-news
unzip fake-news.zip -d data/raw/
```

### Option 2: Manual Download

1. Visit the dataset URL
2. Download the CSV file
3. Place it in `data/raw/` directory

### Option 3: Use Sample Data

For testing purposes, the preprocessing script can generate a small sample dataset.

## Data Preprocessing

The `src/data_preprocessing.py` script will:
- Load raw data from `data/raw/`
- Clean text (remove URLs, special characters, etc.)
- Tokenize and normalize text
- Handle missing values
- Balance classes if needed
- Split into train/validation/test sets
- Save processed data to `data/processed/`

Run preprocessing:
```bash
python src/data_preprocessing.py --input data/raw/news.csv --output data/processed/
```

## Data Statistics

After downloading your dataset, run:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/news.csv'); print(df.describe()); print(df['label'].value_counts())"
```

## Important Notes

⚠️ **Privacy & Ethics**
- Ensure datasets are used ethically and legally
- Respect copyright and terms of use
- Do not redistribute datasets without permission

⚠️ **Bias Considerations**
- Check for dataset bias (political, temporal, source)
- Ensure balanced representation
- Consider domain adaptation for real-world use

## Data Not Included

Due to size and licensing, datasets are not included in this repository. Please download them separately using the instructions above.

## Need Help?

If you encounter issues:
1. Check dataset format matches requirements
2. Verify file paths are correct
3. Ensure sufficient disk space
4. Check file permissions

For questions, please open an issue on the project repository.

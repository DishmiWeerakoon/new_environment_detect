# Environmental Context Recognition System

Identifies whether a user is in **Transportation Mode** or **Conversation Mode** using environmental audio, without training directly on high-level labels.

## Pipeline Overview

```
Raw Audio
   │
   ▼
Feature Extraction (MFCCs, Chroma, Mel Spectrogram)
   │
   ▼
Sound Event Classification (Random Forest / SVM)
   │  car_horn, engine_idling, siren, speech, conversation, …
   ▼
Rule-Based Environment Mapping
   │
   ▼
Environment Mode: Transportation | Conversation | Unknown
```

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   ├── UrbanSound8K/          # Download from urbansounddataset.weebly.com
│   │   │   ├── audio/fold{1-10}/
│   │   │   └── metadata/UrbanSound8K.csv
│   │   └── custom_audio/          # Optional extra speech/conversation clips
│   └── processed/
│       ├── features/              # Extracted feature CSVs
│       ├── labels/                # Label files
│       └── train_test_split/      # Train/test feature arrays (.npy)
├── models/                        # Saved model and scaler (.pkl)
├── notebooks/
│   ├── 01_data_loading.ipynb      # Dataset exploration & visualization
│   ├── 02_feature_extraction.ipynb
│   ├── 03_label_mapping.ipynb     # Sound → environment label mapping
│   ├── 04_model_training.ipynb    # RF / SVM training & cross-validation
│   ├── 05_model_evaluation.ipynb  # Confusion matrix, metrics, feature importance
│   └── 06_prediction.ipynb        # Sliding window + environment detection
├── src/
│   ├── feature_extraction.py
│   ├── sound_classifier.py
│   ├── environment_mapper.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── test_audio/                    # Short WAV files for quick demos
└── requirements.txt
```

## Sound Class Mapping

| Sound Class        | Environment Group   |
|--------------------|---------------------|
| car_horn           | Transportation      |
| engine_idling      | Transportation      |
| siren              | Transportation      |
| jackhammer         | Transportation      |
| drilling           | Transportation      |
| air_conditioner    | Transportation      |
| speech             | Conversation        |
| conversation       | Conversation        |
| children_playing   | Conversation        |
| dog_bark           | Ambient             |
| street_music       | Ambient             |
| gun_shot           | Ambient             |

## Setup

```bash
pip install -r requirements.txt
```

Download [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) and place it under `data/raw/UrbanSound8K/`.

## Workflow

Run notebooks in order (01 → 06), or use the CLI scripts:

```bash
# Extract features from UrbanSound8K
python src/preprocess.py --dataset data/raw/UrbanSound8K --output data/processed

# Train classifier
python src/train.py --features data/processed/features.csv --model models/

# Predict environment from an audio file
python src/predict.py --audio test_audio/sample.wav --model models/
```

## Environment Detection Logic

A **sliding window** (default 2 s, 1 s hop) produces a stream of sound-event predictions. A **rule-based mapper** then counts how many of the last N predictions belong to transportation vs. conversation sound categories:

- ≥ 40 % transportation sounds → **Transportation Mode**
- ≥ 35 % conversation sounds  → **Conversation Mode**
- Otherwise                   → **Unknown / Ambient Mode**

**Majority voting** over a 5-second window further smooths the final label.

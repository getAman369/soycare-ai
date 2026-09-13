# SoyCare AI

An AI-assisted soybean leaf disease detection prototype. Upload a leaf photograph to receive a model prediction, confidence score, safe management guidance, and a locally stored detection history.

## What is included

- Responsive Flask frontend for uploading soybean leaf images
- `/api/predict` endpoint with image validation and prediction output
- Editable disease-management knowledge base
- SQLite-backed scan history and dashboard metrics
- EfficientNetB0 transfer-learning training script
- A safe demo mode while a trained model is not yet available

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Before training, the interface deliberately labels results as **Demo mode**. It does not present random fallback output as a clinical or agricultural diagnosis.

## Train the model

Arrange verified, labelled images as follows:

```text
data/processed/
├── train/<disease-class>/*.jpg
├── validation/<disease-class>/*.jpg
└── test/<disease-class>/*.jpg
```

Every split must have the same class-folder names, in alphabetical order. Then run:

```bash
python -m src.train
```

The saved model is placed at `models/soybean_disease_model.keras`. Update `CLASSES` in `src/predict.py` to exactly match the directory order printed by the training script before deploying.

## Evaluation checklist

For the final report, measure accuracy, precision, recall, F1-score, class-wise confusion matrix, and inference time on the held-out test set. Also compare results on controlled dataset images against real field images.

## Responsible-use note

SoyCare AI is a decision-support prototype, not a replacement for an agronomist. Treatments vary by region, crop stage, disease pressure, product label, and local regulation. Confirm control decisions with a qualified agricultural professional.

## GitHub publishing

After creating an empty repository named `soycare-ai` on GitHub, run:

```bash
git add .
git commit -m "Initial SoyCare AI prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/soycare-ai.git
git push -u origin main
```

Never upload raw farmer images, credentials, or the generated SQLite database unless you have explicit permission.

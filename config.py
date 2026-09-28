"""
Application Configuration
Centralized settings for SoyCare AI Soybean Leaf Disease Classification System.
"""
from pathlib import Path
import os
import logging

# Base Directory paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
HEATMAP_DIR = UPLOAD_DIR / "heatmaps"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "soycare.db"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "soybean_disease_model.keras"
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base" / "disease_recommendations.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
SECRET_KEY = os.getenv("SOYCARE_SECRET_KEY")
DEBUG = os.getenv("SOYCARE_DEBUG", "false").lower() == "true"

# Ensure runtime directories exist
for directory in (UPLOAD_DIR, HEATMAP_DIR, DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Image & Model specifications
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Real-data experiment classes (alphabetical order matching training folder structure)
# These are the classes the currently shipped model was trained on.
CLASSES = [
    "Frogeye leaf spot",
    "Healthy",
    "Soybean rust"
]

# Phase 0 target taxonomy: the six classes the knowledge base documents.
# CLASSES is promoted to this list only after a real dataset passes
# `python -m scripts.audit_dataset` and the model is retrained, so inference
# never indexes a class the loaded weights were not trained for.
TARGET_CLASSES = [
    "Bacterial blight",
    "Downy mildew",
    "Frogeye leaf spot",
    "Healthy",
    "Septoria brown spot",
    "Soybean rust"
]

# Minimum verified real images per class before retraining is considered.
MIN_IMAGES_PER_CLASS = 300

# Source registry for real dataset acquisition. See DATASET_GUIDE.md.
DATASET_SOURCES_PATH = DATA_DIR / "sources.json"

# Risk assessment thresholds
HIGH_RISK_CONFIDENCE_THRESHOLD = 80.0
LOW_CONFIDENCE_WARNING_THRESHOLD = 60.0

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("soycare")

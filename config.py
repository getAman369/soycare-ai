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

# Ensure runtime directories exist
for directory in (UPLOAD_DIR, HEATMAP_DIR, DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Image & Model specifications
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Soybean Disease Classification Classes (Alphabetical order matching training folder structure)
CLASSES = [
    "Bacterial blight",
    "Downy mildew",
    "Frogeye leaf spot",
    "Healthy",
    "Septoria brown spot",
    "Soybean rust"
]

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

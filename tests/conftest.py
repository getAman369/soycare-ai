"""
Pytest configuration and shared fixtures for SoyCare AI test suite.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import sqlite3
import numpy as np
from PIL import Image, ImageDraw

import config
from app import app, initialise_database
import app as app_module
import src.predict as predict_module


class FakePredictor:
    """Deterministic predictor used by API tests instead of local model weights."""

    def predict(self, image_path, filename=None):
        return {
            "disease": "Healthy",
            "confidence": 96.0,
            "top_predictions": [
                {"disease": "Healthy", "confidence": 96.0},
                {"disease": "Soybean rust", "confidence": 2.0},
                {"disease": "Downy mildew", "confidence": 1.0}
            ],
            "heatmap_url": f"/uploads/heatmaps/heatmap_{filename or image_path.name}",
            "is_demo": False,
            "is_low_confidence": False,
            "is_valid_leaf": True,
            "note": "Test prediction."
        }


@pytest.fixture(scope="session")
def temp_test_dir():
    """Creates a temporary workspace directory for test runtime assets."""
    temp_dir = Path(tempfile.mkdtemp(prefix="soycare_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client(temp_test_dir, monkeypatch):
    """Provides a configured Flask test client with an isolated SQLite database."""
    test_db = temp_test_dir / "test_soycare.db"
    test_uploads = temp_test_dir / "uploads"
    test_heatmaps = test_uploads / "heatmaps"
    test_uploads.mkdir(parents=True, exist_ok=True)
    test_heatmaps.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATABASE_PATH", test_db)
    monkeypatch.setattr(config, "UPLOAD_DIR", test_uploads)
    monkeypatch.setattr(config, "HEATMAP_DIR", test_heatmaps)
    monkeypatch.setattr(app_module, "DATABASE_PATH", test_db)
    monkeypatch.setattr(app_module, "UPLOAD_DIR", test_uploads)
    monkeypatch.setattr(predict_module, "HEATMAP_DIR", test_heatmaps)
    monkeypatch.setattr(app_module, "predictor", FakePredictor())

    initialise_database()

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_leaf_image(temp_test_dir):
    """Generates a realistic synthetic soybean green leaf image for testing."""
    img_path = temp_test_dir / "sample_leaf.jpg"
    img = Image.new("RGB", (224, 224), color=(46, 125, 50))
    draw = ImageDraw.Draw(img)
    # Draw leaf vein structure
    draw.line([(112, 10), (112, 214)], fill=(20, 90, 20), width=3)
    for y in range(40, 200, 30):
        draw.line([(112, y), (40, y + 25)], fill=(25, 100, 25), width=2)
        draw.line([(112, y), (184, y + 25)], fill=(25, 100, 25), width=2)
    img.save(img_path, quality=95)
    return img_path


@pytest.fixture
def blank_invalid_image(temp_test_dir):
    """Generates a solid blank non-leaf image to test OOD rejection."""
    img_path = temp_test_dir / "blank_image.jpg"
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    img.save(img_path, quality=95)
    return img_path

"""
Unit tests for DiseasePredictor and image preprocessing validation.
"""
from pathlib import Path
import numpy as np
import pytest

from src.predict import DiseasePredictor, validate_leaf_image
from config import CLASSES, IMAGE_SIZE


class FakeModel:
    """Minimal model double for testing prediction response formatting."""

    def predict(self, image_tensor, verbose=0):
        return np.array([[0.04, 0.90, 0.06]], dtype=np.float32)


def test_validate_leaf_image_valid(sample_leaf_image):
    """Verifies that a genuine green leaf image passes validation."""
    result = validate_leaf_image(sample_leaf_image)
    assert result["is_valid_leaf"] is True


def test_validate_leaf_image_invalid_blank(blank_invalid_image):
    """Verifies that a solid/blank non-leaf image is flagged as invalid."""
    result = validate_leaf_image(blank_invalid_image)
    assert result["is_valid_leaf"] is False
    assert "low contrast" in result["reason"].lower() or "foliage" in result["reason"].lower()


def test_preprocess_image_shape(sample_leaf_image):
    """Verifies that preprocessing converts an image into a (1, 224, 224, 3) float32 tensor."""
    predictor = DiseasePredictor()
    tensor = predictor.preprocess_image(sample_leaf_image)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, *IMAGE_SIZE, 3)
    assert tensor.dtype == np.float32


def test_predict_return_structure(sample_leaf_image, tmp_path):
    """Verifies that the predict method returns all required schema keys."""
    predictor = DiseasePredictor(model_path=tmp_path / "test_model.keras")
    predictor.model = FakeModel()
    prediction = predictor.predict(sample_leaf_image)

    assert "disease" in prediction
    assert prediction["disease"] in CLASSES
    assert "confidence" in prediction
    assert isinstance(prediction["confidence"], (int, float))
    assert "top_predictions" in prediction
    assert len(prediction["top_predictions"]) >= 1
    assert "heatmap_url" in prediction
    assert "is_demo" in prediction
    assert "is_valid_leaf" in prediction
    assert "note" in prediction


def test_predict_fails_when_model_is_unavailable(sample_leaf_image, tmp_path):
    """Prevents fabricated diagnoses when trained weights are missing."""
    predictor = DiseasePredictor(model_path=tmp_path / "missing_model.keras")

    with pytest.raises(RuntimeError, match="trained disease model is unavailable"):
        predictor.predict(sample_leaf_image)

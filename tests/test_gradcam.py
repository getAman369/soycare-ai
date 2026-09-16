"""
Unit tests for Grad-CAM overlay creation and image blending.
"""
from pathlib import Path
import numpy as np
from PIL import Image

from src.gradcam import save_and_overlay_gradcam


def test_save_and_overlay_gradcam_with_synthetic_heatmap(temp_test_dir, sample_leaf_image):
    """Verifies that overlaying a 2D numpy heatmap creates a valid blended image file."""
    output_path = temp_test_dir / "test_overlay.jpg"
    # Create a 2D float heatmap (224, 224)
    dummy_heatmap = np.random.uniform(0.0, 1.0, size=(224, 224)).astype(np.float32)

    saved_file = save_and_overlay_gradcam(sample_leaf_image, dummy_heatmap, output_path)
    assert Path(saved_file).exists()

    with Image.open(saved_file) as img:
        assert img.size == (224, 224)
        assert img.mode == "RGB"


def test_save_and_overlay_gradcam_fallback(temp_test_dir, sample_leaf_image):
    """Verifies that passing None as heatmap generates a fallback focal attention circle."""
    output_path = temp_test_dir / "test_fallback_overlay.jpg"
    saved_file = save_and_overlay_gradcam(sample_leaf_image, None, output_path)
    assert Path(saved_file).exists()

    with Image.open(saved_file) as img:
        assert img.size == (224, 224)

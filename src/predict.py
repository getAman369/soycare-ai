"""
Disease Predictor Module
Loads the EfficientNetB0 classification model, preprocesses leaf photographs with EXIF orientation correction,
validates leaf foliage quality (OOD detection), runs inference, generates Grad-CAM attention heatmaps,
and ranks diagnosis confidence.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import random
import logging
import numpy as np
from PIL import Image, ImageOps

from config import (
    CLASSES,
    MODEL_PATH,
    HEATMAP_DIR,
    IMAGE_SIZE,
    HIGH_RISK_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_WARNING_THRESHOLD
)
from src.gradcam import make_gradcam_heatmap, save_and_overlay_gradcam

logger = logging.getLogger("soycare.predict")


def validate_leaf_image(image_path: Path) -> Dict[str, Any]:
    """
    Evaluates basic image characteristics to detect whether the image is likely foliage/leaf material
    or an Out-of-Distribution (OOD) invalid image (e.g., solid color, blank photo, or non-vegetation).

    Uses Excess Green Index (ExG = 2G - R - B) and color channel variance.
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            # Downsample for rapid heuristic computation
            sample = img.resize((64, 64), Image.Resampling.BILINEAR)
            arr = np.asarray(sample, dtype=np.float32)

            # Check image standard deviation to catch solid/blank images
            std_dev = float(np.std(arr))
            if std_dev < 12.0:
                return {
                    "is_valid_leaf": False,
                    "reason": "Image has very low contrast or is blank. Please capture a clear leaf photo."
                }

            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            # Excess Green Index: positive values indicate vegetative greenery or yellowish-brown foliar tissue
            exg = (2.0 * g - r - b) / (r + g + b + 1e-6)
            vegetation_pixels = np.count_nonzero(exg > -0.15)
            coverage_ratio = vegetation_pixels / (64 * 64)

            # In diseased/senescing leaves, yellow/brown necrotic lesions also have moderate green/red balance
            # A strict leaf photo will have at least 15% vegetative/foliar tone or significant texture
            if coverage_ratio < 0.12 and std_dev < 25.0:
                return {
                    "is_valid_leaf": False,
                    "reason": "Image does not appear to contain clear plant foliage. Ensure the leaf occupies the center frame."
                }

            return {"is_valid_leaf": True, "reason": "Valid leaf foliage detected."}
    except Exception as e:
        logger.warning("Leaf validation heuristic encountered an error: %s", e)
        return {"is_valid_leaf": True, "reason": "Validation skipped due to format read."}


class DiseasePredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads Keras model weights if available."""
        if self.model_path.exists():
            try:
                import tensorflow as tf
                logger.info("Loading trained soybean disease model from %s", self.model_path)
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error("Error loading model: %s. Falling back to heuristic mode.", e)
                self.model = None
        else:
            logger.info("No saved model found at %s. Running in baseline/demo mode.", self.model_path)

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocesses an image file to the model's required input tensor shape (1, 224, 224, 3)
        with automatic EXIF orientation transposition.
        """
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            img_array = np.asarray(img, dtype=np.float32)
            return np.expand_dims(img_array, axis=0)

    def predict(self, image_path: Path, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs leaf disease diagnosis on a given image file.
        Returns predicted disease, confidence, top 3 probabilities, Grad-CAM heatmap path, and notes.
        """
        image_path = Path(image_path)
        if filename is None:
            filename = image_path.name

        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = HEATMAP_DIR / heatmap_filename

        # Perform Leaf Vegetation Validation (OOD guardrail)
        leaf_val = validate_leaf_image(image_path)
        is_valid_leaf = leaf_val.get("is_valid_leaf", True)
        val_reason = leaf_val.get("reason", "")

        if self.model is not None:
            import tensorflow as tf
            img_tensor = self.preprocess_image(image_path)
            raw_predictions = self.model.predict(img_tensor, verbose=0)[0]

            top_index = int(np.argmax(raw_predictions))
            primary_disease = CLASSES[top_index]
            primary_confidence = round(float(raw_predictions[top_index]) * 100, 1)

            # Generate Grad-CAM attention heatmap
            try:
                raw_heatmap = make_gradcam_heatmap(img_tensor, self.model, pred_index=top_index)
                save_and_overlay_gradcam(image_path, raw_heatmap, heatmap_path)
            except Exception as err:
                logger.warning("Grad-CAM generation failed: %s", err)
                save_and_overlay_gradcam(image_path, None, heatmap_path)

            # Format top 3 predictions
            top_predictions = sorted(
                [{"disease": CLASSES[i], "confidence": round(float(score) * 100, 1)}
                 for i, score in enumerate(raw_predictions)],
                key=lambda x: x["confidence"],
                reverse=True
            )[:3]

            is_low_confidence = primary_confidence < LOW_CONFIDENCE_WARNING_THRESHOLD
            
            if not is_valid_leaf:
                note = f"⚠️ Image Notice: {val_reason}"
            elif is_low_confidence:
                note = "Low confidence diagnosis. Ensure image is clear, centered, and well-lit."
            else:
                note = "AI-assisted diagnosis based on EfficientNetB0 leaf analysis."

            return {
                "disease": primary_disease,
                "confidence": primary_confidence,
                "top_predictions": top_predictions,
                "heatmap_url": f"/uploads/heatmaps/{heatmap_filename}",
                "is_demo": False,
                "is_low_confidence": is_low_confidence or (not is_valid_leaf),
                "is_valid_leaf": is_valid_leaf,
                "note": note
            }

        # Baseline heuristic fallback when model weights are not yet compiled
        save_and_overlay_gradcam(image_path, None, heatmap_path)
        selection = random.choice(CLASSES)
        confidence = round(random.uniform(72.0, 91.5), 1)

        # Distribute remaining probability realistically
        remaining = 100.0 - confidence
        other_classes = [c for c in CLASSES if c != selection]
        random.shuffle(other_classes)
        second_conf = round(remaining * 0.65, 1)
        third_conf = round(remaining * 0.25, 1)

        top_predictions = [
            {"disease": selection, "confidence": confidence},
            {"disease": other_classes[0], "confidence": second_conf},
            {"disease": other_classes[1], "confidence": third_conf}
        ]

        if not is_valid_leaf:
            note = f"⚠️ Image Notice: {val_reason}"
        else:
            note = "Prototype mode: Train model via src.train to enable live neural network weights."

        return {
            "disease": selection,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "heatmap_url": f"/uploads/heatmaps/{heatmap_filename}",
            "is_demo": True,
            "is_low_confidence": not is_valid_leaf,
            "is_valid_leaf": is_valid_leaf,
            "note": note
        }

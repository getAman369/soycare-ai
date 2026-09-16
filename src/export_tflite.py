"""
TFLite Model Exporter & Quantization Utility
Converts trained Keras EfficientNetB0 models into optimized TensorFlow Lite (.tflite) binaries
for mobile, embedded (Raspberry Pi), and edge drone deployment.
"""
from pathlib import Path
import argparse
import logging

from config import MODEL_PATH, MODELS_DIR, IMAGE_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("soycare.export_tflite")


def export_tflite(model_path: Path = MODEL_PATH, output_dir: Path = MODELS_DIR, quantize: str = "float16"):
    """
    Converts a saved .keras model into .tflite format with optional quantization.

    Args:
        model_path: Path to the trained Keras model file.
        output_dir: Directory where the exported .tflite files will be saved.
        quantize: Quantization mode ('none', 'float16', 'int8_dynamic').
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logger.error("Model file not found at %s. Please train model with 'python -m src.train' first.", model_path)
        return None

    import tensorflow as tf
    logger.info("Loading Keras model from %s...", model_path)
    keras_model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if quantize == "float16":
        logger.info("Applying Float16 precision quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        output_name = f"{model_path.stem}_fp16.tflite"
    elif quantize == "int8_dynamic":
        logger.info("Applying Dynamic Range INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        output_name = f"{model_path.stem}_int8.tflite"
    else:
        logger.info("Exporting standard unquantized FP32 TFLite model...")
        output_name = f"{model_path.stem}_fp32.tflite"

    tflite_model = converter.convert()
    output_file = output_dir / output_name

    with open(output_file, "wb") as f:
        f.write(tflite_model)

    original_size_mb = model_path.stat().st_size / (1024 * 1024)
    tflite_size_mb = output_file.stat().st_size / (1024 * 1024)
    compression_ratio = (1.0 - (tflite_size_mb / (original_size_mb + 1e-6))) * 100.0

    logger.info("✅ TFLite export complete: %s", output_file)
    logger.info("📦 Original Keras Size: %.2f MB | TFLite Size: %.2f MB (%.1f%% reduction)",
                original_size_mb, tflite_size_mb, compression_ratio)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SoyCare AI Model to TFLite for Edge Deployment")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH), help="Path to input .keras model")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR), help="Output directory for .tflite")
    parser.add_argument("--quantize", type=str, choices=["none", "float16", "int8_dynamic"], default="float16",
                        help="Quantization strategy (default: float16)")
    args = parser.parse_args()

    export_tflite(model_path=Path(args.model_path), output_dir=Path(args.output_dir), quantize=args.quantize)

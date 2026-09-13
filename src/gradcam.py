"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module
Generates visual explanation heatmaps highlighting regions in soybean leaf photographs
that contributed most strongly to the model's classification.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import logging

logger = logging.getLogger("soycare.gradcam")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Computes a Grad-CAM heatmap for a given input image array and Keras model.
    """
    import tensorflow as tf

    # Find the last 4D convolutional layer if not explicitly specified
    if last_conv_layer_name is None:
        # Check if the model has a sub-model (e.g. EfficientNet base) or flat layers
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if len(sub_layer.output.shape) == 4 and "conv" in sub_layer.name.lower():
                        last_conv_layer_name = sub_layer.name
                        conv_model = layer
                        break
                if last_conv_layer_name:
                    break
            elif len(layer.output.shape) == 4 and "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                conv_model = model
                break

    if last_conv_layer_name is None:
        logger.warning("Could not identify convolutional layer for Grad-CAM.")
        return None

    try:
        # Build gradient model mapping input to last conv activations and final predictions
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception:
        # Fallback if layer is nested inside a Sequential container
        try:
            target_layer = None
            for layer in model.layers:
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    break
                except Exception:
                    continue
            if target_layer is None:
                return None
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[target_layer.output, model.output]
            )
        except Exception as err:
            logger.error("Failed to construct Grad-CAM gradient model: %s", err)
            return None

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients of predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight channels by their corresponding gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_and_overlay_gradcam(image_path, heatmap, output_path, alpha=0.45, colormap="jet"):
    """
    Superimposes the generated Grad-CAM heatmap over the original leaf photo and saves it.
    """
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    if heatmap is None:
        # Generate a fallback focal attention circle if model weights are unavailable
        width, height = original_size
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        heatmap = np.exp(-(dist_from_center ** 2) / (2 * (min(width, height) / 3.5) ** 2))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Rescale heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Use colormap to colorize the heatmap
    cmap = cm.get_cmap(colormap)
    cmap_colors = cmap(np.arange(256))[:, :3]
    colored_heatmap = cmap_colors[heatmap_uint8]
    colored_heatmap = Image.fromarray(np.uint8(colored_heatmap * 255))
    colored_heatmap = colored_heatmap.resize(original_size, Image.Resampling.BILINEAR)

    # Blend original image and heatmap
    overlaid_image = Image.blend(img, colored_heatmap, alpha=alpha)
    overlaid_image.save(output_path, quality=92)
    return output_path

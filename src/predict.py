"""Inference module with a safe demonstration fallback before model training."""
from pathlib import Path
import random

CLASSES = ["Healthy", "Bacterial blight", "Downy mildew", "Frogeye leaf spot", "Septoria brown spot", "Soybean rust"]


class DiseasePredictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists():
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, image_path):
        if self.model is not None:
            import numpy as np
            import tensorflow as tf
            image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
            image_array = tf.keras.utils.img_to_array(image)
            probabilities = self.model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
            index = int(np.argmax(probabilities))
            confidence = round(float(probabilities[index]) * 100, 1)
            top_predictions = sorted(
                [{"disease": CLASSES[i], "confidence": round(float(value) * 100, 1)} for i, value in enumerate(probabilities)],
                key=lambda item: item["confidence"], reverse=True
            )[:3]
            return {"disease": CLASSES[index], "confidence": confidence, "top_predictions": top_predictions,
                    "note": "AI-assisted result. Confirm chemical-control decisions with a local agricultural expert."}

        # Makes the interface testable before a trained model is saved. It never claims to be a real diagnosis.
        selection = random.choice(CLASSES)
        confidence = round(random.uniform(62, 88), 1)
        return {"disease": selection, "confidence": confidence,
                "top_predictions": [{"disease": selection, "confidence": confidence}],
                "note": "Demo mode: train and place the model in models/ for a real AI prediction. Confirm advice with an agricultural expert."}

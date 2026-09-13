from flask import Flask, jsonify, render_template, request
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import json
import sqlite3

from src.predict import DiseasePredictor

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATABASE = BASE_DIR / "data" / "soycare.db"
UPLOAD_DIR.mkdir(exist_ok=True)
DATABASE.parent.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
predictor = DiseasePredictor(BASE_DIR / "models" / "soybean_disease_model.keras")


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialise_database():
    with get_db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, detected_at TEXT NOT NULL,
            filename TEXT NOT NULL, disease TEXT NOT NULL, confidence REAL NOT NULL,
            risk_level TEXT NOT NULL
        )""")


def recommendations_for(disease):
    with open(BASE_DIR / "knowledge_base" / "disease_recommendations.json") as file:
        recommendations = json.load(file)
    return recommendations.get(disease, recommendations["Unknown"])


@app.route("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    image = request.files.get("image")
    if not image or not image.filename:
        return jsonify({"error": "Please choose a soybean leaf image."}), 400
    if not image.mimetype.startswith("image/"):
        return jsonify({"error": "Only image files are accepted."}), 400

    extension = Path(image.filename).suffix.lower() or ".jpg"
    filename = f"{uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / filename
    image.save(saved_path)

    result = predictor.predict(saved_path)
    recommendation = recommendations_for(result["disease"])
    risk = "High" if result["confidence"] >= 85 and result["disease"] != "Healthy" else "Monitor"
    with get_db() as db:
        db.execute(
            "INSERT INTO detections (detected_at, filename, disease, confidence, risk_level) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), filename, result["disease"], result["confidence"], risk),
        )
    return jsonify({**result, "risk_level": risk, "recommendation": recommendation})


@app.get("/api/dashboard")
def dashboard():
    with get_db() as db:
        rows = db.execute("SELECT disease, COUNT(*) AS count FROM detections GROUP BY disease ORDER BY count DESC").fetchall()
        total = db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        average = db.execute("SELECT ROUND(AVG(confidence), 1) FROM detections").fetchone()[0] or 0
    return jsonify({"total_detections": total, "average_confidence": average, "disease_counts": [dict(row) for row in rows]})


if __name__ == "__main__":
    initialise_database()
    app.run(debug=True)

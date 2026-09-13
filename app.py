"""
SoyCare AI - Soybean Disease Detection Flask Server
Production-grade backend supporting image classification, Grad-CAM attention heatmaps,
SQLite diagnosis logging, historical search & filtering, CSV export, and disease knowledge API.
"""
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import json
import sqlite3
import csv
import io
import base64
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    Response,
    abort
)

from config import (
    BASE_DIR,
    UPLOAD_DIR,
    HEATMAP_DIR,
    DATABASE_PATH,
    KNOWLEDGE_BASE_PATH,
    OUTPUTS_DIR,
    MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS,
    HIGH_RISK_CONFIDENCE_THRESHOLD,
    CLASSES,
    logger
)
from src.predict import DiseasePredictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = "soycare-prototype-secret-key-2026"

# Initialize single instance of DiseasePredictor
predictor = DiseasePredictor()


def get_db():
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_database():
    """Creates tables, indexes, and runs automatic migrations if columns are missing."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                notes TEXT,
                source TEXT DEFAULT 'upload'
            )
        """)
        # Schema migration check for existing databases
        cursor = db.execute("PRAGMA table_info(detections);")
        columns = [row[1] for row in cursor.fetchall()]
        if "notes" not in columns:
            db.execute("ALTER TABLE detections ADD COLUMN notes TEXT;")
        if "source" not in columns:
            db.execute("ALTER TABLE detections ADD COLUMN source TEXT DEFAULT 'upload';")

        db.execute("CREATE INDEX IF NOT EXISTS idx_detections_date ON detections(detected_at DESC)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_detections_disease ON detections(disease)")
    logger.info("Database initialized and migrated at %s", DATABASE_PATH)


def load_knowledge_base():
    """Loads disease management guidance from JSON."""
    if not KNOWLEDGE_BASE_PATH.exists():
        logger.warning("Knowledge base JSON not found at %s", KNOWLEDGE_BASE_PATH)
        return {}
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def recommendations_for(disease_name):
    """Retrieves agronomic recommendations for a given disease."""
    kb = load_knowledge_base()
    return kb.get(disease_name, kb.get("Unknown", {
        "symptoms": "No profile found for this disease class.",
        "immediate_action": ["Consult an agricultural extension officer."],
        "chemical_control": ["Seek laboratory diagnosis."],
        "cultural_management": ["Maintain regular field inspection."],
        "loss_minimization": "Record observations."
    }))


# --------------------------------------------------------------------------
# Static Image Serving Routes
# --------------------------------------------------------------------------
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


# --------------------------------------------------------------------------
# Frontend View
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------------------------------
# REST API Endpoints
# --------------------------------------------------------------------------
@app.post("/api/predict")
def predict_endpoint():
    """
    Accepts an uploaded image file or base64 image payload (e.g. from camera stream),
    runs prediction with Grad-CAM heatmap generation, and logs diagnosis.
    """
    saved_filename = None
    source = "upload"

    # Handle multipart form file upload
    if "image" in request.files:
        image_file = request.files["image"]
        if not image_file or not image_file.filename:
            return jsonify({"error": "No image file selected. Please select a soybean leaf photograph."}), 400

        ext = Path(image_file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        saved_filename = f"{uuid4().hex}{ext}"
        target_path = UPLOAD_DIR / saved_filename
        image_file.save(target_path)

    # Handle base64 camera capture payload
    elif request.is_json and "image_base64" in request.json:
        data_uri = request.json["image_base64"]
        try:
            header, encoded = data_uri.split(",", 1) if "," in data_uri else ("", data_uri)
            image_data = base64.b64decode(encoded)
            saved_filename = f"{uuid4().hex}.jpg"
            target_path = UPLOAD_DIR / saved_filename
            with open(target_path, "wb") as f:
                f.write(image_data)
            source = "webcam"
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
    else:
        return jsonify({"error": "Please provide an image file or camera capture."}), 400

    # Perform ML Inference + Grad-CAM
    try:
        prediction_result = predictor.predict(target_path, filename=saved_filename)
    except Exception as err:
        logger.error("Inference failed: %s", err)
        return jsonify({"error": f"Model inference failed: {str(err)}"}), 500

    disease = prediction_result["disease"]
    confidence = prediction_result["confidence"]
    recommendation = recommendations_for(disease)

    # Compute risk level
    if disease == "Healthy":
        risk_level = "Healthy"
    elif confidence >= HIGH_RISK_CONFIDENCE_THRESHOLD:
        risk_level = "High Risk"
    else:
        risk_level = "Moderate"

    # Persist record in SQLite
    iso_time = datetime.now(timezone.utc).isoformat()
    with get_db() as db:
        cursor = db.execute(
            """INSERT INTO detections
               (detected_at, filename, disease, confidence, risk_level, notes, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (iso_time, saved_filename, disease, confidence, risk_level, prediction_result.get("note", ""), source)
        )
        record_id = cursor.lastrowid

    return jsonify({
        "id": record_id,
        "detected_at": iso_time,
        "filename": saved_filename,
        "image_url": f"/uploads/{saved_filename}",
        "disease": disease,
        "confidence": confidence,
        "risk_level": risk_level,
        "top_predictions": prediction_result["top_predictions"],
        "heatmap_url": prediction_result.get("heatmap_url"),
        "is_demo": prediction_result.get("is_demo", False),
        "is_low_confidence": prediction_result.get("is_low_confidence", False),
        "recommendation": recommendation,
        "note": prediction_result.get("note", "")
    })


@app.get("/api/history")
def history_endpoint():
    """
    Returns paginated scan history with search and filtering by risk level or disease.
    Query params: page, limit, disease, risk, search
    """
    page = max(1, int(request.args.get("page", 1)))
    limit = min(50, max(1, int(request.args.get("limit", 10))))
    offset = (page - 1) * limit

    disease_filter = request.args.get("disease", "").strip()
    risk_filter = request.args.get("risk", "").strip()
    search_query = request.args.get("search", "").strip()

    query_clauses = []
    params = []

    if disease_filter:
        query_clauses.append("disease = ?")
        params.append(disease_filter)
    if risk_filter:
        query_clauses.append("risk_level = ?")
        params.append(risk_filter)
    if search_query:
        query_clauses.append("(disease LIKE ? OR notes LIKE ? OR filename LIKE ?)")
        search_pattern = f"%{search_query}%"
        params.extend([search_pattern, search_pattern, search_pattern])

    where_sql = f"WHERE {' AND '.join(query_clauses)}" if query_clauses else ""

    with get_db() as db:
        count_query = f"SELECT COUNT(*) FROM detections {where_sql}"
        total_records = db.execute(count_query, params).fetchone()[0]

        select_query = f"""
            SELECT id, detected_at, filename, disease, confidence, risk_level, notes, source
            FROM detections
            {where_sql}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        rows = db.execute(select_query, params + [limit, offset]).fetchall()

    return jsonify({
        "page": page,
        "limit": limit,
        "total_records": total_records,
        "total_pages": (total_records + limit - 1) // limit if total_records > 0 else 1,
        "records": [dict(r) for r in rows]
    })


@app.get("/api/history/export")
def export_history_csv():
    """Exports all scan records as a downloadable CSV spreadsheet."""
    with get_db() as db:
        rows = db.execute("""
            SELECT id, detected_at, filename, disease, confidence, risk_level, notes, source
            FROM detections
            ORDER BY id DESC
        """).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp (UTC)", "Filename", "Disease", "Confidence (%)", "Risk Level", "Notes", "Input Source"])

    for row in rows:
        writer.writerow([
            row["id"],
            row["detected_at"],
            row["filename"],
            row["disease"],
            row["confidence"],
            row["risk_level"],
            row["notes"],
            row["source"]
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=soycare_scan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )


@app.delete("/api/history/<int:record_id>")
def delete_history_record(record_id):
    """Deletes a specific scan record from the database."""
    with get_db() as db:
        db.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    return jsonify({"success": True, "deleted_id": record_id})


@app.get("/api/dashboard")
def dashboard_endpoint():
    """Returns analytics metrics and class-level detection summary."""
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        avg_conf = db.execute("SELECT ROUND(AVG(confidence), 1) FROM detections").fetchone()[0] or 0.0
        high_risk_count = db.execute("SELECT COUNT(*) FROM detections WHERE risk_level = 'High Risk'").fetchone()[0]
        disease_counts = db.execute("""
            SELECT disease, COUNT(*) AS count
            FROM detections
            GROUP BY disease
            ORDER BY count DESC
        """).fetchall()

        recent = db.execute("""
            SELECT id, detected_at, filename, disease, confidence, risk_level
            FROM detections
            ORDER BY id DESC
            LIMIT 5
        """).fetchall()

    return jsonify({
        "total_detections": total,
        "average_confidence": avg_conf,
        "high_risk_detections": high_risk_count,
        "disease_counts": [dict(r) for r in disease_counts],
        "recent_scans": [dict(r) for r in recent]
    })


@app.get("/api/diseases")
def diseases_kb_endpoint():
    """Returns the full agricultural disease knowledge base."""
    return jsonify(load_knowledge_base())


# --------------------------------------------------------------------------
# Error Handlers
# --------------------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the 16 MB maximum upload limit."}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "The requested resource was not found."}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    initialise_database()
    logger.info("Starting SoyCare AI server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)

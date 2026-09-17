"""
Integration tests for SoyCare AI Flask REST API endpoints and database operations.
"""
import io
import base64
from pathlib import Path


def test_index_page(client):
    """Verifies that the homepage renders successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"SoyCare" in response.data or b"Soybean" in response.data


def test_predict_file_upload(client, sample_leaf_image):
    """Verifies POST /api/predict with a multipart file upload."""
    with open(sample_leaf_image, "rb") as f:
        data = {
            "image": (io.BytesIO(f.read()), "test_leaf.jpg")
        }
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    json_data = response.get_json()
    assert "id" in json_data
    assert "disease" in json_data
    assert "confidence" in json_data
    assert "recommendation" in json_data
    assert "heatmap_url" in json_data
    assert json_data["is_valid_leaf"] is True


def test_predict_base64_capture(client, sample_leaf_image):
    """Verifies POST /api/predict with a camera base64 string payload."""
    with open(sample_leaf_image, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{encoded}"

    response = client.post("/api/predict", json={"image_base64": data_uri})
    assert response.status_code == 200
    json_data = response.get_json()
    assert "id" in json_data
    assert "disease" in json_data


def test_predict_missing_payload(client):
    """Verifies that submitting an empty request yields a 400 bad request."""
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert "error" in json_data


def test_history_and_pagination(client, sample_leaf_image):
    """Verifies that scans appear in /api/history and support pagination."""
    # Create two test scans
    with open(sample_leaf_image, "rb") as f:
        img_bytes = f.read()

    client.post("/api/predict", data={"image": (io.BytesIO(img_bytes), "leaf1.jpg")}, content_type="multipart/form-data")
    client.post("/api/predict", data={"image": (io.BytesIO(img_bytes), "leaf2.jpg")}, content_type="multipart/form-data")

    response = client.get("/api/history?page=1&limit=10")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["total_records"] >= 2
    assert len(json_data["records"]) >= 2


def test_history_rejects_invalid_pagination(client):
    """Verifies malformed pagination input returns a client error."""
    response = client.get("/api/history?page=not-a-number&limit=10")

    assert response.status_code == 400
    assert "page and limit" in response.get_json()["error"]


def test_history_csv_export(client, sample_leaf_image):
    """Verifies that /api/history/export returns a valid CSV file stream."""
    with open(sample_leaf_image, "rb") as f:
        client.post("/api/predict", data={"image": (io.BytesIO(f.read()), "export_test.jpg")}, content_type="multipart/form-data")

    response = client.get("/api/history/export")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"ID,Timestamp (UTC),Filename,Disease,Confidence (%),Risk Level,Notes,Input Source" in response.data


def test_delete_history_record(client, sample_leaf_image):
    """Verifies that DELETE /api/history/<id> removes the record from SQLite."""
    with open(sample_leaf_image, "rb") as f:
        res = client.post("/api/predict", data={"image": (io.BytesIO(f.read()), "del_test.jpg")}, content_type="multipart/form-data")
    record_id = res.get_json()["id"]

    del_res = client.delete(f"/api/history/{record_id}")
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True


def test_dashboard_endpoint(client):
    """Verifies /api/dashboard summary metrics."""
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "total_detections" in json_data
    assert "average_confidence" in json_data
    assert "disease_counts" in json_data


def test_diseases_knowledge_base_endpoint(client):
    """Verifies /api/diseases returns full advisory catalog."""
    response = client.get("/api/diseases")
    assert response.status_code == 200
    json_data = response.get_json()
    assert "Healthy" in json_data
    assert "Soybean rust" in json_data

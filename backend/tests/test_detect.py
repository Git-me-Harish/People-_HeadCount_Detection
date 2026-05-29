from io import BytesIO

from PIL import Image


def _png_bytes(w: int = 64, h: int = 64) -> bytes:
    img = Image.new("RGB", (w, h), color=(220, 220, 220))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detect_image_returns_zero_when_detector_disabled(auth_client):
    """With ENABLE_DETECTOR=false (test config), result should be empty but well-formed."""
    files = {"file": ("test.png", _png_bytes(), "image/png")}
    r = auth_client.post("/api/v1/detect/image", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["person_count"] == 0
    assert body["detections"] == []
    assert body["width"] == 64
    assert body["height"] == 64


def test_detect_image_rejects_bad_mime(auth_client):
    files = {"file": ("foo.txt", b"not an image", "text/plain")}
    r = auth_client.post("/api/v1/detect/image", files=files)
    assert r.status_code == 400


def test_detector_status_endpoint(auth_client):
    r = auth_client.get("/api/v1/detect/status")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body
    assert "model_path" in body

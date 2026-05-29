from io import BytesIO

from PIL import Image


def _png() -> bytes:
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_analytics_summary_empty(auth_client):
    r = auth_client.get("/api/v1/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_detections"] >= 0


def test_analytics_after_detection(auth_client):
    auth_client.post(
        "/api/v1/detect/image",
        files={"file": ("a.png", _png(), "image/png")},
    )
    r = auth_client.get("/api/v1/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_detections"] >= 1

    r2 = auth_client.get("/api/v1/analytics/records")
    assert r2.status_code == 200
    assert isinstance(r2.json(), list)
    assert len(r2.json()) >= 1


def test_analytics_timeseries(auth_client):
    r = auth_client.get("/api/v1/analytics/timeseries?days=1&bucket_minutes=60")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

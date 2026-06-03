"""Tests for industry templates endpoint."""

from fastapi.testclient import TestClient


def test_list_templates(client: TestClient, auth_headers: dict) -> None:
    resp = client.get("/api/v1/templates", headers=auth_headers)
    assert resp.status_code == 200
    templates = resp.json()
    assert len(templates) == 8
    verticals = {t["vertical"] for t in templates}
    assert "religious" in verticals
    assert "transit" in verticals


def test_apply_template(client: TestClient, auth_headers: dict) -> None:
    resp = client.post("/api/v1/templates/hospital/apply", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "cameras_created" in data
    assert "alerts_created" in data
    # Idempotent — applying again creates 0 new items
    resp2 = client.post("/api/v1/templates/hospital/apply", headers=auth_headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["cameras_created"]) == 0
    assert len(data2["alerts_created"]) == 0


def test_apply_unknown_template(client: TestClient, auth_headers: dict) -> None:
    resp = client.post("/api/v1/templates/nonexistent/apply", headers=auth_headers)
    # Vertical enum validation
    assert resp.status_code == 422

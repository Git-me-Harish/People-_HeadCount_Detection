"""Tests for public status page endpoints."""

from fastapi.testclient import TestClient


def test_create_public_page(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/api/v1/public/pages",
        json={
            "slug": "test-temple",
            "title": "Temple Queue",
            "camera_ids": [],
            "show_heatmap": False,
            "brand_color": "#6366f1",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["slug"] == "test-temple"
    assert data["title"] == "Temple Queue"


def test_duplicate_slug_rejected(client: TestClient, auth_headers: dict) -> None:
    client.post(
        "/api/v1/public/pages",
        json={
            "slug": "unique-slug-x",
            "title": "A",
            "camera_ids": [],
            "show_heatmap": False,
            "brand_color": "#6366f1",
        },
        headers=auth_headers,
    )
    resp = client.post(
        "/api/v1/public/pages",
        json={
            "slug": "unique-slug-x",
            "title": "B",
            "camera_ids": [],
            "show_heatmap": False,
            "brand_color": "#6366f1",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_public_view_no_auth(client: TestClient, auth_headers: dict) -> None:
    client.post(
        "/api/v1/public/pages",
        json={
            "slug": "public-test-123",
            "title": "Live Feed",
            "camera_ids": [],
            "show_heatmap": False,
            "brand_color": "#6366f1",
        },
        headers=auth_headers,
    )
    # No auth header — should work
    resp = client.get("/api/v1/public/public-test-123")
    assert resp.status_code == 200
    assert resp.json()["slug"] == "public-test-123"


def test_public_view_not_found(client: TestClient) -> None:
    resp = client.get("/api/v1/public/nonexistent-slug-zzz")
    assert resp.status_code == 404

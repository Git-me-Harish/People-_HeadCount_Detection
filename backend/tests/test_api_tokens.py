"""Tests for API token CRUD."""
from fastapi.testclient import TestClient


def test_create_and_list_token(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/api/v1/api-tokens",
        json={"name": "CI Integration", "scopes": "read:all"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "CI Integration"
    assert data["prefix"].startswith("ps_")
    assert "full_token" in data  # one-time reveal
    assert data["full_token"].startswith("ps_")

    # List should show token (without full_token)
    list_resp = client.get("/api/v1/api-tokens", headers=auth_headers)
    assert list_resp.status_code == 200
    tokens = list_resp.json()
    assert len(tokens) >= 1
    assert "full_token" not in tokens[0]


def test_revoke_token(client: TestClient, auth_headers: dict) -> None:
    create_resp = client.post(
        "/api/v1/api-tokens",
        json={"name": "Temp", "scopes": "read:all"},
        headers=auth_headers,
    )
    token_id = create_resp.json()["id"]
    del_resp = client.delete(f"/api/v1/api-tokens/{token_id}", headers=auth_headers)
    assert del_resp.status_code == 204


def test_revoke_nonexistent(client: TestClient, auth_headers: dict) -> None:
    resp = client.delete("/api/v1/api-tokens/99999", headers=auth_headers)
    assert resp.status_code == 404

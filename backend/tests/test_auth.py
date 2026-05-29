def test_register_and_login(client):
    payload = {
        "email": "auth-user@example.com",
        "full_name": "Auth User",
        "password": "supersecret1",
        "organization_name": "AuthOrg",
    }
    r = client.post("/api/v1/auth/register", json=payload)
    assert r.status_code == 201, r.text
    token = r.json()["access_token"]
    assert token

    # /me with token
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == payload["email"]

    # login flow
    r2 = client.post(
        "/api/v1/auth/login",
        json={"email": payload["email"], "password": payload["password"]},
    )
    assert r2.status_code == 200
    assert r2.json()["access_token"]


def test_duplicate_email(client):
    payload = {
        "email": "dup@example.com",
        "full_name": "Dup",
        "password": "supersecret1",
        "organization_name": "DupOrg",
    }
    r1 = client.post("/api/v1/auth/register", json=payload)
    assert r1.status_code == 201
    r2 = client.post("/api/v1/auth/register", json=payload)
    assert r2.status_code == 400


def test_login_invalid(client):
    r = client.post(
        "/api/v1/auth/login",
        json={"email": "nope@example.com", "password": "wrong"},
    )
    assert r.status_code == 401


def test_protected_requires_auth(client):
    r = client.get("/api/v1/cameras")
    assert r.status_code == 401

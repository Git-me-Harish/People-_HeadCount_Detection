def test_alert_crud(auth_client):
    r = auth_client.post(
        "/api/v1/alerts",
        json={"name": "Crowd > 50", "threshold": 50, "webhook_url": None},
    )
    assert r.status_code == 201
    alert_id = r.json()["id"]

    r2 = auth_client.get("/api/v1/alerts")
    assert any(a["id"] == alert_id for a in r2.json())

    r3 = auth_client.patch(f"/api/v1/alerts/{alert_id}", json={"threshold": 100})
    assert r3.json()["threshold"] == 100

    r4 = auth_client.delete(f"/api/v1/alerts/{alert_id}")
    assert r4.status_code == 204

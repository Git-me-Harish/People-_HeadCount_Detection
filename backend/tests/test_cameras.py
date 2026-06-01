def test_camera_crud(auth_client):
    r = auth_client.post(
        "/api/v1/cameras",
        json={"name": "Lobby Cam", "location": "Lobby", "stream_url": "rtsp://x/y"},
    )
    assert r.status_code == 201, r.text
    cam = r.json()
    assert cam["name"] == "Lobby Cam"
    cam_id = cam["id"]

    r2 = auth_client.get("/api/v1/cameras")
    assert r2.status_code == 200
    assert any(c["id"] == cam_id for c in r2.json())

    r3 = auth_client.patch(f"/api/v1/cameras/{cam_id}", json={"location": "Main Entrance"})
    assert r3.status_code == 200
    assert r3.json()["location"] == "Main Entrance"

    r4 = auth_client.delete(f"/api/v1/cameras/{cam_id}")
    assert r4.status_code == 204

    r5 = auth_client.get(f"/api/v1/cameras/{cam_id}")
    assert r5.status_code == 404

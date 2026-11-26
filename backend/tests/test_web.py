from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import mount_frontend


def test_health_reports_status_without_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mongo"] == "ok"
    assert set(body) >= {"device", "whisper_model", "ffprobe", "frontend"}


def test_docs_are_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_security_headers_on_api_responses(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "media-src 'self' blob:" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"


def test_frontend_is_served_and_api_keeps_priority(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>UI</title>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")

    app = FastAPI()

    @app.get("/recordings/ping")
    def ping() -> dict:
        return {"api": True}

    assert mount_frontend(app, tmp_path) is True
    c = TestClient(app)
    assert "<title>UI</title>" in c.get("/").text
    assert c.get("/assets/app.js").status_code == 200
    assert c.get("/recordings/ping").json() == {"api": True}  # not shadowed by the static mount


def test_frontend_not_mounted_when_not_built(tmp_path):
    assert mount_frontend(FastAPI(), tmp_path) is False

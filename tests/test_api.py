"""API 集成测试（FastAPI TestClient，Mock 模式离线）。"""

from fastapi.testclient import TestClient

from app.server import app, _sessions, _stats

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mock_mode"] is True  # CI 环境无 API key


def test_chat_returns_structured_reply():
    resp = client.post("/api/chat", json={"text": "去日本5天漫游包几钱？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "roaming_inquiry"
    assert body["language"] == "zh-CN"
    assert "$98" in body["text"]
    assert body["session_id"]


def test_session_state_persists_across_calls():
    sid = "api-e2e-1"
    r1 = client.post("/api/chat", json={"text": "公司做跨境电商想上云专线", "session_id": sid})
    assert r1.status_code == 200
    r2 = client.post("/api/chat", json={"text": "我电话 91234567", "session_id": sid})
    body = r2.json()
    assert "LD-" in body["text"]
    assert "912****67" in body["text"]
    assert sid in _sessions


def test_lang_override_en():
    resp = client.post("/api/chat", json={"text": "roaming?", "lang": "en"})
    assert resp.json()["language"] == "en"


def test_stats_endpoint():
    client.post("/api/chat", json={"text": "你好"})
    body = client.get("/api/stats").json()
    assert body["total_turns"] >= 1
    assert "by_language" in body and "by_intent" in body


def test_log_written_masked():
    import json

    from app.config import LOG_DIR

    log = LOG_DIR / "sessions.jsonl"
    if log.exists():
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[-1])
        assert "user" in record and "intent" in record

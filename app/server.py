"""FastAPI 服务：/api/chat 对话端点 + 会话日志（脱敏）+ 统计端点。"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import LOG_DIR, load_settings
from .guardrails import mask_pii
from .orchestrator import Orchestrator, Session

settings = load_settings()
orch = Orchestrator(settings)

app = FastAPI(title="Trilingual Customer Service Agent", version="0.1.0")
_sessions: dict[str, Session] = {}
_stats = Counter()  # (intent, language) 计数


class ChatRequest(BaseModel):
    text: str
    session_id: str | None = None
    lang: str | None = None  # auto / zh-HK / zh-CN / en


class ChatReply(BaseModel):
    session_id: str
    text: str
    language: str
    intent: str
    tools: list[str]
    citations: list[str]
    trace: list[dict]
    handoff: bool
    mock_mode: bool


def _append_log(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "sessions.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent.parent / "templates" / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "mock_mode": settings.mock})


@app.post("/api/chat", response_model=ChatReply)
def chat(req: ChatRequest) -> ChatReply:
    session_id = req.session_id or uuid.uuid4().hex[:12]
    session = _sessions.setdefault(session_id, Session(session_id=session_id))
    reply = orch.handle(session, req.text, req.lang if req.lang != "auto" else None)
    _stats[(reply.intent, reply.language)] += 1
    _append_log(
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "user": mask_pii(req.text),
            "assistant": reply.text,
            "language": reply.language,
            "intent": reply.intent,
            "tools": reply.tools,
            "mock_mode": settings.mock,
        }
    )
    return ChatReply(
        session_id=session_id,
        text=reply.text,
        language=reply.language,
        intent=reply.intent,
        tools=reply.tools,
        citations=reply.citations,
        trace=reply.trace,
        handoff=reply.handoff,
        mock_mode=settings.mock,
    )


@app.get("/api/stats")
def stats() -> JSONResponse:
    by_intent = Counter()
    by_lang = Counter()
    for (intent, lang), n in _stats.items():
        by_intent[intent] += n
        by_lang[lang] += n
    return JSONResponse(
        {
            "total_turns": sum(_stats.values()),
            "by_intent": dict(by_intent),
            "by_language": dict(by_lang),
            "mock_mode": settings.mock,
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()

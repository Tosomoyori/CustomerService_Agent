"""配置加载：环境变量 + 极简 .env 解析（不引入第三方依赖）。

约定前缀 APP_：
  APP_LLM_BASE_URL / APP_LLM_API_KEY / APP_LLM_MODEL   OpenAI 兼容接口配置
  APP_MOCK            1=强制离线演示  0=强制真实调用  空=自动（无 key 即 Mock）
  APP_HOST / APP_PORT  服务监听
  APP_RAG_TOP_K / APP_RAG_MIN_SCORE  检索参数
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb"
RUNTIME_DIR = DATA_DIR / "runtime"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"


def _load_dotenv(path: Path) -> None:
    """把 .env 中的 KEY=VALUE 写入 os.environ（已存在的环境变量优先）。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    mock: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    rag_top_k: int = 3
    rag_min_score: float = 0.6


def load_settings() -> Settings:
    base_url = os.environ.get("APP_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("APP_LLM_API_KEY", "").strip()
    model = os.environ.get("APP_LLM_MODEL", "").strip() or "deepseek-chat"
    forced = os.environ.get("APP_MOCK", "").strip()
    if forced == "1":
        mock = True
    elif forced == "0":
        mock = False
    else:
        # 自动判断：没有 key 就进 Mock 模式，保证 clone 下来零配置可跑
        mock = not api_key
    return Settings(
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=model,
        mock=mock,
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("APP_PORT", "8000")),
        rag_top_k=int(os.environ.get("APP_RAG_TOP_K", "3")),
        rag_min_score=float(os.environ.get("APP_RAG_MIN_SCORE", "0.6")),
    )

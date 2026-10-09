"""LLM 客户端：OpenAI 兼容接口（DeepSeek / Qwen 等 OpenAI 兼容服务均可切换）。

本项目的 LLM 是「增强层」而非「依赖层」：
- Mock 模式（默认，无 API key 自动启用）：全部流程由规则 + 模板 + 抽取式回答完成，
  离线可跑、演示可复现，评测不产生 API 费用；
- 真实模式：路由 LLM 兜底、话术由模型在确定性工具结果上生成，
  生成后仍要过金额溯源护栏，越界即回退确定性模板。
"""

from __future__ import annotations

from .config import Settings


class LLMNotAvailable(RuntimeError):
    pass


class OpenAICompatLLM:
    """OpenAI 兼容 Chat Completions 客户端（延迟导入 openai 包）。"""

    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise LLMNotAvailable("missing APP_LLM_API_KEY")
        self._settings = settings
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # 延迟导入：Mock 模式不依赖该包

            self._client = OpenAI(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url or None,
            )
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 512,
             tools: list | None = None) -> str:
        client = self._ensure_client()
        kwargs: dict = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def get_llm(settings: Settings):
    """按配置返回 LLM 实例；Mock 模式返回 None（流程走确定性路径）。"""
    if settings.mock:
        return None
    try:
        return OpenAICompatLLM(settings)
    except LLMNotAvailable:
        return None

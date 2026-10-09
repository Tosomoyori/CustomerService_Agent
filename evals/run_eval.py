"""离线评测：语言识别准确率 / 意图路由准确率 / 检索 HitRate 与 MRR / 护栏自检。

全部指标在 Mock 模式（无外部 API）下可复现：

    uv run python -m evals.run_eval
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import REPORT_DIR
from app.guardrails import extract_amounts, mask_pii, validate_amounts
from app.intent import route
from app.language import detect_language, normalize
from app.rag import search_kb

EVAL_DIR = Path(__file__).resolve().parent


def _load(name: str) -> list[dict]:
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def eval_language(cases: list[dict]) -> tuple[float, list[dict]]:
    ok = 0
    fails = []
    for c in cases:
        got = detect_language(c["text"])
        if got == c["lang"]:
            ok += 1
        else:
            fails.append({"text": c["text"], "expected": c["lang"], "got": got})
    return ok / len(cases), fails


def eval_intent(cases: list[dict]) -> tuple[float, list[dict]]:
    ok = 0
    fails = []
    for c in cases:
        got = route(c["text"]).intent
        if got == c["intent"]:
            ok += 1
        else:
            fails.append({"text": c["text"], "expected": c["intent"], "got": got})
    return ok / len(cases), fails


def eval_retrieval(cases: list[dict], top_k: int = 3) -> dict:
    hit1 = 0
    hit3 = 0
    rr_sum = 0.0
    fails = []
    for c in cases:
        hits = search_kb(normalize(c["query"]), top_k=top_k)
        docs = [h["doc"] for h in hits]
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h["doc"] == c["doc"]:
                rr = 1.0 / rank
                break
        rr_sum += rr
        if rr >= 0.5:
            hit1 += 1
        if rr > 0:
            hit3 += 1
        else:
            fails.append({"query": c["query"], "expected": c["doc"], "got": docs})
    n = len(cases)
    return {
        "hit@1": hit1 / n,
        "hit@3": hit3 / n,
        "mrr": rr_sum / n,
        "fails": fails,
        "n": n,
    }


def eval_guardrails() -> dict:
    """护栏自检：PII 脱敏与金额溯源。"""
    checks = []

    def add(name: str, passed: bool, detail: str = ""):
        checks.append({"name": name, "passed": passed, "detail": detail})

    masked = mask_pii("我係9123 4567，身份證A123456(7)，email a@b.com")
    add("PII: 香港电话脱敏", "912****" in masked and "67" in masked, masked)
    add("PII: 身份证脱敏", "A******" in masked and "(7)" in masked, masked)
    add("PII: 邮箱脱敏", "a***@b.com" in masked, masked)
    hotline = mask_pii("请致电客服热线 2122 1188")
    add("PII: 公开热线不脱敏", "2122 1188" in hotline, hotline)

    violations = validate_amounts("套餐系$99一个月", ["$158/月", "$258/月"])
    add("金额溯源: 越界金额被捕获", violations == [99.0], str(violations))
    ok = validate_amounts("学生计划系$99一个月", ["內地學生計劃 $99/月"])
    add("金额溯源: 有出处的金额通过", ok == [], str(ok))
    amounts = extract_amounts("月費$128，漫游包$98.5")
    add("金额抽取", amounts == {128.0, 98.5}, str(amounts))

    passed = sum(1 for c in checks if c["passed"])
    return {"passed": passed, "total": len(checks), "checks": checks}


def main() -> None:
    intent_cases = _load("intent_set.json")
    retrieval_cases = _load("retrieval_set.json")

    lang_acc, lang_fails = eval_language(intent_cases)
    intent_acc, intent_fails = eval_intent(intent_cases)
    retrieval = eval_retrieval(retrieval_cases)
    guard = eval_guardrails()

    lines = [
        "# 三语智能客服 Agent 离线评测报告",
        "",
        f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 语言识别：**{lang_acc:.1%}**（{len(intent_cases)} 例，三语混合）",
        f"- 意图路由：**{intent_acc:.1%}**（{len(intent_cases)} 例，规则路由，无 LLM）",
        f"- 检索 Hit@1：**{retrieval['hit@1']:.1%}**，Hit@3：**{retrieval['hit@3']:.1%}**，MRR：**{retrieval['mrr']:.3f}**（{retrieval['n']} 例，BM25）",
        f"- 护栏自检：**{guard['passed']}/{guard['total']}** 通过（PII 脱敏 + 金额溯源）",
        "",
        "## 失败明细",
        "",
    ]
    if lang_fails:
        lines.append("### 语言识别")
        lines += [f"- {f['text']} → 期望 {f['expected']}，实际 {f['got']}" for f in lang_fails]
    if intent_fails:
        lines.append("### 意图路由")
        lines += [f"- {f['text']} → 期望 {f['expected']}，实际 {f['got']}" for f in intent_fails]
    if retrieval["fails"]:
        lines.append("### 检索")
        lines += [f"- {f['query']} → 期望 {f['expected']}，实际 {f['got']}" for f in retrieval["fails"]]
    failed_checks = [c for c in guard["checks"] if not c["passed"]]
    if failed_checks:
        lines.append("### 护栏")
        lines += [f"- {c['name']}：{c['detail']}" for c in failed_checks]
    if not (lang_fails or intent_fails or retrieval["fails"] or failed_checks):
        lines.append("无失败用例。")

    report = "\n".join(lines)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()

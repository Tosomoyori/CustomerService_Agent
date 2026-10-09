"""工作流编排：语言识别 → 意图路由 → 槽位/工具 → 话术生成 → 护栏 → 回复。

工作流节点（对应 Dify 等低代码平台的节点映射见 docs/dify_workflow_mapping.md）：

  1 language_detect  三语识别（粤/普/英），用户显式选择优先
  2 intent_route     规则路由（真实模式下 LLM 兜底）
  3 slot_filling     企业方案/工单两类多轮流程的槽位管理
  4 tool_exec        结构化查询与写操作（金额唯一来源）
  5 compose          Mock=确定性模板 / 真实=模型在工具结果上组织话术
  6 guardrail        PII 脱敏、金额溯源校验（越界回退模板）、免责声明
  7 reply            返回结构化 Reply（含 trace，供前端展示推理链路）

Mock 模式下整条链路不依赖任何外部 API，可离线演示、可复现评测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Settings, load_settings
from .guardrails import ensure_disclaimer, mask_contact, mask_pii, validate_amounts
from .intent import (
    INTENT_CHITCHAT,
    INTENT_ENTERPRISE,
    INTENT_HANDOFF,
    INTENT_KB,
    INTENT_RECOMMEND,
    INTENT_ROAMING,
    INTENT_TICKET,
    RouteResult,
    route,
)
from .language import LANG_ZH_CN, LANG_ZH_HK, detect_language, normalize
from .llm import get_llm
from .prompts import build_generation_prompt
from .rag import extractive_answer, search_kb
from .tools import (
    check_coverage,
    create_lead,
    create_ticket,
    parse_days,
    query_plans,
    query_roaming_packs,
    recommend_plan,
    resolve_region,
    search_solutions,
    transfer_human,
)

_CONTACT_RE = re.compile(
    r"(?<!\d)([2-9]\d{3}[\s-]?\d{4})(?!\d)|(?<!\d)(1[3-9]\d{9})(?!\d)"
    r"|([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+)"
)
_REGION_NAMES = {
    "HK": {"zh-HK": "香港", "zh-CN": "香港", "en": "Hong Kong"},
    "CN": {"zh-HK": "內地", "zh-CN": "内地", "en": "Mainland China"},
    "MO": {"zh-HK": "澳門", "zh-CN": "澳门", "en": "Macau"},
    "JP": {"zh-HK": "日本", "zh-CN": "日本", "en": "Japan"},
    "KR": {"zh-HK": "韓國", "zh-CN": "韩国", "en": "Korea"},
    "TH": {"zh-HK": "泰國", "zh-CN": "泰国", "en": "Thailand"},
    "SG": {"zh-HK": "新加坡", "zh-CN": "新加坡", "en": "Singapore"},
    "MY": {"zh-HK": "馬來西亞", "zh-CN": "马来西亚", "en": "Malaysia"},
    "VN": {"zh-HK": "越南", "zh-CN": "越南", "en": "Vietnam"},
    "PH": {"zh-HK": "菲律賓", "zh-CN": "菲律宾", "en": "Philippines"},
    "ID": {"zh-HK": "印尼", "zh-CN": "印尼", "en": "Indonesia"},
    "UK": {"zh-HK": "英國", "zh-CN": "英国", "en": "the UK"},
    "FR": {"zh-HK": "法國", "zh-CN": "法国", "en": "France"},
    "DE": {"zh-HK": "德國", "zh-CN": "德国", "en": "Germany"},
    "IT": {"zh-HK": "意大利", "zh-CN": "意大利", "en": "Italy"},
    "CH": {"zh-HK": "瑞士", "zh-CN": "瑞士", "en": "Switzerland"},
    "ES": {"zh-HK": "西班牙", "zh-CN": "西班牙", "en": "Spain"},
    "US": {"zh-HK": "美國", "zh-CN": "美国", "en": "the US"},
    "CA": {"zh-HK": "加拿大", "zh-CN": "加拿大", "en": "Canada"},
    "AU": {"zh-HK": "澳洲", "zh-CN": "澳大利亚", "en": "Australia"},
    "NZ": {"zh-HK": "新西蘭", "zh-CN": "新西兰", "en": "New Zealand"},
    "EU": {"zh-HK": "歐洲", "zh-CN": "欧洲", "en": "Europe"},
}

_DECLINE_RE = re.compile(r"唔使|唔需要|不用了|不用|算了|先唔|以后再说|暫時|暂时不|no thanks|not now|later", re.IGNORECASE)


@dataclass
class Session:
    session_id: str
    state: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # 脱敏后的轮次 [(role, text)]


@dataclass
class Reply:
    text: str
    language: str
    intent: str
    tools: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    handoff: bool = False


def _region_name(code: str | None, lang: str) -> str:
    if not code:
        return {"zh-HK": "該目的地", "zh-CN": "该目的地", "en": "this destination"}[lang]
    return _REGION_NAMES.get(code, {}).get(lang, code)


def _fmt_price(amount: float, lang: str) -> str:
    if amount == int(amount):
        s = str(int(amount))
    else:
        s = f"{amount:.2f}"
    return f"${s}"


class Orchestrator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.llm = get_llm(self.settings)

    # ------------------------------------------------------------ 入口

    def handle(self, session: Session, text: str, lang_override: str | None = None) -> Reply:
        trace: list[dict] = []
        language = lang_override if lang_override in (LANG_ZH_HK, LANG_ZH_CN, "en") else detect_language(text)
        trace.append({"stage": "language", "detail": language})

        session.history.append(("user", mask_pii(text)))

        # 会话状态驱动的路由覆写：多轮流程（留资/工单）未完结时，下一轮延续原流程；
        # 转人工永远最高优先，允许用户随时退出当前流程。
        pending = None
        if session.state.get("ent", {}).get("ask_contact"):
            pending = INTENT_ENTERPRISE
        elif session.state.get("ticket") is not None:
            pending = INTENT_TICKET
        routed = route(text)
        if pending and routed.intent != INTENT_HANDOFF:
            result = RouteResult(pending, 1, ["session_state"])
        else:
            result = routed
        if self.llm is not None and result.score == 0:
            from .intent import classify_with_llm_fallback

            result = classify_with_llm_fallback(text, self.llm, self.settings)
        trace.append({"stage": "intent", "detail": f"{result.intent}({result.name}) score={result.score}"})

        handlers = {
            INTENT_ROAMING: self._flow_roaming,
            INTENT_RECOMMEND: self._flow_recommend,
            INTENT_ENTERPRISE: self._flow_enterprise,
            INTENT_KB: self._flow_kb,
            INTENT_TICKET: self._flow_ticket,
            INTENT_HANDOFF: self._flow_handoff,
            INTENT_CHITCHAT: self._flow_chitchat,
        }
        reply = handlers[result.intent](session, text, language)
        reply.language = language
        reply.intent = result.intent
        reply.trace = trace + reply.trace
        session.history.append(("assistant", reply.text))
        return reply

    # ------------------------------------------------------------ 工具结果 → 模板（确定性话术）

    def _flow_roaming(self, session: Session, text: str, lang: str) -> Reply:
        trace = [{"stage": "tools", "detail": "query_roaming_packs"}]
        norm = normalize(text)
        # 覆盖查询分支
        if re.search(r"覆盖|覆蓋|coverage|有冇信号|有没有信号", norm):
            region = resolve_region(text) or "HK"
            cov = check_coverage(region)
            t = {
                "zh-HK": f"{_region_name(region, lang)}網絡：{cov['coverage']}（{cov['network']}）。",
                "zh-CN": f"{_region_name(region, lang)}网络：{cov['coverage']}（{cov['network']}）。",
                "en": f"{_region_name(region, lang)} coverage: {cov['coverage']} ({cov['network']}).",
            }[lang]
            return Reply(t, lang, INTENT_ROAMING, tools=["check_coverage"], trace=trace)

        region = resolve_region(text)
        days = parse_days(text)
        if region is None and re.search(r"开通|開通|激活|啟動|启动|充值|enable|activate|top.?up", norm):
            # 子路由：不问目的地、只问开通/充值流程 → 走知识库
            hits = search_kb(norm, top_k=2)
            if hits:
                answer = extractive_answer(norm, hits)
                return Reply(
                    answer.strip(),
                    lang,
                    INTENT_ROAMING,
                    tools=["kb_search"],
                    citations=[f"{h['doc']} · {h['section']}" for h in hits],
                    trace=trace,
                )
        if region is None:
            asks = {
                "zh-HK": "想查邊個目的地嘅漫遊呀？例如：日本、英國、美國，或者內地/大灣區～\n可以話埋我知行程幾多日，我幫你揀最抵嘅數據包。",
                "zh-CN": "想查询哪个目的地的漫游呢？例如：日本、英国、美国，或者内地/大湾区～\n可以告诉我行程几天，我帮你挑最合适的数据包。",
                "en": "Which destination's roaming would you like to check? e.g. Japan, the UK, the US, or Mainland/GBA.\nTell me your trip length and I'll pick the best-value data pack.",
            }
            return Reply(asks[lang], lang, INTENT_ROAMING, trace=trace)

        packs = query_roaming_packs(region, days)
        lines = []
        if packs:
            for p in packs[:2]:
                lines.append(f"• {p['name']} — {_fmt_price(p['price'], lang)}")
        plan_lines = []
        if region in ("HK", "CN", "MO"):
            gba = [p for p in query_plans(persona="gba") if region in p["data_scope"]]
            if gba:
                top = gba[0]
                plan_lines.append(
                    f"• {top['name']} — {_fmt_price(top['monthly_fee'], lang)}/月（"
                    + {"zh-HK": "港澳內三地共享", "zh-CN": "港澳内三地共享", "en": "shared across HK/Mainland/Macau"}[lang]
                    + "）"
                )
        if not lines and not plan_lines:
            t = {
                "zh-HK": f"暫時冇{_region_name(region, lang)}嘅漫遊數據包資料，建議致電熱線 2122 1188 查詢，或者等我幫你轉人工。",
                "zh-CN": f"暂时没有{_region_name(region, lang)}的漫游数据包信息，建议致电热线 2122 1188 查询，或让我帮你转人工。",
                "en": f"No roaming pack data for {_region_name(region, lang)} yet. Please call 2122 1188 or ask me to transfer you to an agent.",
            }
            return Reply(t[lang], lang, INTENT_ROAMING, tools=["query_roaming_packs"], trace=trace)

        head = {
            "zh-HK": f"香港 → {_region_name(region, lang)} 漫遊建議：",
            "zh-CN": f"香港 → {_region_name(region, lang)} 漫游建议：",
            "en": f"Roaming options for {_region_name(region, lang)}:",
        }[lang]
        tips = {
            "zh-HK": "出發前請確認已開通國際漫遊（一般需按金 HK$300，以 Visa/MasterCard 辦理可獲豁免）。",
            "zh-CN": "出发前请确认已开通国际漫游（一般需按金 HK$300，以 Visa/MasterCard 办理可获豁免）。",
            "en": "Make sure international roaming is enabled before departure (a HK$300 deposit normally applies; waived when registered with Visa/MasterCard).",
        }[lang]
        body = head + "\n" + "\n".join(lines + plan_lines) + "\n" + tips
        return Reply(body, lang, INTENT_ROAMING, tools=["query_roaming_packs", "query_plans"], trace=trace)

    def _flow_recommend(self, session: Session, text: str, lang: str) -> Reply:
        trace = [{"stage": "tools", "detail": "recommend_plan"}]
        rec = recommend_plan(text)
        if not rec["persona"] and not rec["region"] and rec["budget"] is None:
            asks = {
                "zh-HK": "想我幫你推薦計劃，可以話埋我知你嘅使用場景：\n• 經常北上／大灣區通勤\n• 在港內地學生\n• 假期外遊\n• 香港本地大用量\n有預算都可以一齊講～",
                "zh-CN": "想让我推荐套餐，可以告诉我你的使用场景：\n• 经常北上／大湾区通勤\n• 在港内地学生\n• 假期外游\n• 香港本地大用量\n有预算也可以一起说～",
                "en": "To recommend a plan, tell me your usage pattern:\n• Frequent GBA / Mainland commuting\n• Mainland student in HK\n• Holiday travelling\n• Heavy local usage in HK\nBudget hints welcome too.",
            }
            return Reply(asks[lang], lang, INTENT_RECOMMEND, trace=trace)

        def fmt_plan(p) -> str:
            extras = []
            if p.get("mainland_number"):
                extras.append({"zh-HK": "送內地號碼", "zh-CN": "送内地号码", "en": "free mainland number"}[lang])
            if p.get("mainland_data_gb"):
                extras.append(f"{p['mainland_data_gb']}GB " + {"zh-HK": "內地數據", "zh-CN": "内地数据", "en": "mainland data"}[lang])
            scope = "/".join(_REGION_NAMES.get(c, {}).get(lang, c) for c in p["data_scope"])
            extra_txt = ("，" + "、".join(extras)) if extras else ""
            unit = {"zh-HK": "/月", "zh-CN": "/月", "en": "/mo"}[lang]
            return f"• {p['name']} — {_fmt_price(p['monthly_fee'], lang)}{unit}（{p['data_gb']}GB {scope}{extra_txt}）"

        lines = []
        if rec["top"]:
            lines.append(fmt_plan(rec["top"]))
        if rec["alt"]:
            lines.append(fmt_plan(rec["alt"]))
        body = {
            "zh-HK": "根據你嘅情況，建議：",
            "zh-CN": "根据你的情况，建议：",
            "en": "Based on your situation, we suggest:",
        }[lang]
        reply_text = body + "\n" + "\n".join(lines)
        return Reply(reply_text, lang, INTENT_RECOMMEND, tools=["query_plans"], trace=trace)

    # ------------------------------------------------------------ 企业方案（多轮槽位 + 留资）

    def _flow_enterprise(self, session: Session, text: str, lang: str) -> Reply:
        from .tools import parse_bandwidth, parse_industry, parse_sites

        state = session.state.setdefault(
            "ent", {"need_parts": [], "industry": None, "sites": None, "bandwidth": None, "ask_contact": False, "contact_tries": 0}
        )
        trace = [{"stage": "tools", "detail": "search_solutions"}]
        state["need_parts"].append(text)
        industry = parse_industry(text) or state["industry"]
        sites = parse_sites(text) or state["sites"]
        bandwidth = parse_bandwidth(text) or state["bandwidth"]
        state["industry"], state["sites"], state["bandwidth"] = industry, sites, bandwidth
        need_text = "；".join(state["need_parts"][-3:])

        # 留资阶段
        if state["ask_contact"]:
            contact = self._find_contact(text)
            if contact:
                company = industry or {"zh-HK": "企业客户", "zh-CN": "企业客户", "en": "Enterprise"}[lang]
                lead = create_lead(company, contact, need_text, lang)
                session.state.pop("ent", None)
                t = {
                    "zh-HK": f"已為你登記商機（編號 {lead['lead_id']}），客戶經理會盡快聯絡你（聯絡方式已脫敏保存：{lead['contact_masked']}）。多謝支持！",
                    "zh-CN": f"已为你登记商机（编号 {lead['lead_id']}），客户经理会尽快联络你（联络方式已脱敏保存：{lead['contact_masked']}）。谢谢支持！",
                    "en": f"Your enquiry is registered ({lead['lead_id']}). Our account manager will contact you soon (contact stored masked: {lead['contact_masked']}). Thank you!",
                }
                return Reply(t[lang], lang, INTENT_ENTERPRISE, tools=["create_lead"], trace=trace)
            if _DECLINE_RE.search(text) or state["contact_tries"] >= 1:
                session.state.pop("ent", None)
                t = {
                    "zh-HK": "冇問題！你可以隨時返嚟查詢，或直接致電 2122 1188 找客戶經理跟進。",
                    "zh-CN": "没问题！你可以随时回来查询，或直接致电 2122 1188 找客户经理跟进。",
                    "en": "No problem! Feel free to come back anytime, or call 2122 1188 to reach an account manager.",
                }
                return Reply(t[lang], lang, INTENT_ENTERPRISE, trace=trace)
            state["contact_tries"] += 1
            t = {
                "zh-HK": "方便留低聯絡電話或 email嗎？方便客戶經理跟你進一步報價。（唔想留也可以話我知「唔使」）",
                "zh-CN": "方便留下联系电话或 email 吗？方便客户经理进一步报价。（不想留也可以说「不用」）",
                "en": "Could you leave a phone number or email for a detailed quotation? (Say 'no thanks' if you prefer not to.)",
            }
            return Reply(t[lang], lang, INTENT_ENTERPRISE, trace=trace)

        solutions = search_solutions(need_text, industry, sites, bandwidth)
        if not solutions:
            t = {
                "zh-HK": "想了解多啲你嘅組網需求：大概係咩行業？有幾多個分支/門市？帶寬要求（如 100M）？\n例如：「連鎖零售 20 間門市，想要性價比高嘅組網」。",
                "zh-CN": "想多了解一下你的组网需求：大概是什么行业？有多少个分支/门店？带宽要求（如 100M）？\n例如：「连锁零售 20 家门店，想要性价比高的组网」。",
                "en": "Could you share more about your networking needs: industry, number of branches/stores, bandwidth (e.g. 100M)?\ne.g. 'Retail chain, 20 stores, cost-effective networking'.",
            }
            return Reply(t[lang], lang, INTENT_ENTERPRISE, trace=trace)

        names = {"zh-HK": "建議方案", "zh-CN": "建议方案", "en": "Recommended solutions"}[lang]
        lines = []
        for sol in solutions[:2]:
            feature = sol["features"][0]
            lines.append(f"• {sol['name']}：{feature}")
        state["ask_contact"] = True
        ask = {
            "zh-HK": "如果想要針對性報價，可以留低聯絡電話或 email，我幫你登記商機畀客戶經理跟進。",
            "zh-CN": "如果想要针对性报价，可以留下联系电话或 email，我帮你登记商机给客户经理跟进。",
            "en": "For a tailored quotation, leave your phone or email and I'll register the enquiry for our account manager.",
        }[lang]
        return Reply(names + "\n" + "\n".join(lines) + "\n" + ask, lang, INTENT_ENTERPRISE, tools=["search_solutions"], trace=trace)

    # ------------------------------------------------------------ 知识问答（RAG）

    def _flow_kb(self, session: Session, text: str, lang: str) -> Reply:
        trace = [{"stage": "tools", "detail": "kb_search(BM25)"}]
        norm = normalize(text)
        hits = search_kb(norm, top_k=3)
        min_score = self.settings.rag_min_score
        if not hits or (hits and hits[0]["score"] < min_score * 10):  # BM25 分数量级与阈值换算
            t = {
                "zh-HK": "呢個問題我暫時搵唔到可靠資料，為免答錯，建議致電客戶服務熱線 2122 1188，或者等我幫你轉人工。",
                "zh-CN": "这个问题我暂时找不到可靠资料，为免答错，建议致电客户服务热线 2122 1188，或让我帮你转人工。",
                "en": "I can't verify this right now. To be safe, please call our hotline 2122 1188, or ask me to transfer you to an agent.",
            }
            return Reply(t[lang], lang, INTENT_KB, trace=trace)
        answer = extractive_answer(norm, hits)
        citations = [f"{h['doc']} · {h['section']}" for h in hits]
        return Reply(answer.strip(), lang, INTENT_KB, tools=["kb_search"], citations=citations, trace=trace)

    # ------------------------------------------------------------ 投诉工单（多轮 + 写操作）

    def _flow_ticket(self, session: Session, text: str, lang: str) -> Reply:
        state = session.state.setdefault("ticket", {"desc": [], "contact": None})
        trace = [{"stage": "tools", "detail": "create_ticket"}]
        norm = normalize(text)
        if re.search(r"退款|退錢|退钱|扣错|扣錯|多收|乱扣", norm):
            category = {"zh-HK": "收費爭議", "zh-CN": "收费争议", "en": "Billing dispute"}[lang]
        elif re.search(r"信号|訊號|上网|上網|断网|驗證碼|验证码|故障|壞", norm):
            category = {"zh-HK": "網絡故障", "zh-CN": "网络故障", "en": "Network issue"}[lang]
        else:
            category = {"zh-HK": "其他問題", "zh-CN": "其他问题", "en": "Other"}[lang]

        contact = self._find_contact(text)
        if contact:
            state["contact"] = mask_contact(contact)
        if len(state["desc"]) < 2:
            state["desc"].append(mask_pii(text))

        if state["contact"] and len(state["desc"]) >= 1:
            desc = "；".join(state["desc"])[:400]
            ticket = create_ticket(category, desc, state["contact"], lang)
            session.state.pop("ticket", None)
            t = {
                "zh-HK": f"已為你登記{category}工單（編號 {ticket['ticket_id']}），同事會於 1 個工作日內以 {ticket['contact_masked']} 聯絡你。不便之處，敬請原諒。",
                "zh-CN": f"已为你登记{category}工单（编号 {ticket['ticket_id']}），同事会在 1 个工作日内以 {ticket['contact_masked']} 联络你。给您带来不便，敬请谅解。",
                "en": f"Your {category} ticket is registered ({ticket['ticket_id']}). Our colleague will contact you within 1 business day at {ticket['contact_masked']}. Sorry for the inconvenience.",
            }
            return Reply(t[lang], lang, INTENT_TICKET, tools=["create_ticket"], trace=trace)

        if not state["contact"]:
            t = {
                "zh-HK": "收到，好唔好意思畀你遇到呢個情況。方便留低聯絡電話或 email 畀我登記工單跟進嗎？",
                "zh-CN": "收到，很抱歉给您带来不便。方便留下联系电话或 email，让我登记工单跟进吗？",
                "en": "Sorry about that. Could you leave a phone number or email so I can register a ticket for follow-up?",
            }
            return Reply(t[lang], lang, INTENT_TICKET, trace=trace)
        return Reply("", lang, INTENT_TICKET, trace=trace)  # 不会到达：contact 分支已覆盖

    # ------------------------------------------------------------ 转人工

    def _flow_handoff(self, session: Session, text: str, lang: str) -> Reply:
        trace = [{"stage": "tools", "detail": "transfer_human"}]
        result = transfer_human(session.session_id, text)
        t = {
            "zh-HK": f"好，幫你轉接人工客服（排隊編號 {result['queue_ref']}）。人工服務時間以外，可以致電熱線 {result['hotline']}，或者喺度話我知你想問咩，我盡量幫你。",
            "zh-CN": f"好的，帮你转接人工客服（排队编号 {result['queue_ref']}）。人工服务时间以外，可以致电热线 {result['hotline']}，或者在这里告诉我你想问什么，我尽力帮你。",
            "en": f"Sure, transferring you to a human agent (queue ref {result['queue_ref']}). Outside service hours, call {result['hotline']}, or tell me your question and I'll do my best.",
        }
        return Reply(t[lang], lang, INTENT_HANDOFF, tools=["transfer_human"], handoff=True, trace=trace)

    # ------------------------------------------------------------ 问候闲聊

    def _flow_chitchat(self, session: Session, text: str, lang: str) -> Reply:
        t = {
            "zh-HK": "你好！我係三語智能客服助理～可以幫你：\n• 查漫遊數據包／大湾区计划（大灣區計劃）\n• 推薦合適你嘅套餐\n• 企業專線/組網方案諮詢\n• 登記投訴工單、轉人工\n想問啲咩？",
            "zh-CN": "你好！我是三语智能客服助理～可以帮你：\n• 查漫游数据包／大湾区计划\n• 推荐合适的套餐\n• 企业专线/组网方案咨询\n• 登记投诉工单、转人工\n想问点什么？",
            "en": "Hello! I'm your trilingual virtual assistant. I can help with:\n• Roaming / GBA plan enquiries\n• Plan recommendations\n• Enterprise connectivity solutions\n• Complaint tickets & human handoff\nHow may I help?",
        }
        return Reply(t[lang], lang, INTENT_CHITCHAT, trace=[{"stage": "compose", "detail": "template"}])

    # ------------------------------------------------------------ 工具函数

    @staticmethod
    def _find_contact(text: str) -> str | None:
        m = _CONTACT_RE.search(text or "")
        if not m:
            return None
        return next((g for g in m.groups() if g), None)

    # ------------------------------------------------------------ 真实模式生成（LLM 话术 + 护栏回退）

    def compose_with_llm(self, language: str, question: str, tool_results_text: str,
                         fallback: str, allowed_sources: list[str]) -> tuple[str, bool]:
        """真实模式：LLM 在工具结果上组织话术；金额越界即回退确定性模板。

        返回 (最终文本, 是否使用llm)。
        """
        if self.llm is None:
            return fallback, False
        try:
            messages = build_generation_prompt(language, question, tool_results_text)
            draft = self.llm.chat(messages)
            violations = validate_amounts(draft, allowed_sources)
            if violations:
                return fallback, False
            return ensure_disclaimer(draft, language), True
        except Exception:
            return fallback, False

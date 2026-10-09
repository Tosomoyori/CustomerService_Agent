"""编排层端到端测试（Mock 模式，全部离线确定性）。"""

from app.orchestrator import Orchestrator, Reply, Session


def make_orch() -> Orchestrator:
    return Orchestrator()


def test_roaming_flow_with_price_from_tool():
    orch = make_orch()
    reply = orch.handle(Session("s1"), "去日本5天，漫游包几钱？")
    assert reply.intent == "roaming_inquiry"
    assert "$98" in reply.text  # rp-jp-kr-5d 5日5GB
    assert "query_roaming_packs" in reply.tools


def test_roaming_flow_cantonese():
    orch = make_orch()
    reply = orch.handle(Session("s2"), "東京漫遊幾多錢呀？")
    assert reply.language == "zh-HK"
    assert "$" in reply.text


def test_roaming_flow_gba_suggests_shared_plan():
    orch = make_orch()
    reply = orch.handle(Session("s3"), "周末去深圳，有咩大灣區計劃？")
    assert reply.intent == "roaming_inquiry"
    assert "大灣區" in reply.text or "大湾区" in reply.text


def test_roaming_asks_destination_when_missing():
    orch = make_orch()
    reply = orch.handle(Session("s4"), "漫游包多少钱？")
    assert "日本" in reply.text or "Japan" in reply.text  # 反问并给出示例


def test_roaming_open_subroute_uses_kb():
    orch = make_orch()
    reply = orch.handle(Session("s5"), "国际漫游怎么开通？")
    assert "2122 1188" in reply.text or "按金" in reply.text


def test_recommend_student_plan():
    orch = make_orch()
    reply = orch.handle(Session("s6"), "我是来港读书的内地学生，推荐下套餐")
    assert reply.intent == "plan_recommend"
    assert "$99" in reply.text


def test_recommend_asks_scenario_when_unknown():
    orch = make_orch()
    reply = orch.handle(Session("s7"), "推荐个套餐")
    assert "通勤" in reply.text or "GBA" in reply.text or "学生" in reply.text


def test_enterprise_flow_multiturn_with_lead():
    orch = make_orch()
    session = Session("s8")
    r1 = orch.handle(session, "我们公司做跨境电商，想上云专线组网")
    assert reply_has(r1, "跨境雲專線") or reply_has(r1, "跨境云专线")
    assert "留" in r1.text or "email" in r1.text  # 请求留资
    r2 = orch.handle(session, "找 Kevin，电话 91234567")
    assert r2.intent == "enterprise_solution"
    assert "LD-" in r2.text
    assert "912****67" in r2.text  # 联系方式脱敏展示


def test_ticket_flow_multiturn():
    orch = make_orch()
    session = Session("s9")
    r1 = orch.handle(session, "我张卡收唔到验证码，好困扰")
    assert r1.intent == "complaint_ticket"
    r2 = orch.handle(session, "我电话 6123 4567")
    assert "TK-" in r2.text
    assert "612****67" in r2.text


def test_handoff_flow():
    orch = make_orch()
    reply = orch.handle(Session("s10"), "帮我转人工")
    assert reply.handoff is True
    assert "Q-" in reply.text


def test_kb_flow_with_citation():
    orch = make_orch()
    reply = orch.handle(Session("s11"), "什么是一卡两号？")
    assert reply.intent == "kb_qa"
    assert reply.citations
    assert "一卡两号" in reply.text


def test_kb_fallback_on_no_evidence():
    orch = make_orch()
    reply = orch.handle(Session("s12"), "qwertyzzz asdfghjkl?")
    assert "2122 1188" in reply.text  # 无可靠资料 → 引导热线，不编造


def test_language_override():
    orch = make_orch()
    reply = orch.handle(Session("s13"), "漫游？", lang_override="en")
    assert reply.language == "en"


def test_history_stores_masked_text():
    orch = make_orch()
    session = Session("s14")
    orch.handle(session, "我电话 91234567 想投诉")
    user_text = session.history[0][1]
    assert "91234567" not in user_text
    assert "****" in user_text


def reply_has(reply: Reply, fragment: str) -> bool:
    return fragment in reply.text

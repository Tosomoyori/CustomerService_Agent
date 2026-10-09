"""工具层测试：地区/实体解析、资费与方案查询、写操作、Function Calling 派发。"""

from app.tools import (
    TOOLS_SCHEMA,
    check_coverage,
    create_lead,
    create_ticket,
    dispatch_tool,
    parse_bandwidth,
    parse_budget,
    parse_days,
    parse_industry,
    parse_sites,
    query_plans,
    query_roaming_packs,
    recommend_plan,
    resolve_region,
    search_solutions,
)


def test_resolve_region_aliases():
    assert resolve_region("去日本旅行") == "JP"
    assert resolve_region("東京漫游") == "JP"
    assert resolve_region("大湾区计划") == "CN"
    assert resolve_region("London trip") == "UK"
    assert resolve_region("聊聊别的") is None


def test_parse_days_and_budget():
    assert parse_days("去英国7日") == 7
    assert parse_days("5天漫游包") == 5
    assert parse_budget("预算200以下") == 200.0


def test_query_plans_filters():
    students = query_plans(persona="student")
    assert students and students[0]["persona"] == "student"
    japan = query_plans(region="JP")
    assert japan and all("JP" in p["data_scope"] for p in japan)
    cheap = query_plans(budget_max=100)
    assert all(p["monthly_fee"] <= 100 for p in cheap)


def test_query_roaming_packs_by_destination():
    packs = query_roaming_packs("JP", days=5)
    assert packs and "JP" in packs[0]["destinations"]
    assert query_roaming_packs("AQ") == []  # 无南极漫游包


def test_recommend_plan_student():
    rec = recommend_plan("我是来港读研的内地学生")
    assert rec["persona"] == "student"
    assert rec["top"]["id"] == "student-100"


def test_recommend_plan_gba_commuter():
    rec = recommend_plan("我每日北上深圳返工，想要大湾区共享数据")
    assert rec["persona"] == "gba"
    assert rec["top"]["persona"] == "gba"


def test_search_solutions_sdwan_vs_iepl():
    sdwan = search_solutions("连锁零售 20 间门店，想要性价比高的组网", industry="零售")
    assert sdwan and sdwan[0]["id"] == "sdwan"
    iepl = search_solutions("金融总部与数据中心之间低时延点对点专线")
    assert iepl and iepl[0]["id"] == "iepl"


def test_parse_entities():
    assert parse_bandwidth("需要100M带宽") == 100
    assert parse_bandwidth("上1G专线") == 1000
    assert parse_sites("20间分店") == 20
    assert parse_industry("我们做跨境电商平台") == "跨境电商"


def test_check_coverage():
    hk = check_coverage("HK")
    assert "5G" in hk["network"]
    ent = check_coverage("US", enterprise=True)
    assert "pop_nodes" in ent and ent["pop_nodes"]


def test_create_ticket_and_lead():
    ticket = create_ticket("网络故障", "深圳收不到验证码", "91234567", "zh-HK")
    assert ticket["ticket_id"].startswith("TK-")
    assert ticket["contact_masked"].startswith("912")
    assert "****" in ticket["contact_masked"]
    lead = create_lead("示例贸易公司", "13612345678", "需要跨境云专线", "zh-CN")
    assert lead["lead_id"].startswith("LD-")


def test_dispatch_tool():
    cov = dispatch_tool("check_coverage", {"region": "JP"})
    assert cov["region"] == "JP"
    packs = dispatch_tool("query_roaming_packs", {"region": "JP", "days": 5})
    assert isinstance(packs, list) and packs


def test_tools_schema_complete():
    names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    assert {
        "query_roaming_packs", "query_plans", "search_solutions", "check_coverage",
        "kb_search_tool", "create_ticket", "create_lead", "transfer_human",
    } <= names
    for t in TOOLS_SCHEMA:
        assert "parameters" in t["function"]

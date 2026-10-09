"""意图路由测试（规则路径，离线确定性）。"""

from app.intent import route


def test_roaming_price():
    assert route("日本漫游多少钱").intent == "roaming_inquiry"


def test_roaming_pack_cantonese():
    assert route("去英國七日，買邊個漫遊數據包好？").intent == "roaming_inquiry"


def test_gba_plan_listing():
    assert route("有咩大灣區計劃推介呀？").intent == "roaming_inquiry"


def test_recommend_student():
    assert route("我是学生，有什么推荐的计划吗？").intent == "plan_recommend"


def test_recommend_suit_me():
    assert route("邊張plan適合我？").intent == "plan_recommend"


def test_enterprise_keywords():
    assert route("公司要做跨境电商，需要云专线组网").intent == "enterprise_solution"
    assert route("Do you have SD-WAN for our retail stores?").intent == "enterprise_solution"


def test_ticket_network_down():
    assert route("我手机上不了网了").intent == "complaint_ticket"
    assert route("收唔到驗證碼").intent == "complaint_ticket"


def test_ticket_billing():
    assert route("话费扣错了，我要退款").intent == "complaint_ticket"


def test_handoff_hard_priority():
    assert route("麻烦转人工").intent == "human_handoff"
    # 工单与转人工同时命中：转人工硬优先
    assert route("投诉，帮我转人工").intent == "human_handoff"


def test_kb():
    assert route("什么是eSIM？").intent == "kb_qa"
    assert route("eSIM 怎么开通？").intent == "kb_qa"


def test_chitchat_greeting():
    assert route("你好").intent == "chitchat"
    assert route("hello").intent == "chitchat"


def test_score_priority_beats_low_score():
    # 「划算」(RECOMMEND 2) 不敌「漫游数据包」(ROAMING 6)
    assert route("去泰国玩5天，买哪个漫游数据包划算？").intent == "roaming_inquiry"

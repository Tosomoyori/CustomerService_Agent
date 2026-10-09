"""意图路由：规则优先（离线可跑、结果可解释），真实模式下可由 LLM 兜底。

意图集合（按优先级从高到低，命中即停）：
  human_handoff      转人工
  complaint_ticket   投诉/故障申报（涉及写操作，优先于咨询类）
  enterprise_solution 企业跨境组网咨询（B 端，商机线索入口）
  plan_recommend     个人套餐推荐（咨询 + 决策辅助）
  roaming_inquiry    漫游/套餐资费查询（结构化查询）
  kb_qa              业务知识问答（RAG）
  chitchat           问候/闲聊

规则打分：对归一化后的文本做关键词/正则匹配，按意图累计得分；
得分为 0 时回落 kb_qa（服务型默认给出有用信息）或 chitchat。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .language import normalize

INTENT_HANDOFF = "human_handoff"
INTENT_TICKET = "complaint_ticket"
INTENT_ENTERPRISE = "enterprise_solution"
INTENT_RECOMMEND = "plan_recommend"
INTENT_ROAMING = "roaming_inquiry"
INTENT_KB = "kb_qa"
INTENT_CHITCHAT = "chitchat"

PRIORITY = [
    INTENT_HANDOFF,
    INTENT_TICKET,
    INTENT_ENTERPRISE,
    INTENT_RECOMMEND,
    INTENT_ROAMING,
    INTENT_KB,
]

# (intent, 权重, 正则) —— 正则在归一化文本上匹配
_RULES: list[tuple[str, int, str]] = [
    # 转人工
    (INTENT_HANDOFF, 3, r"人工|真人|客服专员|搵人倾|搵人讲|揾人|接线|转接人|human|real person|speak to (a )?(person|agent|someone)"),
    # 投诉/故障
    (INTENT_TICKET, 3, r"投诉|投訴|申诉|申訴|退款|退钱|扣错|扣錯|乱扣|乱收費|多收"),
    (INTENT_TICKET, 3, r"故障|坏咗|壞咗|无信号|冇信号|沒信号|没信号|断网|斷網|上唔到网|上不到网|上不了网|连唔上|连不上|连不上网|收唔到验证码|收不到验证码|收不到短訊|收不到短信"),
    (INTENT_TICKET, 2, r"not working|no signal|no service|complaint|refund|can't (connect|login|receive)"),
    # 企业方案
    (INTENT_ENTERPRISE, 3, r"iepl|iplc|sd-?wan|mpls|ip transit|ip转接|云专线|云互联|高速通道"),
    (INTENT_ENTERPRISE, 3, r"企业专线|公司专线|组网|组網|专线接入|点对点专線|数据中心|機櫃|机柜|私有云"),
    (INTENT_ENTERPRISE, 2, r"企业|公司|分行|分店|分支|門市|门店|办公室上网|跨境组网|企业上云"),
    (INTENT_ENTERPRISE, 2, r"enterprise|office network|branch (office|network)|leased line|private line"),
    # 套餐推荐
    (INTENT_RECOMMEND, 3, r"推荐.{0,6}(套餐|计划|plan)|推薦.{0,6}(套餐|計劃|plan)|边张plan|邊張plan|边个plan|边个卡|边张卡|哪个套餐|哪個套餐|哪种套餐"),
    (INTENT_RECOMMEND, 2, r"适合我|適合我|划算|抵用|最平|最便宜|划算嗎|suggest|recommend|which plan"),
    (INTENT_RECOMMEND, 2, r"学生计划|學生計劃|学生卡|北上|通勤|大湾区卡|大灣區卡|一卡两号|一卡兩號"),
    # 漫游/资费查询
    (INTENT_ROAMING, 3, r"漫游|漫遊|roaming|数据包|數據包|日包|流量包|上網卡|上网卡|储值卡|儲值卡|旅游卡|旅遊卡|覆盖|覆蓋|coverage"),
    (INTENT_ROAMING, 2, r"几钱|幾钱|多少钱|多少錢|月费|月費|价钱|價錢|价格|收费|收費|price|cost|how much"),
    (INTENT_ROAMING, 2, r"流量|數據|数据|用量|gb|5g|4g|sim|esim"),
    (INTENT_ROAMING, 1, r"套餐|套餐費|计划|計劃|plan"),
    # 知识问答
    (INTENT_KB, 3, r"什么是|什麼是|开通|開通"),
    (INTENT_KB, 2, r"点解要|如何|怎样|怎麼|怎么办理|怎么办|how (to|do|can)|what is|can i|where (can|to)|why|充值"),
    (INTENT_KB, 1, r"\?|？|点充|点开|点申请|点買|點買"),
]

_GREET_RE = re.compile(r"^(你好|您好|hi|hello|hey|哈囉|哈啰|早晨|早上好|午安|晚安|嗨)[!！。~ ～]*$", re.IGNORECASE)

_INTENT_NAMES = {
    INTENT_HANDOFF: "转人工",
    INTENT_TICKET: "投诉工单",
    INTENT_ENTERPRISE: "企业方案",
    INTENT_RECOMMEND: "套餐推荐",
    INTENT_ROAMING: "漫游资费",
    INTENT_KB: "知识问答",
    INTENT_CHITCHAT: "问候闲聊",
}


@dataclass
class RouteResult:
    intent: str
    score: int
    matched: list[str]
    by_llm: bool = False

    @property
    def name(self) -> str:
        return _INTENT_NAMES.get(self.intent, self.intent)


def route(text: str) -> RouteResult:
    """规则路由：归一化 → 按意图累计匹配得分。

    裁决策略：转人工/投诉工单属安全关键意图，命中即优先返回；
    其余意图分数优先，同分时按优先序（咨询类让位业务类）。
    """
    norm = normalize(text or "").lower()
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}
    for intent, weight, pattern in _RULES:
        for m in re.finditer(pattern, norm, re.IGNORECASE):
            scores[intent] = scores.get(intent, 0) + weight
            matched.setdefault(intent, []).append(m.group(0))
    # 安全关键意图：硬优先
    for intent in (INTENT_HANDOFF, INTENT_TICKET):
        if scores.get(intent, 0) > 0:
            return RouteResult(intent, scores[intent], matched.get(intent, []))
    # 其余意图：分数优先，平手按 PRIORITY 序
    best = None
    for idx, intent in enumerate(PRIORITY):
        score = scores.get(intent, 0)
        if score <= 0:
            continue
        if best is None or score > best[1]:
            best = (intent, score, idx)
    if best:
        return RouteResult(best[0], best[1], matched.get(best[0], []))
    if _GREET_RE.match((text or "").strip()):
        return RouteResult(INTENT_CHITCHAT, 1, [text.strip()])
    # 无规则命中：带问号/疑问词默认知识问答，否则闲聊
    if re.search(r"\?|？|什么|咩|点|點|how|what", norm):
        return RouteResult(INTENT_KB, 0, [])
    return RouteResult(INTENT_CHITCHAT, 0, [])


def classify_with_llm_fallback(text: str, llm, settings) -> RouteResult:
    """规则路由 + LLM 兜底（真实模式）。规则零命中时调用 LLM 分类。"""
    result = route(text)
    if result.score > 0 or llm is None:
        return result
    prompt = (
        "你是电信客服机器人的意图分类器。从下列意图中选一个最匹配的，只输出意图 token：\n"
        "human_handoff(转人工) complaint_ticket(投诉故障) enterprise_solution(企业专线组网) "
        "plan_recommend(个人套餐推荐) roaming_inquiry(漫游资费查询) kb_qa(业务知识问答) chitchat(问候闲聊)\n"
        f"用户输入：{text}\n意图："
    )
    try:
        out = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )
        token = (out or "").strip().split()[0] if (out or "").strip() else ""
        token = token.strip("()（） 。.")
        if token in _INTENT_NAMES:
            return RouteResult(token, 1, ["llm"], by_llm=True)
    except Exception:
        pass
    return result

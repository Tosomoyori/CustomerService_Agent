"""工具层：结构化资费/方案/覆盖查询 + 工单/线索落库（SQLite）+ 转人工。

设计原则（与产品约束一致）：
- 金额的唯一来源是工具查询结果，模板/LLM 只负责组织话术，不参与计算；
- 工单与商机线索是写操作：联系方式先脱敏再落库（PDPO 数据最小化）；
- 全部工具为纯函数风格，便于 Function Calling 与离线测试。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from .config import DATA_DIR, RUNTIME_DIR
from .guardrails import mask_contact
from .language import normalize
from .rag import search_kb

_lock = threading.Lock()


def _load_json(name: str) -> dict:
    path = DATA_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _tariff() -> dict:
    return _load_json("tariff_plans.json")


@lru_cache(maxsize=1)
def _solutions() -> dict:
    return _load_json("enterprise_solutions.json")


@lru_cache(maxsize=1)
def _coverage() -> dict:
    return _load_json("coverage.json")


# ---------------------------------------------------------------- 地区解析

def resolve_region(text: str) -> str | None:
    """从文本解析地区代码（含中英别名、繁简与粤语口径）。"""
    norm = normalize(text or "").lower()
    aliases: dict[str, list[str]] = _tariff()["destination_aliases"]
    for code, words in aliases.items():
        for w in words:
            if w.lower() == norm.strip():
                return code
    # 子串匹配（长别名优先，避免「美國」被「國」类短词截胡）
    hits: list[tuple[int, str]] = []
    for code, words in aliases.items():
        for w in words:
            if w and w.lower() in norm:
                hits.append((len(w), code))
    if hits:
        hits.sort(reverse=True)
        return hits[0][1]
    ent_aliases = _coverage()["region_aliases"]
    for w, code in ent_aliases.items():
        if w in norm:
            return code
    return None


def parse_days(text: str) -> int | None:
    m = re.search(r"(\d{1,2})\s*(日|天|day)", normalize(text or "").lower())
    return int(m.group(1)) if m else None


def parse_budget(text: str) -> float | None:
    m = re.search(r"(?:预算|以下|budget|之內|之内|以内)[^\d]{0,4}(\d{2,4})", normalize(text or ""))
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{2,4})\s*(?:以下|之內|之内|以内|budget)", normalize(text or ""))
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- 个人侧工具

def query_plans(persona: str | None = None, region: str | None = None, budget_max: float | None = None) -> list[dict]:
    """按人群/目的地/预算筛选套餐，按价格升序。"""
    plans = _tariff()["plans"]
    out = []
    for p in plans:
        if persona and p["persona"] != persona:
            continue
        if region and region not in p["data_scope"]:
            continue
        if budget_max and p["monthly_fee"] > budget_max:
            continue
        out.append(p)
    out.sort(key=lambda p: p["monthly_fee"])
    return out


def query_roaming_packs(region: str, days: int | None = None, data_need_gb: float | None = None) -> list[dict]:
    """查询目的地漫游包：目的地必须覆盖，天数/用量做贴近度排序。"""
    packs = [p for p in _tariff()["roaming_packs"] if region in p["destinations"]]
    if not packs:
        return []
    def fitness(p):
        s = 0.0
        if days:
            s -= abs((p["days"] or 1) - days) if p["days"] else 0.5
        if data_need_gb:
            s -= abs(p["data_gb"] - data_need_gb)
        return (s, -p["price"])
    packs.sort(key=fitness)
    return packs


def recommend_plan(text: str) -> dict:
    """个人套餐推荐：从文本推断人群/预算/目的地，返回推荐与备选。"""
    norm = normalize(text or "")
    persona = None
    if re.search(r"学生|學生|student|内地生|內地生", norm):
        persona = "student"
    elif re.search(r"北上|通勤|大湾区|大灣區|跨境|兩地|两地|内地上班", norm):
        persona = "gba"
    elif re.search(r"旅行|旅游|旅遊|外遊|外游|去日本|去韩国|去韓國|旅行_sim|travel|假期|trip", norm):
        persona = "travel"
    elif re.search(r"本地|香港用|多用|大用量|追剧|打机|打機", norm):
        persona = "local"
    region = resolve_region(text)
    budget = parse_budget(text)
    # 回退顺序：人群+地区 → 仅人群 → 仅地区 → 全量（人群信号优先级高于地区，
    # 避免「来港的内地学生」因命中内地而丢掉学生人群）
    plans = query_plans(persona=persona, region=region, budget_max=budget)
    if not plans and persona:
        plans = query_plans(persona=persona, budget_max=budget)
    if not plans and (region or budget_max is not None):
        plans = query_plans(region=region, budget_max=budget)
    if not plans:
        plans = query_plans()
    top = plans[0] if plans else None
    alt = plans[1] if len(plans) > 1 else None
    return {"persona": persona, "region": region, "budget": budget, "top": top, "alt": alt}


def check_coverage(region: str, enterprise: bool = False) -> dict:
    cov = _coverage()
    region = (region or "").upper()
    if enterprise:
        return {
            "pop_nodes": cov["enterprise"]["pop_nodes"],
            "pop_note": cov["enterprise"]["pop_note"],
            "submarine_cables": cov["enterprise"]["submarine_cables"],
            "submarine_note": cov["enterprise"]["submarine_note"],
        }
    for item in cov["mobile"]:
        if item["region"] == region:
            return item
    return {"region": region, "network": "-", "coverage": "暂无数据", "note": cov["meta"]["note"]}


# ---------------------------------------------------------------- 企业侧工具

def search_solutions(text: str, industry: str | None = None, sites: int | None = None, bandwidth_mbps: int | None = None) -> list[dict]:
    """企业方案匹配：关键词命中 + 行业映射加分 + 站点/带宽约束过滤。"""
    norm = normalize(text or "").lower()
    data = _solutions()
    industry_map = data["industries"]
    industry_key = None
    if industry:
        for key in industry_map:
            if key in normalize(industry):
                industry_key = key
                break
    else:
        for key in industry_map:
            if key in norm:
                industry_key = key
                break

    scored: list[tuple[float, dict]] = []
    for sol in data["solutions"]:
        score = 0.0
        for kw in sol["keywords"]:
            if kw.lower() in norm:
                score += 3.0
        for sc in sol["scenarios"]:
            for frag in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9-]+", normalize(sc).lower()):
                if frag in norm and len(frag) >= 2:
                    score += 1.0
        if industry_key and sol["id"] in industry_map.get(industry_key, []):
            score += 2.5
        if sites and sites < sol["min_sites"]:
            score -= 3.0
        if bandwidth_mbps:
            lo, hi = sol["bandwidth_mbps"]
            if not (lo <= bandwidth_mbps <= hi):
                score -= 2.0
        if score > 0:
            scored.append((score, sol))
    scored.sort(key=lambda x: -x[0])
    return [dict(sol, match_score=round(s, 2)) for s, sol in scored[:3]]


def parse_bandwidth(text: str) -> int | None:
    norm = normalize(text or "")
    m = re.search(r"(\d{1,5})\s*(g|gbps)(?![a-z])", norm.lower())
    if m:
        return int(m.group(1)) * 1000
    m = re.search(r"(\d{1,5})\s*m(?![a-z])", norm.lower())
    return int(m.group(1)) if m else None


def parse_sites(text: str) -> int | None:
    norm = normalize(text or "")
    m = re.search(r"(\d{1,3})\s*(个站点|个分支|间分店|个分店|间门市|个门市|sites|branches|stores|分店|分支|站点|門市|门市)", norm.lower())
    return int(m.group(1)) if m else None


INDUSTRY_KEYWORDS = ["金融", "银行", "銀行", "零售", "连锁", "連鎖", "制造", "製造", "游戏", "遊戲",
                     "物流", "电商", "電商", "跨境电商", "互联网", "互聯網", "教育", "专业服务", "專業服務", "律师事务所"]


def parse_industry(text: str) -> str | None:
    norm = normalize(text or "")
    # 长词优先：避免「跨境电商」被「电商」截胡
    for kw in sorted(INDUSTRY_KEYWORDS, key=len, reverse=True):
        if kw in norm:
            return kw
    return None


# ---------------------------------------------------------------- 写操作工具

def _db_path() -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR / "agent.db"


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
               id TEXT PRIMARY KEY, category TEXT, description TEXT,
               contact TEXT, language TEXT, created_at TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS leads (
               id TEXT PRIMARY KEY, company TEXT, contact TEXT, need TEXT,
               tier TEXT, language TEXT, created_at TEXT)"""
    )


def create_ticket(category: str, description: str, contact: str, language: str) -> dict:
    """创建投诉/故障工单。联系方式脱敏后落库。"""
    ticket_id = f"TK-{uuid.uuid4().hex[:8].upper()}"
    masked = mask_contact(contact) if contact else ""
    conn = sqlite3.connect(_db_path())
    try:
        _init_db(conn)
        with _lock:
            conn.execute(
                "INSERT INTO tickets VALUES (?,?,?,?,?,?)",
                (ticket_id, category, description[:500], masked, language,
                 time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
    finally:
        conn.close()
    return {"ticket_id": ticket_id, "contact_masked": masked}


def create_lead(company: str, contact: str, need: str, language: str, tier: str = "warm") -> dict:
    """登记企业商机线索。联系方式脱敏后落库。"""
    lead_id = f"LD-{uuid.uuid4().hex[:8].upper()}"
    masked = mask_contact(contact) if contact else ""
    conn = sqlite3.connect(_db_path())
    try:
        _init_db(conn)
        with _lock:
            conn.execute(
                "INSERT INTO leads VALUES (?,?,?,?,?,?,?)",
                (lead_id, company[:100], masked, need[:500], tier, language,
                 time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
    finally:
        conn.close()
    return {"lead_id": lead_id, "contact_masked": masked}


def transfer_human(session_id: str, reason: str) -> dict:
    """转人工：返回队列参考号（演示环境为模拟排队）。"""
    return {
        "queue_ref": f"Q-{uuid.uuid4().hex[:6].upper()}",
        "session_id": session_id,
        "reason": reason[:200],
        "hotline": "2122 1188",
    }


# ---------------------------------------------------------------- KB 工具（Function Calling 用）

def kb_search_tool(query: str, top_k: int = 3) -> list[dict]:
    return search_kb(query, top_k=top_k)


# ---------------------------------------------------------------- Function Calling Schema（真实模式）

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_roaming_packs",
            "description": "按目的地查询漫游数据包资费（金额唯一来源）",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "地区代码，如 JP/US/UK"},
                    "days": {"type": "integer", "description": "行程天数"},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_plans",
            "description": "查询个人月费/储值套餐（persona: gba/student/local/travel）",
            "parameters": {
                "type": "object",
                "properties": {
                    "persona": {"type": "string"},
                    "region": {"type": "string"},
                    "budget_max": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_solutions",
            "description": "企业跨境组网方案匹配（IEPL/IPLC/SD-WAN/MPLS/云专线/IDC/IoT）",
            "parameters": {
                "type": "object",
                "properties": {
                    "need_text": {"type": "string"},
                    "industry": {"type": "string"},
                    "sites": {"type": "integer"},
                    "bandwidth_mbps": {"type": "integer"},
                },
                "required": ["need_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_coverage",
            "description": "查询网络/专线资源覆盖",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "enterprise": {"type": "boolean"},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_search_tool",
            "description": "检索业务知识库（公司介绍/套餐指引/漫游指引/服务政策等）",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "登记投诉/故障工单（写操作，需先获用户同意）",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                    "contact": {"type": "string"},
                    "language": {"type": "string"},
                },
                "required": ["category", "description", "contact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_lead",
            "description": "登记企业客户商机线索（写操作，需先获用户同意）",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "contact": {"type": "string"},
                    "need": {"type": "string"},
                    "language": {"type": "string"},
                    "tier": {"type": "string"},
                },
                "required": ["company", "contact", "need"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_human",
            "description": "转人工客服（终态动作）",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
]


def dispatch_tool(name: str, args: dict) -> object:
    """按名称执行工具（真实模式 Function Calling 与测试共用入口）。"""
    table = {
        "query_roaming_packs": lambda a: query_roaming_packs(a["region"], a.get("days")),
        "query_plans": lambda a: query_plans(a.get("persona"), a.get("region"), a.get("budget_max")),
        "search_solutions": lambda a: search_solutions(
            a["need_text"], a.get("industry"), a.get("sites"), a.get("bandwidth_mbps")
        ),
        "check_coverage": lambda a: check_coverage(a["region"], a.get("enterprise", False)),
        "kb_search_tool": lambda a: kb_search_tool(a["query"], a.get("top_k", 3)),
        "create_ticket": lambda a: create_ticket(
            a["category"], a["description"], a["contact"], a.get("language", "zh-HK")
        ),
        "create_lead": lambda a: create_lead(
            a["company"], a["contact"], a["need"], a.get("language", "zh-HK"), a.get("tier", "warm")
        ),
        "transfer_human": lambda a: transfer_human(a.get("session_id", "-"), a["reason"]),
    }
    if name not in table:
        raise KeyError(f"unknown tool: {name}")
    return table[name](args)

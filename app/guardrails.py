"""护栏：PII 脱敏 + 金额溯源 + 免责声明。

对应产品硬约束：
- 隐私（PDPO 数据最小化）：日志与落库前对电话/证件/邮箱脱敏；
- 资费防幻觉：回复中出现的每一个金额，必须能在本轮工具结果或知识库
  检索片段中找到出处；违反即触发回退（调用方负责回退到确定性模板）。
"""

from __future__ import annotations

import re

# 客服热线等公开号码不得被当成用户 PII 脱敏
_PUBLIC_NUMBERS = {"21221188", "2122 1188", "1188"}

_PHONE_PATTERNS = [
    re.compile(r"(?<!\d)([2-9]\d{3}[\s-]?\d{4})(?!\d)"),  # 香港固网/手机 8 位
    re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)"),  # 内地手机 11 位
]
_HKID_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]\d{6})([\(（]?[0-9A][\)）]?)")
_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)")
_AMOUNT_RE = re.compile(
    r"(?:HK\$|hk\$|港币|港元|蚊|\$|￥|¥|元|蚊)\s*([0-9][0-9,，]*(?:\.[0-9]+)?)"
    r"|([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(?:港元|港币|蚊|元|块)"
)


def _protect_public(text: str) -> tuple[str, list[str]]:
    """把公开号码临时替换为占位符，返回 (替换后文本, 占位符列表)。"""
    found = []
    out = text
    for idx, num in enumerate(_PUBLIC_NUMBERS):
        if num in out:
            placeholder = f"\x00PUB{idx}\x00"
            out = out.replace(num, placeholder)
            found.append((placeholder, num))
    return out, found


def _restore_public(text: str, found: list[tuple[str, str]]) -> str:
    for placeholder, num in found:
        text = text.replace(placeholder, num)
    return text


def mask_pii(text: str) -> str:
    """对文本中的电话/香港身份证/邮箱脱敏（保留尾部少量数字便于人工核对）。"""
    if not text:
        return text
    out, protected = _protect_public(text)
    for pattern in _PHONE_PATTERNS:
        out = pattern.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[-2:], out)
    out = _HKID_RE.sub(lambda m: m.group(1)[0] + "******" + (m.group(2) or ""), out)
    out = _EMAIL_RE.sub(lambda m: m.group(1)[:1] + "***@" + m.group(2), out)
    return _restore_public(out, protected)


def mask_contact(contact: str) -> str:
    """落库用联系方式脱敏：保留前 3 后 2。"""
    c = (contact or "").strip()
    if not c:
        return ""
    return c[:3] + "****" + c[-2:] if len(c) >= 7 else "****"


def extract_amounts(text: str) -> set[float]:
    """提取文本中的金额（港币口径），返回数值集合。"""
    if not text:
        return set()
    amounts: set[float] = set()
    for m in _AMOUNT_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw:
            try:
                amounts.add(float(raw.replace(",", "").replace("，", "")))
            except ValueError:
                continue
    return amounts


def validate_amounts(reply: str, allowed_sources: list[str]) -> list[float]:
    """金额溯源校验：返回回复中无法在允许来源中找到的金额列表（空列表=通过）。

    allowed_sources：本轮工具输出文本 + 知识库检索片段。
    """
    mentioned = extract_amounts(reply)
    allowed: set[float] = set()
    for src in allowed_sources:
        allowed |= extract_amounts(src)
    return sorted(mentioned - allowed)


DISCLAIMERS = {
    "zh-HK": "（资费与优惠以官方渠道为准，以上为查询时点数据，仅供參考）",
    "zh-CN": "（资费与优惠以官方渠道为准，以上为查询时点数据，仅供参考）",
    "en": "(Prices are for reference at query time; official channels prevail.)",
}


def ensure_disclaimer(reply: str, language: str) -> str:
    """资费类回复追加免责声明（已含则跳过）。"""
    tail = DISCLAIMERS.get(language, DISCLAIMERS["zh-HK"])
    if not reply or tail.strip("（）()") in reply:
        return reply
    return f"{reply.rstrip()}\n{tail}"

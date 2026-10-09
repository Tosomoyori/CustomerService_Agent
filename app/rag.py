"""轻量 RAG：结构化切片 + BM25 检索 + 抽取式回答与引用溯源。

设计要点：
- 切片按 Markdown 标题层级（# / ##）切成「文档-小节」语义单元，再按段落聚合、
  超长段落按句子切分，避免把一个主题切成两半（对齐 JD「数据清洗、切片管理」）；
- 检索用纯 Python 实现的 BM25（CJK 一元+二元 token，ASCII 词元），零向量库依赖，
  离线可跑、结果可复现；查询侧先经 language.normalize 归一化，
  保证粤语/繁体查询能命中简体知识库；
- 回答采用抽取式：从命中小节中挑与查询重叠度最高的句子拼装，
  并强制附「来源：文档 · 小节」引用，金额等事实可溯源。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from .config import KB_DIR
from .language import normalize

_CHUNK_MAX_CHARS = 400
_SENT_SPLIT_RE = re.compile(r"(?<=[。；！？!?])|\n+")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str
    tokens: list[str]


def _tokenize(text: str) -> list[str]:
    """CJK 一元+二元 token 与 ASCII 小写词元混合分词。"""
    text = normalize(text.lower())
    tokens: list[str] = []
    ascii_tokens = re.findall(r"[a-z0-9]{2,}", text)
    tokens.extend(ascii_tokens)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        tokens.extend(run)  # 一元
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # 二元
    return tokens


def chunk_markdown(raw: str, doc: str) -> list[Chunk]:
    """按标题层级切片：# 文档题 → ## 小节 → 段落聚合（≤400 字）。"""
    chunks: list[Chunk] = []
    current_section = ""
    buffer: list[str] = []

    def flush():
        text = "".join(buffer).strip()
        buffer.clear()
        if not text:
            return
        if len(text) <= _CHUNK_MAX_CHARS:
            chunks.append(_make_chunk(text, doc, current_section, len(chunks)))
            return
        # 超长小节按句子切分聚合
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        pack = ""
        for sent in sentences:
            if pack and len(pack) + len(sent) > _CHUNK_MAX_CHARS:
                chunks.append(_make_chunk(pack, doc, current_section, len(chunks)))
                pack = sent
            else:
                pack = f"{pack}{sent}"
        if pack:
            chunks.append(_make_chunk(pack, doc, current_section, len(chunks)))

    for line in raw.splitlines():
        if line.startswith("## "):
            flush()
            current_section = line[3:].strip()
        elif line.startswith("# "):
            flush()
        else:
            buffer.append(line + "\n")
    flush()
    return chunks


def _make_chunk(text: str, doc: str, section: str, idx: int) -> Chunk:
    label = f"{doc} · {section}" if section else doc
    # 标题双计：产品名/小节名常只出现在标题（正文用代称），
    # 双计可让「SD-WAN」类查询命中对应产品小节而非顺带提及的公司简介
    return Chunk(
        chunk_id=f"{doc}#{idx}",
        doc=doc,
        section=section,
        text=text,
        tokens=_tokenize(f"{label} {label} {text}"),
    )


class BM25Index:
    """标准 Okapi BM25（k1=1.5, b=0.75），纯 Python 实现。"""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_freq: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        for chunk in chunks:
            counts: dict[str, int] = {}
            for tok in chunk.tokens:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            self.doc_len.append(len(chunk.tokens))
            for tok in counts:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1
        self.n = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def _idf(self, tok: str) -> float:
        df = self.doc_freq.get(tok, 0)
        if df == 0:
            return 0.0
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str], idx: int) -> float:
        s = 0.0
        counts = self.tf[idx]
        dl = self.doc_len[idx]
        for tok in query_tokens:
            f = counts.get(tok, 0)
            if not f:
                continue
            s += self._idf(tok) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            )
        return s

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q_tokens = _tokenize(query)
        scored = [(self.chunks[i], self.score(q_tokens, i)) for i in range(self.n)]
        scored.sort(key=lambda x: -x[1])
        return [(c, s) for c, s in scored[:top_k] if s > 0]


@lru_cache(maxsize=1)
def load_index() -> BM25Index:
    """加载 data/kb/*.md 构建索引（进程内单例）。"""
    chunks: list[Chunk] = []
    if KB_DIR.exists():
        for path in sorted(KB_DIR.glob("*.md")):
            doc = path.stem
            raw = path.read_text(encoding="utf-8")
            chunks.extend(chunk_markdown(raw, doc))
    return BM25Index(chunks)


def search_kb(query: str, top_k: int = 3) -> list[dict]:
    """检索入口：返回 [{doc, section, text, score}]。"""
    hits = load_index().search(query, top_k=top_k)
    return [
        {"doc": c.doc, "section": c.section, "text": c.text, "score": round(s, 4)}
        for c, s in hits
    ]


def extractive_answer(query: str, hits: list[dict], max_sentences: int = 3) -> str:
    """抽取式回答：按命中排序取与查询重叠最高的句子，附引用。"""
    if not hits:
        return ""
    q_tokens = set(_tokenize(query))
    picked: list[str] = []
    seen_docs: list[str] = []
    for rank, hit in enumerate(hits):
        label = f"{hit['doc']} · {hit['section']}" if hit["section"] else hit["doc"]
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(hit["text"]) if len(s.strip()) >= 8]
        ranked = sorted(
            sentences,
            key=lambda s: -len(q_tokens & set(_tokenize(s))),
        )
        # 首个命中最多取 2 句（保留上下文细节，如按金/热线），其余命中各取 1 句
        quota = 2 if rank == 0 else 1
        for sent in ranked[:quota]:
            if len(picked) < max_sentences and sent not in picked:
                picked.append(sent.strip())
                seen_docs.append(label)
    body = "".join(f"- {s}\n" for s in picked)
    cites = "；".join(dict.fromkeys(seen_docs))
    return f"{body}\n（来源：{cites}）"

"""RAG 切片与 BM25 检索测试。"""

from app.language import normalize
from app.rag import _tokenize, chunk_markdown, extractive_answer, load_index, search_kb


def test_index_built_from_kb():
    index = load_index()
    assert index.n >= 20  # 6 份文档切片后应有不少于 20 个 chunk


def test_chunks_carry_doc_and_section():
    index = load_index()
    sample = index.chunks[0]
    assert sample.doc and isinstance(sample.section, str)
    assert sample.text.strip()


def test_chunk_size_bounded():
    index = load_index()
    for c in index.chunks:
        assert len(c.text) <= 450


def test_retrieval_simplified_query():
    hits = search_kb("国际漫游按金多少钱", top_k=3)
    assert hits and hits[0]["doc"] == "roaming_guide"


def test_retrieval_traditional_cantonese_query():
    # 繁体/粤语查询经归一化后命中简体知识库
    hits = search_kb(normalize("漫遊按金幾多錢"), top_k=3)
    assert hits and hits[0]["doc"] == "roaming_guide"


def test_retrieval_enterprise_en():
    hits = search_kb("SD-WAN branch deployment", top_k=3)
    assert hits and hits[0]["doc"].startswith("enterprise")


def test_retrieval_no_match_returns_empty():
    assert search_kb("zzzz qqqq xxxxx", top_k=3) == []


def test_extractive_answer_has_citation():
    hits = search_kb("一卡两号", top_k=2)
    answer = extractive_answer("一卡两号", hits)
    assert "来源" in answer
    assert "一卡两号" in answer


def test_tokenize_mixed_script():
    tokens = _tokenize("電訊5G網絡")
    assert "5g" in tokens
    assert "电讯" in tokens


def test_chunk_markdown_splits_sections():
    raw = "# T\n\nintro\n\n## A\n\nalpha content here\n\n## B\n\nbeta content here\n"
    chunks = chunk_markdown(raw, "demo")
    sections = [c.section for c in chunks]
    assert "A" in sections and "B" in sections

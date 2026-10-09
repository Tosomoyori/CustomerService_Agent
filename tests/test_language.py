"""三语识别与检索归一化测试。"""

from app.language import detect_language, normalize


def test_simplified_mandarin():
    assert detect_language("日本漫游多少钱一天") == "zh-CN"


def test_cantonese_marker_char():
    assert detect_language("日本漫遊幾多錢呀") == "zh-HK"


def test_cantonese_colloquial_words():
    assert detect_language("有冇大灣區plan推介") == "zh-HK"
    assert detect_language("上唔到網點算") == "zh-HK"


def test_traditional_maps_to_hk():
    assert detect_language("企業專線查詢") == "zh-HK"


def test_english():
    assert detect_language("How much is a Japan roaming pack?") == "en"
    assert detect_language("I want SD-WAN for 20 stores") == "en"


def test_empty_defaults_to_hk():
    assert detect_language("") == "zh-HK"


def test_normalize_cantonese_words():
    assert "哪里有得买" in normalize("邊度有得買")


def test_normalize_cantonese_chars():
    assert normalize("係咪有漫遊").startswith("是")
    assert "冇" not in normalize("有冇覆蓋")


def test_normalize_keeps_ascii():
    assert "sd-wan" in normalize("SD-WAN 组网").lower()

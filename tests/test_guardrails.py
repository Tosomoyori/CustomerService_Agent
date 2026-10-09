"""护栏测试：PII 脱敏、金额抽取与溯源、免责声明。"""

from app.guardrails import (
    ensure_disclaimer,
    extract_amounts,
    mask_contact,
    mask_pii,
    validate_amounts,
)


def test_mask_pii_hk_phone():
    masked = mask_pii("我电话系 9123 4567")
    assert "9123 4567" not in masked
    assert "912" in masked and "67" in masked


def test_mask_pii_mainland_phone():
    masked = mask_pii("打我电话 13612345678")
    assert "13612345678" not in masked
    assert "136****78" in masked


def test_mask_pii_hkid_and_email():
    masked = mask_pii("身份证 A123456(7)，邮箱 alice@example.com")
    assert "A123456" not in masked
    assert "alice@example.com" not in masked
    assert "a***@example.com" in masked


def test_public_hotline_not_masked():
    text = "请致电客服热线 2122 1188"
    assert mask_pii(text) == text


def test_mask_contact():
    assert mask_contact("91234567") == "912****67"
    assert mask_contact("abc") == "****"
    assert mask_contact("") == ""


def test_extract_amounts_variants():
    assert extract_amounts("月費 $128") == {128.0}
    assert extract_amounts("按金 300 港元") == {300.0}
    assert extract_amounts("漫游包 HK$98.5") == {98.5}
    assert extract_amounts("平到 $30蚊") == {30.0}
    assert extract_amounts("没有价格") == set()


def test_validate_amounts_catches_invention():
    violations = validate_amounts("推荐计划只要 $66/月", ["大灣區 $128/月", "$168/月"])
    assert violations == [66.0]


def test_validate_amounts_passes_sourced():
    assert validate_amounts("学生计划 $99/月，含 100GB", ["內地學生計劃 100GB $99"]) == []


def test_ensure_disclaimer_once():
    text = "查询结果：$98"
    once = ensure_disclaimer(text, "zh-HK")
    assert "官方渠道为准" in once
    assert ensure_disclaimer(once, "zh-HK") == once

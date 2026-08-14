from datetime import datetime

from hydrous.text_utils import compare_dates, content_words, date_key, parse_date, tokenize, word_overlap


def test_parse_longmemeval_format():
    assert parse_date("2023/04/10 (Mon) 23:07") == datetime(2023, 4, 10, 23, 7)


def test_parse_iso_and_plain():
    assert parse_date("2023-05-01") == datetime(2023, 5, 1)
    assert parse_date("2023-05-01 08:30") == datetime(2023, 5, 1, 8, 30)
    assert parse_date("2023-05-01T08:30:00") == datetime(2023, 5, 1, 8, 30)


def test_parse_garbage_returns_none():
    assert parse_date("not a date") is None
    assert parse_date("") is None
    assert parse_date(None) is None


def test_date_key_normalizes():
    assert date_key("2023/04/10 (Mon) 23:07") == "2023-04-10 23:07"
    assert date_key("2023-05-01") == "2023-05-01"


def test_compare_dates_mixed_formats():
    assert compare_dates("2023/04/10 (Mon) 23:07", "2023-04-11") < 0
    assert compare_dates("2023-05-01", "2023/05/01 (Mon) 00:00") == 0
    assert compare_dates("2023-06-15", "2023-05-01") > 0


def test_tokenize_and_content_words():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]
    assert "the" not in content_words("the quick fox")
    assert "quick" in content_words("the quick fox")


def test_word_overlap():
    assert word_overlap("favorite color", "my favorite color is blue") == 1.0
    assert word_overlap("shoe size", "cooking italian food") == 0.0
    assert word_overlap("", "anything") == 0.0

from src.preprocessing import clean_text


def test_lowercase_conversion():
    result = clean_text("HELLO WORLD")
    assert result == "hello world"


def test_special_character_removal():
    result = clean_text("Hello!!! 123")
    assert result == "hello"
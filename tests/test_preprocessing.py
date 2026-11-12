
from rafiai.preprocessing import clean_text


def test_clean_text_lowercase():
    assert clean_text("I LOVE THIS") == "i love this"


def test_clean_text_cannot():
    assert "NOT_rely" in clean_text("I cannot rely on this application")


def test_clean_text_could_not():
    assert "NOT_finish" in clean_text("I could not finish the task")


def test_clean_text_no_difficulty():
    assert "NOT_difficulty" in clean_text(
        "I had no difficulty completing the task"
    )


def test_clean_text_without_difficulties():
    assert "NOT_difficulties" in clean_text(
        "Everything was completed without any difficulties"
    )


def test_clean_text_never():
    assert "NOT_use" in clean_text("I never use this application")


def test_clean_text_not_a_problem():
    assert "NOT_problem" in clean_text(
        "This application is not a problem to use"
    )


def test_clean_text_not_a_bad_decision():
    assert "NOT_bad" in clean_text(
        "This was not a bad decision"
    )

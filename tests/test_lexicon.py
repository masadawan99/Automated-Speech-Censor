"""Unit tests for censor.lexicon two-tier lookup and normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from censor.lexicon import ProfanityLexicon, normalize_token


@pytest.fixture(scope="module")
def lexicon() -> ProfanityLexicon:
    csv_path = Path(__file__).resolve().parent.parent / "lexicon" / "profanity_lexicon.csv"
    return ProfanityLexicon.from_csv(csv_path)


def test_tier_a_unambiguous_and_inflections(lexicon: ProfanityLexicon) -> None:
    for tok in (
        "fuck",
        "Fucking!",
        "fucked",
        "fuckin'",
        "SHIT",
        "bullshit,",
        "asshole",
        "mother-fucker",
        "ass",
        "Dick",
        "bitch",
        "bitch's",
        "cock",
        "pussy",
        "bastard",
        "whore",
        "slut",
        "blowjob",
        "nigga",
        "niggas",
        "nigger",
        "negro",
        "negros",
        "negroes",
        "chink",
        "spic",
        "paki",
    ):
        norm, tier = lexicon.classify_word(tok)
        assert tier == "TIER_A", f"Expected TIER_A for {tok!r}, got {tier} (norm={norm!r})"


def test_tier_b_context_dependent(lexicon: ProfanityLexicon) -> None:
    for tok in ("prick", "pricks", "balls", "hoe", "hoes", "ho", "tit", "boob", "sod", "tramp", "spunk"):
        _, tier = lexicon.classify_word(tok)
        assert tier == "TIER_B", f"Expected TIER_B for {tok!r}, got {tier}"


def test_scunthorpe_and_clean_words_stay_clean(lexicon: ProfanityLexicon) -> None:
    for tok in (
        "class",
        "assess",
        "Dickens",
        "cocktail",
        "cockpit",
        "Scunthorpe",
        "assistant",
        "titmouse",
        "hello",
        "raccoon",
        "cocoon",
        "spicy",
    ):
        _, tier = lexicon.classify_word(tok)
        assert tier == "CLEAN", f"False positive on clean word {tok!r}: got {tier}"


def test_mild_intensifiers_stay_clean(lexicon: ProfanityLexicon) -> None:
    for tok in ("hell", "damn", "damned", "dammit", "crap", "crappy", "heck", "darn"):
        _, tier = lexicon.classify_word(tok)
        assert tier == "CLEAN", f"Expected CLEAN for mild word {tok!r}, got {tier}"


def test_whisper_asterisk_masked_tokens_are_tier_a(lexicon: ProfanityLexicon) -> None:
    for tok in ("f***", "s**t", "****", "b*tch"):
        norm, tier = lexicon.classify_word(tok)
        assert tier == "TIER_A", f"Expected TIER_A for masked token {tok!r}, got {tier}"
        assert norm != ""


def test_normalize_token() -> None:
    assert normalize_token(" Fucking! ") == "fucking"
    assert normalize_token("****") == "*"

"""Two-tier profanity lexicon lookup, normalization, and compound handling."""

from __future__ import annotations

import csv
from pathlib import Path
import re

DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parent.parent / "lexicon" / "profanity_lexicon.csv"
)

MILD_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "hell",
        "hells",
        "damn",
        "damned",
        "dammit",
        "damnit",
        "goddamn",
        "goddamned",
        "crap",
        "crappy",
        "craps",
        "heck",
        "darn",
        "dang",
    }
)

_OUTER_PUNCT_RE = re.compile(r"^[^a-zA-Z0-9*]+|[^a-zA-Z0-9*']+$")
_ALNUM_RE = re.compile(r"[^a-z0-9]")


def normalize_token(token: str) -> str:
    """Return lowercased alphanumeric normalized form of a raw ASR token."""
    cleaned = _ALNUM_RE.sub("", token.lower())
    if not cleaned and "*" in token:
        return "*"
    return cleaned


class ProfanityLexicon:
    """Two-tier English profanity lexicon (TIER_A unambiguous, TIER_B context-dependent)."""

    def __init__(
        self,
        tier_a: set[str] | frozenset[str],
        tier_b: set[str] | frozenset[str],
    ) -> None:
        self.tier_a: frozenset[str] = frozenset(w.strip().lower() for w in tier_a if w.strip())
        self.tier_b: frozenset[str] = frozenset(
            w.strip().lower() for w in tier_b if w.strip() and w.strip().lower() not in self.tier_a
        )

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> ProfanityLexicon:
        """Load lexicon from CSV with columns ``word,tier``."""
        csv_path = Path(path) if path is not None else DEFAULT_LEXICON_PATH
        tier_a: set[str] = set()
        tier_b: set[str] = set()
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = (row.get("word") or "").strip().lower()
                tier = (row.get("tier") or "").strip().upper()
                if not word or word in MILD_EXCLUSIONS:
                    continue
                if tier == "TIER_A":
                    tier_a.add(word)
                elif tier == "TIER_B":
                    tier_b.add(word)
        return cls(tier_a=tier_a, tier_b=tier_b)

    def _lookup_exact(self, candidate: str) -> str:
        if not candidate or candidate in MILD_EXCLUSIONS:
            return "CLEAN"
        if candidate in self.tier_a:
            return "TIER_A"
        if candidate in self.tier_b:
            return "TIER_B"
        return "CLEAN"

    def _lookup_variants(self, cleaned: str) -> str:
        # 1. Exact lowercased token
        tier = self._lookup_exact(cleaned)
        if tier != "CLEAN":
            return tier

        # 2. Strip possessive ('s / 's) or trailing apostrophe (fuckin', bitch's)
        variants = [cleaned]
        for suffix in ("'s", "’s", "'", "’"):
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                variants.append(cleaned[: -len(suffix)])

        for v in variants:
            tier = self._lookup_exact(v)
            if tier != "CLEAN":
                return tier
            alnum = _ALNUM_RE.sub("", v)
            tier = self._lookup_exact(alnum)
            if tier != "CLEAN":
                return tier

        # 3. Hyphenated compounds: check both joined form and parts (TIER_A > TIER_B)
        if "-" in cleaned:
            parts = [p for p in cleaned.split("-") if p]
            best = "CLEAN"
            for part in parts:
                part_tier = self._lookup_variants(part)
                if part_tier == "TIER_A":
                    return "TIER_A"
                if part_tier == "TIER_B":
                    best = "TIER_B"
            if best != "CLEAN":
                return best

        return "CLEAN"

    def classify_word(self, raw_word: str) -> tuple[str, str]:
        """Classify a raw ASR word token into ``(norm, tier)``.

        Returns:
            ``(norm, tier)`` where ``tier`` is ``"CLEAN"``, ``"TIER_A"``, or ``"TIER_B"``.
        """
        norm = normalize_token(raw_word)
        # Any Whisper '*' masked profanity token (e.g. 'f***', 's**t', '****') -> TIER_A
        if "*" in raw_word:
            return norm, "TIER_A"

        stripped = _OUTER_PUNCT_RE.sub("", raw_word.strip().lower().replace("’", "'"))
        if not stripped:
            return norm, "CLEAN"

        tier = self._lookup_variants(stripped)
        return norm, tier

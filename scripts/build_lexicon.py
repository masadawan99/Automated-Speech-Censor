"""Build and clean the two-tier English profanity lexicon (lexicon/profanity_lexicon.csv).

Downloads the LDNOOBW English list, removes multi-word phrases, mild intensifiers,
and noisy everyday/anatomy terms, merges curated Tier A (unambiguous) and Tier B
(context-dependent) profanity plus common spoken inflections, and writes
``lexicon/profanity_lexicon.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
import urllib.request

LDNOOBW_EN_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
)

# Mild intensifiers intentionally excluded per requirements & implementation plan
MILD_EXCLUSIONS: set[str] = {
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
    "bloody",
    "suck",
    "sucks",
    "sucked",
    "sucking",
    "jerk",
    "jerks",
    "screw",
    "screwed",
    "screwing",
    "butt",
    "butts",
}

# Unambiguous profanity + common spoken ASR inflections -> ALWAYS beeped (Tier A)
TIER_A_CORE: set[str] = {
    "fuck",
    "fucks",
    "fucked",
    "fucker",
    "fuckers",
    "fucking",
    "fuckin",
    "fuckface",
    "fuckhead",
    "fucktard",
    "fucktards",
    "fuckwit",
    "fuckoff",
    "fuckup",
    "fuckedup",
    "motherfuck",
    "motherfucker",
    "motherfuckers",
    "motherfucking",
    "motherfuckin",
    "clusterfuck",
    "mindfuck",
    "shit",
    "shits",
    "shitty",
    "shitting",
    "shitted",
    "shitter",
    "shitblimp",
    "shithead",
    "shitface",
    "shitshow",
    "shitstorm",
    "bullshit",
    "bullshitting",
    "dipshit",
    "horseshit",
    "apeshit",
    "batshit",
    "chickenshit",
    "jackshit",
    "cunt",
    "cunts",
    "cuntface",
    "ass",
    "asses",
    "arse",
    "arses",
    "asshole",
    "assholes",
    "arsehole",
    "arseholes",
    "dumbass",
    "dumbasses",
    "jackass",
    "jackasses",
    "badass",
    "smartass",
    "fatass",
    "lardass",
    "dick",
    "dicks",
    "dickhead",
    "dickheads",
    "dickface",
    "dickwad",
    "cock",
    "cocks",
    "cocksucker",
    "cocksuckers",
    "cocksucking",
    "bitch",
    "bitches",
    "bitching",
    "bitchin",
    "bitchy",
    "bitched",
    "pussy",
    "pussies",
    "bastard",
    "bastards",
    "whore",
    "whores",
    "whoring",
    "slut",
    "sluts",
    "slutty",
    "skank",
    "skanks",
    "skanky",
    "piss",
    "pissed",
    "pisser",
    "pissing",
    "douche",
    "douches",
    "douchebag",
    "douchebags",
    "scumbag",
    "scumbags",
    "twat",
    "twats",
    "wank",
    "wanker",
    "wankers",
    "wanking",
    "bollocks",
    "bugger",
    "buggers",
    "turd",
    "turds",
    "crapper",
    "cum",
    "cums",
    "cumming",
    "jizz",
    "boner",
    "boners",
    "dildo",
    "dildos",
    "blowjob",
    "blowjobs",
    "handjob",
    "handjobs",
    "rimjob",
    "rimjobs",
    "boobs",
    "tits",
    "titty",
    "titties",
    # Racial, ethnic, homophobic, and ableist slurs
    "nigga",
    "niggas",
    "niggaz",
    "nigger",
    "niggers",
    "sandnigger",
    "sandniggers",
    "negro",
    "negros",
    "negroes",
    "coon",
    "coons",
    "darkie",
    "darkies",
    "darky",
    "jigaboo",
    "jigaboos",
    "chink",
    "chinks",
    "gook",
    "gooks",
    "slanteye",
    "slanteyes",
    "zipperhead",
    "zipperheads",
    "jap",
    "japs",
    "kike",
    "kikes",
    "spic",
    "spics",
    "wetback",
    "wetbacks",
    "beaner",
    "beaners",
    "paki",
    "pakis",
    "raghead",
    "ragheads",
    "towelhead",
    "towelheads",
    "redskin",
    "redskins",
    "injun",
    "injuns",
    "honky",
    "honkey",
    "honkies",
    "wop",
    "wops",
    "dago",
    "dagos",
    "kraut",
    "krauts",
    "gyppo",
    "gyppos",
    "faggot",
    "faggots",
    "fag",
    "fags",
    "dyke",
    "dykes",
    "tranny",
    "trannies",
    "retard",
    "retarded",
    "retards",
}

# Context-dependent profanity + inflections -> disambiguated by classifier (Tier B)
TIER_B_CORE: set[str] = {
    "balls",
    "boob",
    "ho",
    "hoe",
    "hoes",
    "prick",
    "pricks",
    "sod",
    "spunk",
    "tit",
    "tramp",
}

# Strong unambiguous stems whose single-word LDNOOBW derivatives are safe as Tier A
STRONG_TIER_A_STEMS: tuple[str, ...] = (
    "fuck",
    "shit",
    "cunt",
    "asshole",
    "arsehole",
    "cocksuck",
    "dickhead",
    "motherfuck",
    "twat",
    "wank",
)

# Everyday/medical/noisy words present in LDNOOBW that cause false positives
NOISY_EXCLUSIONS: set[str] = {
    "adult",
    "anal",
    "anus",
    "bang",
    "big",
    "black",
    "breast",
    "breasts",
    "clit",
    "clitoris",
    "crack",
    "date",
    "erect",
    "erection",
    "finger",
    "gay",
    "girl",
    "girls",
    "hard",
    "head",
    "hole",
    "horny",
    "hot",
    "knob",
    "lesbian",
    "naked",
    "naughty",
    "nude",
    "nudity",
    "orgasm",
    "penis",
    "pipe",
    "pole",
    "porn",
    "porno",
    "pornography",
    "pubic",
    "rectum",
    "rim",
    "rod",
    "sack",
    "scrotum",
    "semen",
    "sex",
    "sexual",
    "sexy",
    "shaft",
    "skin",
    "snatch",
    "sperm",
    "spit",
    "stiff",
    "stroke",
    "swallow",
    "tail",
    "testicle",
    "testicles",
    "tight",
    "tool",
    "vagina",
    "vaginal",
    "vulva",
    "wet",
    "wood",
    "xxx",
}


def fetch_ldnoobw_words(timeout: float = 15.0) -> list[str]:
    """Download the LDNOOBW English list; return empty list if offline."""
    try:
        req = urllib.request.Request(
            LDNOOBW_EN_URL,
            headers={"User-Agent": "automated-speech-censor/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        return [line.strip().lower() for line in text.splitlines() if line.strip()]
    except Exception as exc:
        print(f"[build_lexicon] Warning: could not fetch LDNOOBW ({exc}); using curated core only.")
        return []


def build_lexicon(output_path: Path) -> dict[str, str]:
    """Construct the cleaned word -> tier mapping and write it to CSV."""
    ldnoobw_raw = fetch_ldnoobw_words()
    entries: dict[str, str] = {}

    for raw in ldnoobw_raw:
        # Drop multi-word phrases and non-alphabetic entries
        if " " in raw or "-" in raw or not re.fullmatch(r"[a-z]+", raw):
            continue
        if len(raw) < 2:
            continue
        if raw in MILD_EXCLUSIONS or raw in NOISY_EXCLUSIONS:
            continue
        if any(stem in raw for stem in STRONG_TIER_A_STEMS):
            entries[raw] = "TIER_A"

    # Merge curated Tier B first, then Tier A so Tier A always takes precedence
    for word in TIER_B_CORE:
        if word not in MILD_EXCLUSIONS:
            entries[word] = "TIER_B"

    for word in TIER_A_CORE:
        if word not in MILD_EXCLUSIONS:
            entries[word] = "TIER_A"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "tier"])
        for word in sorted(entries):
            writer.writerow([word, entries[word]])

    tier_a_count = sum(1 for t in entries.values() if t == "TIER_A")
    tier_b_count = sum(1 for t in entries.values() if t == "TIER_B")
    print(
        f"[build_lexicon] Wrote {len(entries)} words "
        f"(TIER_A={tier_a_count}, TIER_B={tier_b_count}) -> {output_path}"
    )
    return entries


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_csv = repo_root / "lexicon" / "profanity_lexicon.csv"
    build_lexicon(out_csv)


if __name__ == "__main__":
    main()

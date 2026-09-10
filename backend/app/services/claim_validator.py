"""
Claim Validator 2.0 — verifies specific factual claims in the generated
response against the REAL chart/Dasha data actually fed to the LLM, not
just plausible-sounding text. Three independent checks:

1. Year/timeframe claims — must match the real Dasha timeline (existing).
2. Absolute language — blocked when evidence confidence is low (existing).
3. Chart-fact claims — planet-in-sign and planet-in-house statements must
   match the actual computed chart. This catches the most dangerous
   hallucination: the LLM confidently stating a WRONG planet placement.
4. AI Giveaway phrases — robotic phrases that break the immersion (e.g.
   "according to the database", "as per the provided JSON") are blocked.
"""
import re
from typing import List, Optional, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABSOLUTE_CLAIM_PATTERNS = [
    r"\bwill definitely\b", r"\byou will surely\b", r"\bguaranteed\b",
    r"\b100%\b", r"\bcertainly will\b",
    r"\bpakka\b.*\bhoga\b", r"\bzaroor\b.*\bhoga\b",
]

# Phrases that make the AI sound like a robot/database and break immersion.
AI_GIVEAWAY_PHRASES = [
    r"according to the (database|data|json|provided data|system)",
    r"as per the (provided json|json data|database|system|classical texts)",
    r"the (database|json|data) (shows|says|indicates|states|contains)",
    r"based on the (provided json|database|data provided to me)",
    r"the (provided|given) chart data (shows|says|indicates)",
    r"i (cannot|can't|am unable to) access",
    r"i don'?t have access to",
    r"as an ai",
    r"as a language model",
    r"i'?m just an? (ai|language model|chatbot)",
]
AI_GIVEAWAY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in AI_GIVEAWAY_PHRASES]

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

PLANET_NAMES = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Word-form house numbers so we catch "in the tenth house" as well as "10th house"
HOUSE_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12,
}
HOUSE_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(PLANET_NAMES) + r")\b[^.]{0,30}?\b("
    + "|".join(HOUSE_WORDS.keys()) + r")\s+house\b",
    re.IGNORECASE
)

# Matches phrases like "Mercury is in Virgo", "Mercury in Virgo", "Saturn placed in Libra"
PLANET_SIGN_PATTERN = re.compile(
    r"\b(" + "|".join(PLANET_NAMES) + r")\b[^.]{0,25}?\b(" + "|".join(ZODIAC_SIGNS) + r")\b",
    re.IGNORECASE
)

# Matches phrases like "Mercury in the 1st house", "10th house lord Mercury"
# Covers both numeric (1st, 10th) and word-form (first, tenth)
PLANET_HOUSE_NUMERIC_PATTERN = re.compile(
    r"\b(" + "|".join(PLANET_NAMES) + r")\b[^.]{0,20}?\b(1[0-2]|[1-9])(?:st|nd|rd|th)\s+house\b",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_years_from_timeline(dasha_timeline: str) -> List[int]:
    if not dasha_timeline:
        return []
    return [int(y) for y in YEAR_PATTERN.findall(dasha_timeline)]


def _get_house_for_sign(sign_name: str, ascendant_sign: str) -> Optional[int]:
    try:
        asc_idx = ZODIAC_SIGNS.index(ascendant_sign)
        sign_idx = ZODIAC_SIGNS.index(sign_name)
        return ((sign_idx - asc_idx) % 12) + 1
    except ValueError:
        return None


def _verify_planet_sign_claims(text: str, planets: List[dict]) -> List[str]:
    """Checks every 'Planet in Sign' claim in the response against the
    real chart. Returns a failure message for each mismatch."""
    failures = []
    if not planets:
        return failures

    actual_signs = {p.get("name"): p.get("sign_name") for p in planets if p.get("name")}

    for match in PLANET_SIGN_PATTERN.finditer(text):
        claimed_planet = match.group(1).strip().capitalize()
        claimed_sign = match.group(2).strip().capitalize()

        actual_sign = actual_signs.get(claimed_planet)
        if actual_sign and actual_sign.lower() != claimed_sign.lower():
            failures.append(
                f"The response states '{claimed_planet} is in {claimed_sign}', but the actual "
                f"chart data shows {claimed_planet} is in {actual_sign}. Correct this specific "
                f"placement — do not guess planet positions, use only the chart data provided."
            )

    return failures


def _verify_planet_house_claims(text: str, planets: List[dict], ascendant_sign: Optional[str]) -> List[str]:
    """Checks every 'Planet in Nth house' claim against the real chart.
    Covers both numeric (1st, 10th) and word-form (first, tenth) house mentions.
    """
    failures = []
    if not planets or not ascendant_sign:
        return failures

    # Build lookup: planet name → actual house number
    actual_houses: Dict[str, int] = {}
    for p in planets:
        pname = p.get("name")
        psign = p.get("sign_name", "")
        if pname and psign:
            h = _get_house_for_sign(psign, ascendant_sign)
            if h:
                actual_houses[pname] = h

    def _check_claim(claimed_planet: str, claimed_house: int) -> Optional[str]:
        claimed_planet = claimed_planet.strip().capitalize()
        actual_house = actual_houses.get(claimed_planet)
        if actual_house and actual_house != claimed_house:
            return (
                f"The response states '{claimed_planet} is in the {claimed_house}th house', but "
                f"based on the actual chart, {claimed_planet} is in the {actual_house}th house. "
                f"Correct this — do not state a house placement that doesn't match the chart data provided."
            )
        return None

    # Numeric form: "1st house", "10th house"
    for match in PLANET_HOUSE_NUMERIC_PATTERN.finditer(text):
        msg = _check_claim(match.group(1), int(match.group(2)))
        if msg:
            failures.append(msg)

    # Word form: "first house", "tenth house"
    for match in HOUSE_WORD_PATTERN.finditer(text):
        house_num = HOUSE_WORDS.get(match.group(2).lower())
        if house_num:
            msg = _check_claim(match.group(1), house_num)
            if msg:
                failures.append(msg)

    return failures


def _verify_ai_giveaways(text: str) -> List[str]:
    """Scans the response for robotic / immersion-breaking phrases that make
    the AI sound like a database query engine rather than a wise astrologer.
    Returns one failure per detected giveaway."""
    failures = []
    for pattern in AI_GIVEAWAY_PATTERNS:
        match = pattern.search(text)
        if match:
            snippet = match.group(0)
            failures.append(
                f"The response contains an immersion-breaking AI phrase: \"{snippet}\". "
                f"Rewrite this response WITHOUT mentioning databases, JSON, AI, or data systems. "
                f"Speak as a genuine Vedic astrologer who has studied the birth chart directly."
            )
            break  # One failure is enough to trigger a regeneration
    return failures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_claims(
    response_text: str,
    dasha_timeline: str = "",
    evidence_vote: Optional[Dict] = None,
    planets: Optional[List[dict]] = None,
    ascendant_sign: Optional[str] = None,
) -> List[str]:
    """Returns a list of specific validation failures across FOUR checks:
    1. Year/timeframe claims vs. real Dasha timeline
    2. Absolute/deterministic language vs. evidence confidence
    3. Planet-sign / planet-house claims vs. the real computed chart
    4. AI giveaway phrases that break the immersion
    """
    failures = []
    text = response_text.strip()
    if not text:
        return failures

    # --- Check 1: Year claims ---
    mentioned_years = [int(y) for y in YEAR_PATTERN.findall(text)]
    if mentioned_years and dasha_timeline:
        valid_years = set(_extract_years_from_timeline(dasha_timeline))
        for year in mentioned_years:
            if valid_years and year not in valid_years:
                if not any(abs(year - vy) <= 1 for vy in valid_years):
                    failures.append(
                        f"The response states the year {year}, but this does not match any "
                        f"period in the actual Dasha timeline provided. Only cite years/periods "
                        f"that appear in the timeline data — if unsure of an exact year, describe "
                        f"the period by Dasha lord name instead of a specific year."
                    )
    elif mentioned_years and not dasha_timeline:
        failures.append(
            f"The response states a specific year ({mentioned_years[0]}) but no real Dasha "
            f"timeline data was available. Do not invent a specific year without timeline "
            f"evidence — speak in terms of Dasha periods instead."
        )

    # --- Check 2: Absolute language vs confidence ---
    has_absolute = any(re.search(p, text, re.IGNORECASE) for p in ABSOLUTE_CLAIM_PATTERNS)
    if has_absolute:
        confidence = evidence_vote.get("confidence_pct", 50) if evidence_vote else 50
        verdict = evidence_vote.get("verdict") if evidence_vote else None
        if confidence < 70 or verdict == "mixed":
            failures.append(
                f"The response uses absolute/guaranteed language (e.g. 'will definitely', "
                f"'guaranteed', '100%'), but the evidence confidence is only {confidence}% "
                f"({verdict or 'uncertain'}). Remove absolute claims — astrology should never "
                f"be stated as a certainty, especially when evidence is mixed or moderate."
            )

    # --- Check 3: Chart-fact verification ---
    if planets:
        failures.extend(_verify_planet_sign_claims(text, planets))
        if ascendant_sign:
            failures.extend(_verify_planet_house_claims(text, planets, ascendant_sign))

    # --- Check 4: AI Giveaway phrases ---
    failures.extend(_verify_ai_giveaways(text))

    return failures


def build_claim_correction_instructions(failures: List[str]) -> str:
    if not failures:
        return ""
    lines = ["IMPORTANT — the following claims in your previous answer are unsupported or incorrect, fix them:"]
    for i, f in enumerate(failures, 1):
        lines.append(f"{i}. {f}")
    return "\n".join(lines)


def build_streamed_correction_note(failures: List[str], language: str = "Hinglish") -> str:
    """For STREAMED responses where regeneration isn't possible — appends a
    brief, honest correction note after the fact rather than leaving a
    detected factual error uncorrected in the user's view. Only used when
    validate_claims() finds a real chart-fact mismatch."""
    if not failures:
        return ""

    labels = {
        "English": "\n\n📝 Correction: ",
        "Hindi": "\n\n📝 सुधार: ",
        "Hinglish": "\n\n📝 Correction: ",
    }
    prefix = labels.get(language, labels["Hinglish"])

    correction_lines = []
    for f in failures:
        # Only surface chart-fact mismatches to the user (not giveaway or absolute-language failures
        # which are internal quality issues, not public-facing corrections)
        if "actual chart" in f.lower() or "actual chart data shows" in f.lower():
            correction_lines.append(f.split(".")[0] + ".")

    if not correction_lines:
        return ""

    return prefix + " ".join(correction_lines)
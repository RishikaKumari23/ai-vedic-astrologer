"""
Next-Best-Action Recommender System (RecSys) — Phase 1: Rule-Based Engine
=========================================================================

WHAT IS THIS? (RecSys Learning Note)
-------------------------------------
The simplest form of a Recommender System is NOT machine learning — it is a set of
smart rules written by a domain expert. This is called a "Knowledge-Based Recommender"
or "Expert System". Medical diagnosis systems, legal advisory tools, and traditional
astrology apps all use this approach.

WHY START HERE?
- No training data required (we don't have user interaction logs yet)
- Immediately produces highly relevant, personalized results
- Easy to debug and explain ("this question appeared because user has Saturn Mahadasha")
- A great baseline to beat with the ML phases we'll build next

HOW IT WORKS:
1. Read the user's chart signals: Dasha lord, retrograde planets, house placements
2. Apply a priority-ordered list of rules (like a decision tree)
3. Each matching rule contributes candidate questions
4. Return the top 3 highest-scoring candidates as suggestions

FILES THAT CALL THIS:
- chat_service.py: replaces get_instant_suggestions(topic, language) call
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION TEMPLATES
# Each template uses {planet}, {sign}, {house}, {dasha_lord} as fill-in slots.
# We fill them at runtime with the actual user chart data.
# ─────────────────────────────────────────────────────────────────────────────

# Questions triggered by which planet is the current Mahadasha lord
DASHA_LORD_QUESTIONS = {
    "Saturn": [
        "How should I navigate delays and setbacks during my Saturn Mahadasha?",
        "What areas of life does Saturn Mahadasha most impact for a {ascendant} Ascendant?",
        "What remedies help ease the challenges of Saturn Mahadasha?",
    ],
    "Rahu": [
        "How does Rahu Mahadasha affect my ambitions and sudden changes?",
        "What unexpected events or obsessions does Rahu Mahadasha bring?",
        "Are there any remedies to reduce the confusion during Rahu Mahadasha?",
    ],
    "Ketu": [
        "How does Ketu Mahadasha create spiritual detachment in my life?",
        "Why do I feel directionless or isolated during Ketu Mahadasha?",
        "What is the spiritual purpose of my Ketu Mahadasha?",
    ],
    "Jupiter": [
        "Which life areas will expand the most during my Jupiter Mahadasha?",
        "How does Jupiter Mahadasha affect wealth and wisdom for my chart?",
        "Is Jupiter Mahadasha favorable for marriage and children in my chart?",
    ],
    "Mars": [
        "How does Mars Mahadasha affect my energy, anger, and ambitions?",
        "Is my Mars Mahadasha good for career and competition?",
        "What are the health risks to watch during Mars Mahadasha?",
    ],
    "Venus": [
        "How does Venus Mahadasha affect love, relationships, and luxury?",
        "Is Venus Mahadasha favorable for marriage in my chart?",
        "How does Venus Mahadasha support creativity and wealth?",
    ],
    "Mercury": [
        "How does Mercury Mahadasha affect my communication and intellect?",
        "Is Mercury Mahadasha good for business and education?",
        "How does Mercury interact with my other planets during this Dasha?",
    ],
    "Moon": [
        "How does Moon Mahadasha affect my emotions and mental peace?",
        "What does Moon Mahadasha mean for relationships and my mother?",
        "Is Moon Mahadasha favorable for travel or change of residence?",
    ],
    "Sun": [
        "How does Sun Mahadasha strengthen my confidence and authority?",
        "What does Sun Mahadasha mean for my career and father's influence?",
        "How does Sun Mahadasha affect my health and vitality?",
    ],
}

# Questions triggered when a specific planet is retrograde (Vakri)
RETROGRADE_QUESTIONS = {
    "Saturn": [
        "How does retrograde Saturn in my chart create karmic delays in career?",
        "What is the deeper spiritual lesson of my retrograde Saturn?",
        "Does retrograde Saturn in house {house} affect my responsibilities differently?",
    ],
    "Jupiter": [
        "How does retrograde Jupiter in my chart affect luck and wisdom?",
        "Does retrograde Jupiter delay marriage or children in my chart?",
        "What is the karmic meaning of retrograde Jupiter in house {house}?",
    ],
    "Mars": [
        "How does retrograde Mars change the expression of my ambitions?",
        "Does retrograde Mars in my chart create internal anger or frustration?",
        "What remedies help channel retrograde Mars energy positively?",
    ],
    "Mercury": [
        "How does retrograde Mercury in my chart affect my communication style?",
        "Does retrograde Mercury cause misunderstandings in my relationships?",
        "Is retrograde Mercury in house {house} good or bad for business?",
    ],
    "Venus": [
        "How does retrograde Venus in my chart affect love and relationships?",
        "Does retrograde Venus cause old relationships to return?",
        "What is the karmic pattern retrograde Venus creates in my love life?",
    ],
    "Rahu": [
        "How does retrograde Rahu intensify my obsessions and desires?",
        "What karmic lessons does retrograde Rahu carry in my chart?",
    ],
    "Ketu": [
        "How does retrograde Ketu deepen my spiritual detachment?",
        "What past-life karma does retrograde Ketu in house {house} represent?",
    ],
}

# Questions triggered by specific house placements (planet in a significant house)
HOUSE_PLACEMENT_QUESTIONS = {
    7: [
        "How does {planet} in my 7th house affect my marriage prospects?",
        "What kind of life partner does {planet} in the 7th house indicate for me?",
    ],
    10: [
        "How does {planet} in my 10th house affect my career and reputation?",
        "Does {planet} in the 10th house make me suited for government or business?",
    ],
    8: [
        "How does {planet} in my 8th house affect my longevity and hidden matters?",
        "What does {planet} in the 8th house mean for sudden events in my life?",
    ],
    5: [
        "How does {planet} in my 5th house affect children and creativity?",
        "What does {planet} in the 5th house say about my intelligence and past karma?",
    ],
    12: [
        "How does {planet} in my 12th house affect my expenses and spirituality?",
        "Does {planet} in the 12th house indicate foreign travel or settlement?",
    ],
    1: [
        "How does {planet} in my 1st house shape my personality and health?",
        "What strengths and weaknesses does {planet} in the Ascendant give me?",
    ],
}

# Questions triggered by the topic the user just asked about (context follow-ups)
TOPIC_FOLLOWUP_QUESTIONS = {
    "marriage": [
        "What does my Dasha say about the timing of my marriage?",
        "Does my chart show a love marriage or an arranged marriage?",
        "What planet is blocking or delaying my marriage?",
    ],
    "career": [
        "Which Dasha period is best for a promotion or job change?",
        "Does my chart favor business, government, or private sector?",
        "How strong is my 10th house for a leadership role?",
    ],
    "finance": [
        "Which planet in my chart controls my wealth and income?",
        "What is the best Dasha period for financial growth in my chart?",
        "Does my chart indicate sudden wealth or gradual accumulation?",
    ],
    "health": [
        "Which planets in my chart are affecting my health right now?",
        "What does my 6th house say about chronic health issues?",
        "What Vedic remedies help boost my overall vitality?",
    ],
    "education": [
        "Is my chart favorable for higher studies or going abroad?",
        "What does my 4th and 9th house say about my education?",
        "Which Dasha period is best for completing my degree or exams?",
    ],
    "abroad": [
        "Does my chart indicate a permanent move or just travel abroad?",
        "Which planets and houses in my chart support foreign settlement?",
        "What is the best Dasha for opportunities abroad?",
    ],
    "children": [
        "How strong is my 5th house for having children?",
        "What does my chart say about the timing of having a child?",
        "Which planet governs my 5th house and how strong is it?",
    ],
    "remedies": [
        "What gemstone is most beneficial for my Ascendant and chart?",
        "Are there any specific mantras that would help my current Dasha?",
        "What dietary or lifestyle changes does my chart recommend?",
    ],
}

# Fallback questions shown when no rules match
DEFAULT_QUESTIONS = [
    "What does my current Mahadasha mean for the next 2 years?",
    "Which is my strongest planet and what does it mean for me?",
    "What does my Ascendant sign reveal about my personality?",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


def _compute_house(planet_sign: str, ascendant_sign: str) -> Optional[int]:
    """
    Vedic Whole Sign System:
        house = ((planet_sign_index - ascendant_sign_index) % 12) + 1
    """
    try:
        planet_idx = ZODIAC_SIGNS.index(planet_sign)
        asc_idx = ZODIAC_SIGNS.index(ascendant_sign)
        return ((planet_idx - asc_idx) % 12) + 1
    except (ValueError, TypeError):
        return None


def _fill(template: str, **kwargs) -> str:
    """Safely fills a question template with chart data."""
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_rule_based_suggestions(
    session: dict,
    topic: Optional[str],
    language: str = "English"
) -> list[str]:
    """
    Phase 1 RecSys — Knowledge-Based / Expert Rule Engine.

    Signal priority order (highest to lowest):
      1. Current Mahadasha lord  → drives 6-20 year life theme
      2. Retrograde (Vakri) planets → unusual, karmic effects
      3. Sensitive house placements  → 7th, 10th, 8th, 5th, 12th, 1st
      4. Current conversation topic  → what the user is already exploring

    Returns top 3 deduplicated question strings.
    """
    candidates: list[str] = []

    # ── Parse chart data from session cache ──────────────────────────────────
    planets: list[dict] = []
    ascendant_sign: str = ""
    dasha_lord: str = ""

    try:
        raw = session.get("kundli_raw")
        if raw:
            parsed = json.loads(raw)
            planets = parsed.get("planets", []) or []
            ascendant_sign = parsed.get("ascendant_sign", "")
    except Exception as e:
        logger.warning(f"[RecSys P1] kundli_raw parse error: {e}")

    try:
        dasha_raw = session.get("kundli_dasha")
        if dasha_raw:
            dasha_info = json.loads(dasha_raw)
            maha = dasha_info.get("current_mahadasha") or {}
            dasha_lord = maha.get("lord", "")
    except Exception as e:
        logger.warning(f"[RecSys P1] kundli_dasha parse error: {e}")

    # ── Signal 1: Mahadasha Lord ─────────────────────────────────────────────
    if dasha_lord and dasha_lord in DASHA_LORD_QUESTIONS:
        for q in DASHA_LORD_QUESTIONS[dasha_lord][:2]:
            candidates.append(_fill(q, ascendant=ascendant_sign, dasha_lord=dasha_lord))
        logger.info(f"[RecSys P1] Dasha signal fired: {dasha_lord}")

    # ── Signal 2: Retrograde Planets ─────────────────────────────────────────
    retro_planets = [
        p for p in planets
        if str(p.get("isRetro", "")).lower() == "true"
        and p.get("name") in RETROGRADE_QUESTIONS
    ]
    for retro_planet in retro_planets[:1]:
        p_name = retro_planet.get("name", "")
        p_sign = retro_planet.get("sign_name", "")
        p_house = _compute_house(p_sign, ascendant_sign) if ascendant_sign else None
        for q in RETROGRADE_QUESTIONS[p_name][:1]:
            candidates.append(_fill(q, planet=p_name, sign=p_sign, house=p_house or "?"))
        logger.info(f"[RecSys P1] Retrograde signal fired: {p_name}")

    # ── Signal 3: Sensitive House Placements ─────────────────────────────────
    SENSITIVE_HOUSES = [7, 10, 8, 5, 12, 1]
    for planet_data in planets:
        if len(candidates) >= 3:
            break
        p_name = planet_data.get("name", "")
        p_sign = planet_data.get("sign_name", "")
        if not p_sign or not ascendant_sign or p_name == dasha_lord:
            continue
        house = _compute_house(p_sign, ascendant_sign)
        if house in SENSITIVE_HOUSES and house in HOUSE_PLACEMENT_QUESTIONS:
            q = HOUSE_PLACEMENT_QUESTIONS[house][0]
            candidates.append(_fill(q, planet=p_name, house=house))
            logger.info(f"[RecSys P1] House signal fired: {p_name} in house {house}")
            break

    # ── Signal 4: Topic Context ───────────────────────────────────────────────
    if topic and topic in TOPIC_FOLLOWUP_QUESTIONS and len(candidates) < 3:
        candidates.append(TOPIC_FOLLOWUP_QUESTIONS[topic][0])
        logger.info(f"[RecSys P1] Topic signal fired: {topic}")

    # ── Fallback ──────────────────────────────────────────────────────────────
    if not candidates:
        logger.info("[RecSys P1] No signals matched — using fallback")
        candidates = DEFAULT_QUESTIONS.copy()

    # ── Deduplicate and return top 3 ─────────────────────────────────────────
    seen: set = set()
    result: list[str] = []
    for q in candidates:
        if q not in seen:
            seen.add(q)
            result.append(q)

    logger.info(f"[RecSys P1] Returning {len(result[:3])} suggestions")
    return result[:3]

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


# =============================================================================
# PHASE 2 — CONTENT-BASED FILTERING
# =============================================================================
"""
WHAT IS CONTENT-BASED FILTERING? (RecSys Learning Note)
---------------------------------------------------------
Content-Based Filtering recommends items by comparing FEATURES of items to
FEATURES of the user. It does NOT need other users' data — it only needs to
understand the current user's profile.

Real-world analogy:
  - Spotify tags every song: genre=["jazz"], tempo="slow", mood="melancholic"
  - Your listening profile: genre_pref=["jazz","blues"], mood_pref=["melancholic"]
  - Score = how much the song's features overlap with your preferences
  - Recommend the highest scoring songs

Our version:
  - Each question is tagged: dasha_lords=["Saturn"], houses=[10], topics=["career"]
  - User profile: dasha_lord="Saturn", planets_in_house={10: "Saturn"}, topic="career"
  - Score = weighted feature intersection (the discrete version of cosine similarity)
  - Recommend top-scoring questions from the 70-question bank

WHY WEIGHTED FEATURE INTERSECTION INSTEAD OF PURE COSINE SIMILARITY?
  Pure cosine similarity needs all features to be continuous numbers (floats).
  Our features are categorical (names of planets, house numbers, topic strings).
  Weighted feature intersection is the CORRECT mathematical equivalent for
  categorical feature vectors. It gives the same ranking behavior as cosine
  similarity but works on symbolic/categorical data.

SCORING FORMULA:
  score(question, user) = Σ weight_i × match_i
  where match_i = 1 if feature i matches, 0 otherwise
  and weight_i is the importance of that feature type
"""

# Feature weights — how much each type of match contributes to the score
WEIGHT_DASHA_LORD   = 3.0   # Dasha lord dominates 6-20 years of life → highest weight
WEIGHT_RETROGRADE   = 2.5   # Retrograde planets have very unusual, specific effects
WEIGHT_HOUSE        = 2.0   # Specific house match is highly relevant
WEIGHT_PLANET       = 1.5   # General planet mention
WEIGHT_TOPIC        = 1.0   # Topic (life area) match
WEIGHT_ITEM_BASE    = 1.0   # The question's own base weight from the bank


def _build_user_profile(session: dict, topic: Optional[str]) -> dict:
    """
    Builds the user feature profile — the 'user vector' in Content-Based Filtering.

    This is the equivalent of Spotify building your taste profile from your
    listening history. Here we build an 'astrological taste profile' from the
    user's birth chart data.

    Returns a dict with:
      dasha_lord      : str  — current Mahadasha lord
      retrograde_set  : set  — names of all retrograde planets
      house_planet_map: dict — {house_number: planet_name} for sensitive houses
      planet_set      : set  — all planet names in chart
      topic           : str  — current conversation topic
    """
    profile = {
        "dasha_lord": "",
        "retrograde_set": set(),
        "house_planet_map": {},
        "planet_set": set(),
        "topic": topic or "",
    }

    try:
        raw = session.get("kundli_raw")
        if raw:
            parsed = json.loads(raw)
            planets: list[dict] = parsed.get("planets", []) or []
            ascendant_sign: str = parsed.get("ascendant_sign", "")

            for p in planets:
                p_name = p.get("name", "")
                p_sign = p.get("sign_name", "")
                if not p_name:
                    continue
                profile["planet_set"].add(p_name)
                if str(p.get("isRetro", "")).lower() == "true":
                    profile["retrograde_set"].add(p_name)
                if p_sign and ascendant_sign:
                    house = _compute_house(p_sign, ascendant_sign)
                    if house:
                        profile["house_planet_map"][house] = p_name
    except Exception as e:
        logger.warning(f"[RecSys P2] kundli_raw parse error: {e}")

    try:
        dasha_raw = session.get("kundli_dasha")
        if dasha_raw:
            dasha_info = json.loads(dasha_raw)
            maha = dasha_info.get("current_mahadasha") or {}
            profile["dasha_lord"] = maha.get("lord", "")
    except Exception as e:
        logger.warning(f"[RecSys P2] kundli_dasha parse error: {e}")

    return profile


def _score_question(question: dict, profile: dict) -> float:
    """
    Scores a single question against the user profile using Weighted Feature
    Intersection — the categorical equivalent of cosine similarity.

    Think of it as: 'how many of this question's features match this user's
    astrological fingerprint, weighted by importance?'

    The score is NOT normalized (unlike cosine similarity which divides by
    vector magnitudes). We don't need normalization because all questions in
    the bank have comparable feature cardinality.
    """
    score = 0.0

    # Base quality weight of the question itself (from question bank)
    score += question.get("weight", 1.0) * WEIGHT_ITEM_BASE

    # ── Feature Match 1: Dasha Lord (highest weight) ──────────────────────
    # "Is this question about the planet that dominates this user's current life phase?"
    if profile["dasha_lord"] and profile["dasha_lord"] in question["dasha_lords"]:
        score += WEIGHT_DASHA_LORD

    # ── Feature Match 2: Retrograde Planet ───────────────────────────────
    # "Is this a retrograde-specific question AND does the user have that planet retrograde?"
    if question.get("is_retrograde"):
        for retro_planet in question["planets"]:
            if retro_planet in profile["retrograde_set"]:
                score += WEIGHT_RETROGRADE
                break  # count only once per question

    # ── Feature Match 3: House Match ─────────────────────────────────────
    # "Does this question focus on a house where the user has a notable planet?"
    user_houses = set(profile["house_planet_map"].keys())
    question_houses = set(question["houses"])
    house_overlap = user_houses & question_houses
    if house_overlap:
        score += WEIGHT_HOUSE * len(house_overlap)

    # ── Feature Match 4: Planet Match ────────────────────────────────────
    # "Does this question mention a planet that is active/notable in the user's chart?"
    # Active = in Mahadasha, retrograde, or in a sensitive house
    active_planets = profile["retrograde_set"] | {profile["dasha_lord"]}
    active_planets.update(profile["house_planet_map"].values())
    planet_overlap = active_planets & set(question["planets"])
    if planet_overlap:
        score += WEIGHT_PLANET * len(planet_overlap)

    # ── Feature Match 5: Topic Match ─────────────────────────────────────
    # "Does this question's life area match what the user is currently asking about?"
    if profile["topic"] and profile["topic"] in question["topics"]:
        score += WEIGHT_TOPIC

    return score


def get_content_based_suggestions(
    session: dict,
    topic: Optional[str],
    language: str = "English",
    exclude: Optional[list[str]] = None
) -> list[str]:
    """
    Phase 2 RecSys — Content-Based Filtering.

    Steps:
    1. Build the user feature profile (the 'user vector') from chart data
    2. Score every question in the Question Bank against the user profile
    3. Sort by score descending, exclude questions already shown by Phase 1
    4. Return the top 3

    This gives questions the RULE ENGINE (Phase 1) might miss — especially
    questions about the user's chart that aren't covered by the priority rules.
    For example, a Jupiter in 9th house question would only appear via Phase 2
    since Phase 1 only handles the top-priority Mahadasha signal.

    RecSys Theory:
    This is a "User-to-Item" recommendation: we represent the user as a feature
    vector and score items (questions) by their feature overlap with the user.
    The absence of other users' data is exactly why this is called "Content-Based"
    (vs Collaborative Filtering which uses other users' behavior).
    """
    from app.data.question_bank import QUESTION_BANK

    excluded_set = set(exclude or [])

    # Step 1: Build user profile
    profile = _build_user_profile(session, topic)
    logger.info(
        f"[RecSys P2] User profile: dasha={profile['dasha_lord']}, "
        f"retro={profile['retrograde_set']}, "
        f"houses={list(profile['house_planet_map'].keys())[:5]}, "
        f"topic={profile['topic']}"
    )

    # Step 2: Score all questions
    scored: list[tuple[float, str]] = []
    for q in QUESTION_BANK:
        if q["text"] in excluded_set:
            continue
        score = _score_question(q, profile)
        scored.append((score, q["text"]))

    # Step 3: Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        logger.info(
            f"[RecSys P2] Top scores: "
            + " | ".join(f"{s:.1f}" for s, _ in scored[:5])
        )

    # Step 4: Return top 3 question texts
    result = [text for _, text in scored[:3]]
    logger.info(f"[RecSys P2] Returning {len(result)} content-based suggestions")
    return result


# =============================================================================
# HYBRID SCORER — Combines Phase 1 + Phase 2
# =============================================================================
"""
HYBRID RECOMMENDER SYSTEM (RecSys Learning Note)
--------------------------------------------------
Real production systems (Netflix, Spotify, Amazon) combine multiple recommenders
into a Hybrid System, because each approach has different strengths and blind spots:

  Phase 1 (Rule-Based):    high precision, low diversity, works cold-start
  Phase 2 (Content-Based): high diversity, personalized, wider question coverage
  Phase 3 (Session-Based): adapts to the CURRENT CONVERSATION's trajectory

Our blending strategy (Weighted Score Fusion):
  - Build a candidate pool from Phase 1 + Phase 2 (up to 8 unique candidates)
  - Score every candidate with all 3 techniques
  - Final score = 0.3 × P1_score + 0.4 × P2_score + 0.3 × P3_score
  - Return Top 3 by final score

This is called a "Weighted Score Fusion Hybrid" — the same architecture used
in Netflix's production recommender described in their 2015 paper.
"""


# =============================================================================
# PHASE 3 — SESSION-BASED (RECENCY-WEIGHTED) SCORING
# =============================================================================
"""
WHAT IS SESSION-BASED RECOMMENDATION? (RecSys Learning Note)
-------------------------------------------------------------
A pure Content-Based system (Phase 2) ignores ORDER — it doesn't care whether
the user asked about marriage 10 messages ago or just now.

Session-Based Recommendation says: "The user's RECENT actions are more
predictive of their NEXT action than older ones."

Real-world analogy:
  - If you watched 3 thrillers on Netflix today (in this session), the next
    recommendation should be a thriller — even if you've historically watched
    more comedies. Recency signals CURRENT intent.

Our version:
  - If the user just asked about marriage (turn 1) → marriage (turn 2) → career (turn 3)
  - Recency weights: career=1.0, marriage=0.71, marriage=0.50
    → career_total = 1.0, marriage_total = 1.21
    → Recommend questions that match BOTH career and marriage (the ongoing mix)
  - This is called "Exponential Decay Weighting"

MATHEMATICAL FORMULA:
  recency_weight(position) = decay_factor ^ position
  where position 0 = most recent message, position 1 = one before that, etc.
  decay_factor = 0.7 (a classic choice — not too aggressive, not too flat)

  session_score(candidate) = Σ (recency_weight(i) × topic_match(candidate, topic_i))
                              for i in range(last_N_messages)
"""

# Decay factor: how fast older messages lose influence
# 0.7 means each step back in history = 30% less influence
RECENCY_DECAY = 0.7

# Number of past user messages to look at
SESSION_WINDOW = 8

# Keyword lists for detecting a topic from raw message text
# This is a simple but effective "bag of words" topic classifier
SESSION_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "career":      ["career", "job", "work", "business", "promotion", "salary",
                    "office", "company", "naukri", "10th house", "profession"],
    "marriage":    ["marriage", "marry", "wedding", "partner", "spouse",
                    "relationship", "7th house", "shaadi", "rishta", "love"],
    "finance":     ["money", "wealth", "finance", "income", "salary",
                    "investment", "savings", "11th house", "paisa", "earn"],
    "health":      ["health", "illness", "disease", "body", "fitness",
                    "sick", "hospital", "6th house", "sehat"],
    "education":   ["education", "study", "exam", "college", "degree",
                    "university", "school", "9th house", "padhai"],
    "children":    ["children", "child", "baby", "pregnancy", "5th house",
                    "santan", "beta", "beti", "son", "daughter"],
    "abroad":      ["abroad", "foreign", "overseas", "immigration", "visa",
                    "12th house", "USA", "UK", "Canada", "settle"],
    "spirituality": ["spiritual", "moksha", "karma", "meditation", "temple",
                     "god", "12th house", "dharma", "astrology"],
    "remedies":    ["remedy", "gemstone", "mantra", "puja", "upay",
                    "stone", "ring", "yantra", "donation", "pooja"],
}

# Build a reverse lookup: question_text → list of matching topics (from question bank)
# This is computed once at module load (not per-request) for performance
_QUESTION_TOPIC_LOOKUP: dict[str, list[str]] = {}


def _build_question_topic_lookup() -> dict[str, list[str]]:
    """
    Pre-computes a dict mapping question_text → [topics].
    Called once at startup. Cached in _QUESTION_TOPIC_LOOKUP.
    """
    from app.data.question_bank import QUESTION_BANK
    lookup = {}
    for q in QUESTION_BANK:
        lookup[q["text"]] = q.get("topics", [])
    return lookup


def _detect_topic_from_text(text: str) -> Optional[str]:
    """
    Simple keyword-based topic detector for session history messages.

    RecSys Theory: This is a "bag of words" classifier — it doesn't care
    about word order, only presence of keywords. It's fast and works well
    for short astrological queries. For production you'd use the full
    LLM-powered topic classifier, but for session scoring, speed matters.
    """
    text_lower = text.lower()
    best_topic = None
    best_count = 0
    for topic, keywords in SESSION_TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_topic = topic
    return best_topic if best_count > 0 else None


def _extract_session_topic_weights(recent_messages: list[dict]) -> dict[str, float]:
    """
    Reads the last N user messages and produces a dict of {topic: cumulative_weight}.

    Applies exponential recency decay:
      weight(position) = RECENCY_DECAY ^ position
      position 0 = most recent user message

    Example:
      messages = [career_msg, marriage_msg, career_msg]  (oldest → newest)
      After reversing → [career_msg(pos=0), marriage_msg(pos=1), career_msg(pos=2)]
      career_weight  = 1.0^0 + 0.7^2 = 1.0 + 0.49 = 1.49
      marriage_weight = 0.7^1 = 0.70
      → career is the dominant current intent

    RecSys Theory: This is identical to how TikTok weights your "session graph"
    to decide what video to show next. The only difference is TikTok uses neural
    networks to embed videos; we use keyword bags for topics.
    """
    # Only look at user messages (ignore assistant responses)
    user_msgs = [
        m for m in recent_messages
        if m.get("role") == "user"
    ][-SESSION_WINDOW:]

    # Reverse so position 0 = most recent
    user_msgs_reversed = list(reversed(user_msgs))

    topic_weights: dict[str, float] = {}
    for position, msg in enumerate(user_msgs_reversed):
        text = msg.get("content", "")
        topic = _detect_topic_from_text(text)
        if topic:
            recency_weight = RECENCY_DECAY ** position
            topic_weights[topic] = topic_weights.get(topic, 0.0) + recency_weight
            logger.debug(
                f"[RecSys P3] msg[{position}] → topic='{topic}' weight={recency_weight:.2f}"
            )

    logger.info(f"[RecSys P3] Session topic weights: {topic_weights}")
    return topic_weights


def _compute_session_score(candidate_text: str, topic_weights: dict[str, float]) -> float:
    """
    Scores a candidate question by how well it matches the session's
    topic trajectory (computed by _extract_session_topic_weights).

    A candidate question that matches the user's most recent topic gets
    a high score; a question about something unrelated to this session
    gets a low score.

    This is the core function of Session-Based Recommendation.
    """
    if not topic_weights:
        return 0.0

    # Get the topics tagged to this candidate question
    global _QUESTION_TOPIC_LOOKUP
    if not _QUESTION_TOPIC_LOOKUP:
        _QUESTION_TOPIC_LOOKUP = _build_question_topic_lookup()

    candidate_topics = set(_QUESTION_TOPIC_LOOKUP.get(candidate_text, []))

    # Also check Phase 1 rule-based questions (not in the bank) via keyword detection
    if not candidate_topics:
        detected = _detect_topic_from_text(candidate_text)
        if detected:
            candidate_topics = {detected}

    score = 0.0
    for topic, weight in topic_weights.items():
        if topic in candidate_topics:
            score += weight

    return score


def get_suggestions(
    session: dict,
    topic: Optional[str],
    language: str = "English",
    recent_messages: Optional[list[dict]] = None
) -> list[str]:
    """
    Main entry point — Full 3-Phase Hybrid RecSys.

    Architecture: Weighted Score Fusion (same as Netflix production system)

    Phase 1 (Rule-Based)    → 0.30 weight
    Phase 2 (Content-Based) → 0.40 weight  ← highest weight (most information)
    Phase 3 (Session-Based) → 0.30 weight

    Final score = 0.30 × P1 + 0.40 × P2 + 0.30 × P3
    → Re-rank all candidates → Return top 3

    Parameters:
        session         : The active user session dict (has kundli_raw, kundli_dasha, etc.)
        topic           : The topic the user just asked about (e.g. "career")
        language        : "English" or "Hinglish"
        recent_messages : Last N messages from conversation history
                          (list of {role, content} dicts, oldest first)
    """
    from app.data.question_bank import QUESTION_BANK

    # ── Phase 1: Rule-Based candidates ─────────────────────────────────────
    p1_results = get_rule_based_suggestions(session, topic, language)

    # ── Phase 2: Score entire question bank ─────────────────────────────────
    profile = _build_user_profile(session, topic)
    p2_scored: dict[str, float] = {}
    for q in QUESTION_BANK:
        p2_scored[q["text"]] = _score_question(q, profile)

    # Also give Phase 1 questions a P2 score (look them up in bank, default 0)
    for q_text in p1_results:
        if q_text not in p2_scored:
            p2_scored[q_text] = 0.0

    # ── Phase 3: Session recency weights ────────────────────────────────────
    topic_weights: dict[str, float] = {}
    if recent_messages:
        topic_weights = _extract_session_topic_weights(recent_messages)
    elif topic:
        # Fallback: if no history, treat the current topic as position-0 with weight 1.0
        topic_weights = {topic: 1.0}

    # ── Build unified candidate pool ─────────────────────────────────────────
    # All unique questions from P1 + top-10 from P2 (by content score)
    top_p2_by_content = sorted(p2_scored.items(), key=lambda x: x[1], reverse=True)[:10]
    candidate_pool: set[str] = set(p1_results)
    for text, _ in top_p2_by_content:
        candidate_pool.add(text)

    # ── Score fusion: 0.30 × P1 + 0.40 × P2_norm + 0.30 × P3_norm ──────────
    # Normalize P2 and P3 scores to [0, 1] range before combining
    max_p2 = max(p2_scored.values()) if p2_scored else 1.0
    max_p3 = max(
        (_compute_session_score(t, topic_weights) for t in candidate_pool),
        default=1.0
    ) or 1.0

    p1_set = set(p1_results)
    final_scores: list[tuple[float, str]] = []

    for candidate in candidate_pool:
        p1_score = 1.0 if candidate in p1_set else 0.0
        p2_raw   = p2_scored.get(candidate, 0.0)
        p2_norm  = p2_raw / max_p2 if max_p2 > 0 else 0.0
        p3_raw   = _compute_session_score(candidate, topic_weights)
        p3_norm  = p3_raw / max_p3 if max_p3 > 0 else 0.0

        final = 0.30 * p1_score + 0.40 * p2_norm + 0.30 * p3_norm
        final_scores.append((final, candidate))
        logger.debug(
            f"[RecSys Fusion] '{candidate[:50]}...' "
            f"P1={p1_score:.2f} P2={p2_norm:.2f} P3={p3_norm:.2f} → {final:.3f}"
        )

    # ── Sort and return top 3 ───────────────────────────────────────────────
    final_scores.sort(key=lambda x: x[0], reverse=True)
    result = [text for _, text in final_scores[:3]]

    logger.info(
        f"[RecSys Hybrid P1+P2+P3] Top 3 scores: "
        + " | ".join(f"{s:.3f}" for s, _ in final_scores[:3])
    )
    return result


"""
Question Bank — Content-Based Filtering Item Catalogue
=======================================================

WHAT IS THIS? (RecSys Learning Note)
-------------------------------------
In Content-Based Filtering, every "item" (a question) is described by a set of
features — this is called the "item feature vector". The system compares this
to the "user feature vector" (their birth chart) and recommends the items
whose features overlap the most with the user's profile.

Think of it like Netflix tagging every movie with:
  genre=["thriller", "drama"], director="Nolan", language="English"

We tag every question with:
  dasha_lords=["Saturn"], planets=["Saturn"], houses=[10], topics=["career"]

TAGGING SCHEMA:
  - dasha_lords : list[str]  — Dasha lords this question is most relevant for
  - planets     : list[str]  — Planets directly referenced in the question
  - houses      : list[int]  — House numbers this question focuses on
  - topics      : list[str]  — Life areas covered (career, marriage, health…)
  - is_retrograde: bool      — True if question is specifically about a Vakri planet
  - weight      : float      — Base importance weight (1.0 = normal, 1.5 = high-signal)
"""

from typing import TypedDict


class Question(TypedDict):
    text: str
    dasha_lords: list
    planets: list
    houses: list
    topics: list
    is_retrograde: bool
    weight: float


QUESTION_BANK: list[Question] = [

    # ── SATURN ─────────────────────────────────────────────────────────────
    {
        "text": "How does Saturn Mahadasha create delays and discipline in my life?",
        "dasha_lords": ["Saturn"], "planets": ["Saturn"], "houses": [],
        "topics": ["career", "general"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "What karmic lessons is my retrograde Saturn teaching me?",
        "dasha_lords": ["Saturn"], "planets": ["Saturn"], "houses": [],
        "topics": ["general", "spirituality"], "is_retrograde": True, "weight": 1.5
    },
    {
        "text": "How does Saturn in my 10th house shape my career and authority?",
        "dasha_lords": ["Saturn"], "planets": ["Saturn"], "houses": [10],
        "topics": ["career"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "What remedies reduce the hardships of a difficult Saturn placement?",
        "dasha_lords": ["Saturn"], "planets": ["Saturn"], "houses": [],
        "topics": ["remedies"], "is_retrograde": False, "weight": 1.2
    },
    {
        "text": "How does Saturn in the 7th house delay or complicate marriage?",
        "dasha_lords": ["Saturn"], "planets": ["Saturn"], "houses": [7],
        "topics": ["marriage"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Does retrograde Saturn in my 5th house affect having children?",
        "dasha_lords": ["Saturn"], "planets": ["Saturn"], "houses": [5],
        "topics": ["children"], "is_retrograde": True, "weight": 1.4
    },
    {
        "text": "When does my Saturn Mahadasha end and what comes next?",
        "dasha_lords": ["Saturn"], "planets": ["Saturn"], "houses": [],
        "topics": ["timing", "general"], "is_retrograde": False, "weight": 1.3
    },

    # ── JUPITER ─────────────────────────────────────────────────────────────
    {
        "text": "How does Jupiter Mahadasha expand wealth and spiritual growth?",
        "dasha_lords": ["Jupiter"], "planets": ["Jupiter"], "houses": [],
        "topics": ["finance", "spirituality"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "Is Jupiter Mahadasha favorable for marriage and family in my chart?",
        "dasha_lords": ["Jupiter"], "planets": ["Jupiter"], "houses": [7],
        "topics": ["marriage", "children"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "How does retrograde Jupiter in my chart affect wisdom and timing of luck?",
        "dasha_lords": ["Jupiter"], "planets": ["Jupiter"], "houses": [],
        "topics": ["general"], "is_retrograde": True, "weight": 1.4
    },
    {
        "text": "What does Jupiter in my 9th house say about my luck and higher education?",
        "dasha_lords": ["Jupiter"], "planets": ["Jupiter"], "houses": [9],
        "topics": ["education", "abroad"], "is_retrograde": False, "weight": 1.3
    },
    {
        "text": "How does Jupiter in the 5th house bless children and intelligence?",
        "dasha_lords": ["Jupiter"], "planets": ["Jupiter"], "houses": [5],
        "topics": ["children", "education"], "is_retrograde": False, "weight": 1.3
    },

    # ── RAHU ─────────────────────────────────────────────────────────────────
    {
        "text": "How does Rahu Mahadasha create sudden gains and obsessive ambitions?",
        "dasha_lords": ["Rahu"], "planets": ["Rahu"], "houses": [],
        "topics": ["career", "finance"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "What illusions or confusion does Rahu Mahadasha create in relationships?",
        "dasha_lords": ["Rahu"], "planets": ["Rahu"], "houses": [7],
        "topics": ["marriage"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Does Rahu in the 10th house bring fame or instability in career?",
        "dasha_lords": ["Rahu"], "planets": ["Rahu"], "houses": [10],
        "topics": ["career"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "What foreign opportunities does Rahu in the 12th house bring?",
        "dasha_lords": ["Rahu"], "planets": ["Rahu"], "houses": [12],
        "topics": ["abroad"], "is_retrograde": False, "weight": 1.3
    },
    {
        "text": "How to manage the mental restlessness and anxiety of Rahu Mahadasha?",
        "dasha_lords": ["Rahu"], "planets": ["Rahu"], "houses": [],
        "topics": ["health", "remedies"], "is_retrograde": False, "weight": 1.3
    },

    # ── KETU ─────────────────────────────────────────────────────────────────
    {
        "text": "Why do I feel spiritually detached and directionless during Ketu Mahadasha?",
        "dasha_lords": ["Ketu"], "planets": ["Ketu"], "houses": [],
        "topics": ["spirituality", "general"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "What past-life karma is Ketu activating in my current life?",
        "dasha_lords": ["Ketu"], "planets": ["Ketu"], "houses": [],
        "topics": ["spirituality"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Does Ketu in the 7th house cause separation or unusual marriage?",
        "dasha_lords": ["Ketu"], "planets": ["Ketu"], "houses": [7],
        "topics": ["marriage"], "is_retrograde": False, "weight": 1.4
    },

    # ── VENUS ────────────────────────────────────────────────────────────────
    {
        "text": "How does Venus Mahadasha attract love, luxury, and creative success?",
        "dasha_lords": ["Venus"], "planets": ["Venus"], "houses": [],
        "topics": ["marriage", "finance"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "Is Venus Mahadasha the ideal time to get married in my chart?",
        "dasha_lords": ["Venus"], "planets": ["Venus"], "houses": [7],
        "topics": ["marriage", "timing"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "How does retrograde Venus make me revisit old relationships?",
        "dasha_lords": ["Venus"], "planets": ["Venus"], "houses": [],
        "topics": ["marriage"], "is_retrograde": True, "weight": 1.4
    },
    {
        "text": "What artistic talents does Venus in my 5th house bestow?",
        "dasha_lords": ["Venus"], "planets": ["Venus"], "houses": [5],
        "topics": ["career", "education"], "is_retrograde": False, "weight": 1.2
    },

    # ── MARS ─────────────────────────────────────────────────────────────────
    {
        "text": "How does Mars Mahadasha boost my energy, courage, and competitive drive?",
        "dasha_lords": ["Mars"], "planets": ["Mars"], "houses": [],
        "topics": ["career", "health"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "Is Mangal Dosha in my chart really affecting my marriage prospects?",
        "dasha_lords": ["Mars"], "planets": ["Mars"], "houses": [1, 4, 7, 8, 12],
        "topics": ["marriage"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "How does Mars in my 1st house make me aggressive or action-oriented?",
        "dasha_lords": ["Mars"], "planets": ["Mars"], "houses": [1],
        "topics": ["general"], "is_retrograde": False, "weight": 1.3
    },
    {
        "text": "What health issues should I watch during Mars Mahadasha?",
        "dasha_lords": ["Mars"], "planets": ["Mars"], "houses": [],
        "topics": ["health"], "is_retrograde": False, "weight": 1.3
    },

    # ── MERCURY ──────────────────────────────────────────────────────────────
    {
        "text": "How does Mercury Mahadasha enhance intellect, business, and communication?",
        "dasha_lords": ["Mercury"], "planets": ["Mercury"], "houses": [],
        "topics": ["career", "education"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "Is Mercury Mahadasha a good time to start a new business or study?",
        "dasha_lords": ["Mercury"], "planets": ["Mercury"], "houses": [],
        "topics": ["career", "education", "timing"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "How does retrograde Mercury in my chart create communication delays?",
        "dasha_lords": ["Mercury"], "planets": ["Mercury"], "houses": [],
        "topics": ["career", "general"], "is_retrograde": True, "weight": 1.4
    },

    # ── MOON ─────────────────────────────────────────────────────────────────
    {
        "text": "How does Moon Mahadasha affect my emotional stability and mental health?",
        "dasha_lords": ["Moon"], "planets": ["Moon"], "houses": [],
        "topics": ["health", "general"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "What does a strong or weak Moon in my chart say about my mother?",
        "dasha_lords": ["Moon"], "planets": ["Moon"], "houses": [4],
        "topics": ["general"], "is_retrograde": False, "weight": 1.3
    },
    {
        "text": "How does Moon Mahadasha affect travel and change of residence?",
        "dasha_lords": ["Moon"], "planets": ["Moon"], "houses": [12],
        "topics": ["abroad", "timing"], "is_retrograde": False, "weight": 1.2
    },

    # ── SUN ──────────────────────────────────────────────────────────────────
    {
        "text": "How does Sun Mahadasha strengthen my confidence and leadership?",
        "dasha_lords": ["Sun"], "planets": ["Sun"], "houses": [],
        "topics": ["career", "general"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "What does a strong or debilitated Sun mean for my father and authority figures?",
        "dasha_lords": ["Sun"], "planets": ["Sun"], "houses": [10],
        "topics": ["career", "general"], "is_retrograde": False, "weight": 1.3
    },

    # ── CAREER & 10TH HOUSE ──────────────────────────────────────────────────
    {
        "text": "What does my 10th house lord placement say about my career path?",
        "dasha_lords": [], "planets": [], "houses": [10],
        "topics": ["career"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Is my chart better suited for a job, business, or government career?",
        "dasha_lords": [], "planets": [], "houses": [10, 6, 7],
        "topics": ["career"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Which upcoming Dasha period is best for a promotion or career change?",
        "dasha_lords": [], "planets": [], "houses": [10],
        "topics": ["career", "timing"], "is_retrograde": False, "weight": 1.3
    },
    {
        "text": "Does my chart support a career in medicine, law, technology, or arts?",
        "dasha_lords": [], "planets": [], "houses": [10, 6],
        "topics": ["career", "education"], "is_retrograde": False, "weight": 1.2
    },

    # ── MARRIAGE & 7TH HOUSE ────────────────────────────────────────────────
    {
        "text": "What does my 7th house lord tell me about my future partner?",
        "dasha_lords": [], "planets": [], "houses": [7],
        "topics": ["marriage"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "Does my chart indicate a love marriage or an arranged marriage?",
        "dasha_lords": [], "planets": ["Venus", "Mars", "Moon"], "houses": [5, 7],
        "topics": ["marriage"], "is_retrograde": False, "weight": 1.5
    },
    {
        "text": "What is the timing for my marriage based on my current Dasha?",
        "dasha_lords": [], "planets": [], "houses": [7],
        "topics": ["marriage", "timing"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Is there a yoga in my chart that indicates delayed marriage?",
        "dasha_lords": ["Saturn", "Ketu"], "planets": ["Saturn"], "houses": [7],
        "topics": ["marriage"], "is_retrograde": False, "weight": 1.4
    },

    # ── FINANCE & WEALTH ────────────────────────────────────────────────────
    {
        "text": "What does my 2nd and 11th house say about my potential for wealth?",
        "dasha_lords": [], "planets": [], "houses": [2, 11],
        "topics": ["finance"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Which Dasha period brings sudden financial gains in my chart?",
        "dasha_lords": ["Rahu", "Jupiter", "Venus"], "planets": [], "houses": [2, 11],
        "topics": ["finance", "timing"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Does my chart show a Dhana Yoga or strong wealth accumulation?",
        "dasha_lords": [], "planets": ["Jupiter", "Venus"], "houses": [2, 5, 9, 11],
        "topics": ["finance"], "is_retrograde": False, "weight": 1.3
    },

    # ── HEALTH & 6TH HOUSE ──────────────────────────────────────────────────
    {
        "text": "Which planets in my chart are currently weakening my health?",
        "dasha_lords": [], "planets": ["Sun", "Moon", "Mars"], "houses": [6, 8, 12],
        "topics": ["health"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "What does my 6th house indicate about chronic health vulnerabilities?",
        "dasha_lords": [], "planets": [], "houses": [6],
        "topics": ["health"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "What Ayurvedic or Vedic remedies help strengthen my constitution?",
        "dasha_lords": [], "planets": [], "houses": [6, 1],
        "topics": ["health", "remedies"], "is_retrograde": False, "weight": 1.2
    },

    # ── EDUCATION & 4TH/9TH HOUSE ───────────────────────────────────────────
    {
        "text": "Is my chart favorable for higher education or studying abroad?",
        "dasha_lords": ["Jupiter", "Mercury", "Rahu"], "planets": [], "houses": [4, 9, 12],
        "topics": ["education", "abroad"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "What does my 9th house say about luck and higher learning in life?",
        "dasha_lords": [], "planets": [], "houses": [9],
        "topics": ["education", "spirituality"], "is_retrograde": False, "weight": 1.3
    },

    # ── ABROAD & 12TH HOUSE ─────────────────────────────────────────────────
    {
        "text": "Does my chart indicate foreign settlement or just travel abroad?",
        "dasha_lords": ["Rahu", "Moon"], "planets": ["Rahu"], "houses": [12, 9],
        "topics": ["abroad"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Which Dasha is best for a job or study opportunity abroad?",
        "dasha_lords": ["Rahu", "Jupiter"], "planets": [], "houses": [12],
        "topics": ["abroad", "timing"], "is_retrograde": False, "weight": 1.3
    },

    # ── SPIRITUALITY & 12TH HOUSE ───────────────────────────────────────────
    {
        "text": "What is my spiritual path according to my 9th and 12th house?",
        "dasha_lords": ["Ketu", "Jupiter"], "planets": ["Ketu", "Jupiter"], "houses": [9, 12],
        "topics": ["spirituality"], "is_retrograde": False, "weight": 1.3
    },
    {
        "text": "Which gemstone is most beneficial for my Ascendant and weak planets?",
        "dasha_lords": [], "planets": [], "houses": [1],
        "topics": ["remedies"], "is_retrograde": False, "weight": 1.3
    },
    {
        "text": "What mantras or rituals can help activate my strongest Yoga?",
        "dasha_lords": [], "planets": [], "houses": [],
        "topics": ["remedies", "spirituality"], "is_retrograde": False, "weight": 1.2
    },

    # ── CHILDREN & 5TH HOUSE ────────────────────────────────────────────────
    {
        "text": "How strong is my 5th house for having children and when is the timing?",
        "dasha_lords": ["Jupiter"], "planets": ["Jupiter"], "houses": [5],
        "topics": ["children", "timing"], "is_retrograde": False, "weight": 1.4
    },
    {
        "text": "Does retrograde Jupiter or Saturn in my chart delay childbirth?",
        "dasha_lords": ["Jupiter", "Saturn"], "planets": ["Jupiter", "Saturn"], "houses": [5],
        "topics": ["children"], "is_retrograde": True, "weight": 1.4
    },

    # ── PERSONALITY & ASCENDANT ──────────────────────────────────────────────
    {
        "text": "What does my Ascendant sign reveal about my core personality?",
        "dasha_lords": [], "planets": [], "houses": [1],
        "topics": ["general"], "is_retrograde": False, "weight": 1.2
    },
    {
        "text": "What is my strongest planet (Atmakaraka) and what does it mean?",
        "dasha_lords": [], "planets": [], "houses": [],
        "topics": ["general"], "is_retrograde": False, "weight": 1.2
    },
    {
        "text": "How does the 8th house in my chart shape my interest in the occult and hidden matters?",
        "dasha_lords": [], "planets": [], "houses": [8],
        "topics": ["spirituality", "general"], "is_retrograde": False, "weight": 1.1
    },
]

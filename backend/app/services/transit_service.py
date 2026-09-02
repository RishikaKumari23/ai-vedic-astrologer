import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from app.utils.logger import logger
from app.services.kundli_service import kundli_service, ZODIAC_SIGNS_ORDER

# Daily in-memory & database transit cache
_TRANSIT_CACHE: Dict[str, Any] = {}

ZODIAC = ZODIAC_SIGNS_ORDER

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya", "Ashlesha",
    "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

BENEFIC_HOUSES_FROM_MOON = {
    "Sun": [3, 6, 10, 11],
    "Moon": [1, 3, 6, 7, 10, 11],
    "Mars": [3, 6, 11],
    "Mercury": [2, 4, 6, 8, 10, 11],
    "Jupiter": [2, 5, 7, 9, 11],
    "Venus": [1, 2, 3, 4, 5, 8, 9, 11, 12],
    "Saturn": [3, 6, 11],
    "Rahu": [3, 6, 10, 11],
    "Ketu": [3, 6, 11],
}

HOUSE_NAMES = {
    1: "1st House (Self, Health & Vitality)",
    2: "2nd House (Wealth, Family & Speech)",
    3: "3rd House (Courage, Siblings & Skills)",
    4: "4th House (Home, Mother & Peace of Mind)",
    5: "5th House (Intellect, Romance & Creativity)",
    6: "6th House (Daily Work, Health & Overcoming Obstacles)",
    7: "7th House (Partnerships, Marriage & Business)",
    8: "8th House (Transformation, Longevity & Deep Insights)",
    9: "9th House (Higher Wisdom, Fortune & Dharma)",
    10: "10th House (Career, Public Image & Leadership)",
    11: "11th House (Gains, Friendships & Aspirations)",
    12: "12th House (Expenses, Spirituality & Solitude)",
}


class TransitService:
    def get_current_transits(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Fetches and caches real-time planetary positions for today (or target_date)."""
        if not target_date:
            target_date = date.today().strftime("%d-%m-%Y")
        
        cache_key = f"transits_{target_date}"
        if cache_key in _TRANSIT_CACHE:
            return _TRANSIT_CACHE[cache_key]

        try:
            # Fetch real-time planetary positions for standard coordinates (New Delhi reference)
            kundli_raw = kundli_service.fetch_kundli(
                name="Transit",
                date=target_date,
                time="12:00",
                latitude=28.6139,
                longitude=77.2090,
                max_retries=0
            )

            if kundli_raw and "planetary_positions" in kundli_raw:
                planets = []
                positions = kundli_raw.get("planetary_positions", [])
                planet_lords = kundli_raw.get("planet_lords", {})

                for p in positions:
                    name = p.get("name")
                    if not name or name == "Ascendant":
                        continue
                    sign = p.get("sign_name", "")
                    is_retro = str(p.get("isRetro", "")).lower() == "true"
                    lord_info = planet_lords.get(name, {})
                    degree = lord_info.get("degree")

                    nak_name = ""
                    if degree is not None:
                        try:
                            nak_idx = int(float(degree) / (360.0 / 27.0)) % 27
                            nak_name = NAKSHATRAS[nak_idx]
                        except Exception:
                            pass

                    planets.append({
                        "name": name,
                        "sign": sign,
                        "is_retrograde": is_retro,
                        "degree": degree,
                        "nakshatra": nak_name
                    })

                result = {
                    "date": target_date,
                    "planets": planets,
                    "raw": kundli_raw
                }
                _TRANSIT_CACHE[cache_key] = result
                return result
        except Exception as e:
            logger.error(f"Error fetching real-time transits: {e}")

        # Fallback realistic positions for 2026 if network is unavailable
        fallback_planets = [
            {"name": "Sun", "sign": "Leo", "is_retrograde": False, "degree": 135.2, "nakshatra": "Purva Phalguni"},
            {"name": "Moon", "sign": "Taurus", "is_retrograde": False, "degree": 42.5, "nakshatra": "Rohini"},
            {"name": "Mars", "sign": "Gemini", "is_retrograde": False, "degree": 68.0, "nakshatra": "Ardra"},
            {"name": "Mercury", "sign": "Leo", "is_retrograde": False, "degree": 138.1, "nakshatra": "Purva Phalguni"},
            {"name": "Jupiter", "sign": "Taurus", "is_retrograde": False, "degree": 48.6, "nakshatra": "Rohini"},
            {"name": "Venus", "sign": "Virgo", "is_retrograde": False, "degree": 160.4, "nakshatra": "Hasta"},
            {"name": "Saturn", "sign": "Pisces", "is_retrograde": True, "degree": 352.1, "nakshatra": "Uttara Bhadrapada"},
            {"name": "Rahu", "sign": "Pisces", "is_retrograde": True, "degree": 344.0, "nakshatra": "Purva Bhadrapada"},
            {"name": "Ketu", "sign": "Virgo", "is_retrograde": True, "degree": 164.0, "nakshatra": "Hasta"},
        ]
        return {"date": target_date, "planets": fallback_planets, "fallback": True}

    def calculate_gochar_overlay(self, session: Dict, current_transits: Optional[Dict] = None) -> Dict[str, Any]:
        """Calculates transit house placements, Sade Sati status, and Gochar impact for a profile."""
        if not current_transits:
            current_transits = self.get_current_transits()

        kundli_raw_str = session.get("kundli_raw")
        if not kundli_raw_str:
            return {"available": False, "reason": "No natal chart available"}

        try:
            natal_data = json.loads(kundli_raw_str)
            natal_planets = natal_data.get("planets", [])
            natal_ascendant = natal_data.get("ascendant_sign", "")

            natal_moon_sign = ""
            for p in natal_planets:
                if p.get("name") == "Moon":
                    natal_moon_sign = p.get("sign_name") or p.get("sign") or ""
                    break

            if not natal_ascendant:
                return {"available": False, "reason": "Missing Ascendant sign"}

            asc_idx = ZODIAC.index(natal_ascendant) if natal_ascendant in ZODIAC else 0
            moon_idx = ZODIAC.index(natal_moon_sign) if natal_moon_sign in ZODIAC else asc_idx

            transit_planets = current_transits.get("planets", [])
            transit_details = []

            saturn_sign = ""
            jupiter_sign = ""
            rahu_sign = ""

            for tp in transit_planets:
                p_name = tp["name"]
                t_sign = tp["sign"]
                if t_sign not in ZODIAC:
                    continue

                t_idx = ZODIAC.index(t_sign)

                # House from Natal Ascendant (Lagna Gochar)
                lagna_house = (t_idx - asc_idx + 12) % 12 + 1
                
                # House from Natal Moon (Chandra Gochar)
                moon_house = (t_idx - moon_idx + 12) % 12 + 1

                is_benefic_from_moon = moon_house in BENEFIC_HOUSES_FROM_MOON.get(p_name, [])

                if p_name == "Saturn":
                    saturn_sign = t_sign
                elif p_name == "Jupiter":
                    jupiter_sign = t_sign
                elif p_name == "Rahu":
                    rahu_sign = t_sign

                transit_details.append({
                    "name": p_name,
                    "current_sign": t_sign,
                    "degree": tp.get("degree"),
                    "nakshatra": tp.get("nakshatra"),
                    "is_retrograde": tp.get("is_retrograde", False),
                    "lagna_house": lagna_house,
                    "lagna_house_desc": HOUSE_NAMES.get(lagna_house, f"{lagna_house}th House"),
                    "moon_house": moon_house,
                    "is_favorable": is_benefic_from_moon,
                })

            # Evaluate Sade Sati Status
            sade_sati_status = "Inactive"
            sade_sati_desc = "Not currently under Sade Sati."
            if saturn_sign and natal_moon_sign:
                sat_idx = ZODIAC.index(saturn_sign) if saturn_sign in ZODIAC else -1
                moon_diff = (sat_idx - moon_idx + 12) % 12 + 1
                if moon_diff == 12:
                    sade_sati_status = "Phase 1: Rising (Aarohi)"
                    sade_sati_desc = "Saturn in 12th from Natal Moon. Focus on mental calm, mindful expenses, and inner development."
                elif moon_diff == 1:
                    sade_sati_status = "Phase 2: Peak (Madhya)"
                    sade_sati_desc = "Saturn transiting over Natal Moon. High responsibility, personal maturation, and discipline required."
                elif moon_diff == 2:
                    sade_sati_status = "Phase 3: Setting (Avarohi)"
                    sade_sati_desc = "Saturn in 2nd from Natal Moon. Financial restructuring, family grounding, and transition into stability."
                elif moon_diff == 4:
                    sade_sati_status = "Dhaiya (4th House Shani / Kantaka)"
                    sade_sati_desc = "Saturn in 4th from Moon. Focus on domestic harmony and career focus."
                elif moon_diff == 8:
                    sade_sati_status = "Ashtama Shani (8th House Shani)"
                    sade_sati_desc = "Saturn in 8th from Moon. Transformative period encouraging patience and health awareness."

            # Key Highlights
            highlights = []
            for td in transit_details:
                if td["name"] == "Jupiter":
                    if td["is_favorable"]:
                        highlights.append(f"✨ Jupiter in {td['current_sign']} ({td['lagna_house_desc']}): Favorable expansion and positive opportunities.")
                    else:
                        highlights.append(f"🌱 Jupiter in {td['current_sign']} ({td['lagna_house_desc']}): Steady learning and preparation phase.")
                elif td["name"] == "Saturn":
                    highlights.append(f"🪐 Saturn in {td['current_sign']} ({td['lagna_house_desc']}): Demands structure, perseverance, and long-term focus.")
                elif td["name"] == "Rahu":
                    highlights.append(f"⚡ Rahu in {td['current_sign']} ({td['lagna_house_desc']}): Sparks ambition and unconventional growth in this sphere.")

            return {
                "available": True,
                "profile_name": session.get("name", "User"),
                "relation": session.get("relation", "Self"),
                "natal_ascendant": natal_ascendant,
                "natal_moon_sign": natal_moon_sign,
                "transit_date": current_transits.get("date"),
                "sade_sati": {
                    "status": sade_sati_status,
                    "description": sade_sati_desc,
                    "is_active": sade_sati_status != "Inactive"
                },
                "transits": transit_details,
                "highlights": highlights
            }
        except Exception as e:
            logger.error(f"Failed to calculate Gochar overlay: {e}")
            return {"available": False, "error": str(e)}

    def format_gochar_for_prompt(self, gochar_data: Dict) -> str:
        """Formats transit overlay into a concise, grounded prompt block for LLM inference."""
        if not gochar_data or not gochar_data.get("available"):
            return "No real-time transit data available."

        lines = [
            f"=== Real-Time Planetary Transits (Gochar for {gochar_data.get('profile_name')}) ===",
            f"- Natal Lagna: {gochar_data.get('natal_ascendant')} | Natal Moon: {gochar_data.get('natal_moon_sign')}",
            f"- Sade Sati Status: {gochar_data.get('sade_sati', {}).get('status')} ({gochar_data.get('sade_sati', {}).get('description')})",
            "- Active Key Transits:"
        ]

        for t in gochar_data.get("transits", []):
            if t["name"] in ["Jupiter", "Saturn", "Rahu", "Ketu", "Mars", "Sun"]:
                fav = "Favorable" if t.get("is_favorable") else "Neutral/Challenging"
                lines.append(f"  * {t['name']} in {t['current_sign']} (Transiting {t['lagna_house_desc']} from Lagna, {t['moon_house']}th from Moon — {fav})")

        return "\n".join(lines)


transit_service = TransitService()

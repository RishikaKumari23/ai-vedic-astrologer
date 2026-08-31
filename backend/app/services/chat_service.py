from typing import Dict, Any, List, Optional
from datetime import datetime
from difflib import SequenceMatcher
import json
import threading
from app.memory.database import db
from app.services.llm_service import llm_service
from app.services.geocoding_service import geocoding_service
from app.services.kundli_service import kundli_service
from app.rag.vector_store import vector_store
from app.rag.embeddings import EmbeddingsProvider
from app.prompts.templates import ASTROLOGER_PROMPT, MISSING_INFO_PROMPT
from app.config.settings import settings
from app.utils.logger import logger
from app.services.intent_service import classify_intent, get_response_contract
from app.services.claim_validator import validate_claims, build_claim_correction_instructions
from app.services.topic_service import (
    classify_topic, build_topic_emphasis, get_search_bias,
    build_explanation_footer, TOPIC_CHART_FACTORS, get_instant_suggestions,
    rank_favorable_periods, format_dasha_timeline_for_prompt,
    build_evidence_vote, format_evidence_vote_for_prompt
)
from app.services.hybrid_router import route_topic
from app.services.dasha_api_service import dasha_api_service
from app.services.yoga_service import detect_yogas, format_yogas_for_prompt


class ChatService:
    def __init__(self):
        self.embeddings_provider = EmbeddingsProvider()

    def _format_history_for_llm(self, history: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in history:
            if msg["role"] == "system":
                continue
            role_name = "User" if msg["role"] == "user" else "Astrologer"
            formatted.append(f"{role_name}: {msg['content']}")
        return "\n".join(formatted)


    def _get_context_window(self, session_id: str, session: dict) -> str:
        """Returns the formatted context: [Rolling Summary] + [Last 4 messages]."""
        summary = session.get("conversation_summary")
        # Only fetch last 4 messages (2 Q&A pairs) to save context
        recent_history = db.get_history(session_id, limit=4)
        recent_text = self._format_history_for_llm(recent_history)
        
        if summary:
            return f"[SUMMARY OF EARLIER CONVERSATION]: {summary}\n\n[RECENT MESSAGES]:\n{recent_text}"
        return recent_text

    def _update_rolling_summary(self, session_id: str, session: dict) -> None:
        """Runs in background. Every 8 unsummarized messages, summarizes the oldest 6."""
        try:
            history = db.get_history(session_id, limit=100)
            total_msgs = len(history)
            last_count = session.get("last_summarized_msg_count") or 0
            
            # We want to keep the last 2 messages (1 Q&A pair) unsummarized
            unsummarized_count = total_msgs - last_count
            if unsummarized_count >= 8:  # 4 Q&A pairs
                # Grab at most 8 messages (4 turns) at a time to prevent massive KV cache OOM on M1 Macs
                chunk_end = min(last_count + 8, total_msgs - 2)
                messages_to_summarize = history[last_count : chunk_end]
                
                # We also need to update last_count to chunk_end instead of total_msgs - 2
                new_last_count = chunk_end
                
                if not messages_to_summarize:
                    return
                
                text_to_summarize = self._format_history_for_llm(messages_to_summarize)
                existing_summary = session.get("conversation_summary") or ""
                
                prompt = (
                    "You are a summarization assistant. Summarize the following astrological conversation "
                    "in 2-3 concise sentences. Focus on the main topics discussed, predictions made, and user details.\n"
                )
                if existing_summary:
                    prompt += f"\nPrevious summary:\n{existing_summary}\n"
                prompt += f"\nNew messages to append to summary:\n{text_to_summarize}\n"
                prompt += "\nReturn ONLY the new integrated summary, nothing else."
                
                new_summary = llm_service.generate(prompt=prompt, temperature=0.3)
                
                # Update DB
                db.update_session(session_id, {
                    "conversation_summary": new_summary.strip(),
                    "last_summarized_msg_count": new_last_count
                })
                session["conversation_summary"] = new_summary.strip()
                session["last_summarized_msg_count"] = new_last_count
                logger.info("Successfully updated rolling conversation summary.")
        except Exception as e:
            logger.error(f"Failed to update rolling summary: {e}")

    def _to_24h(self, time_str: str) -> str:
        if not time_str:
            return ""
        try:
            from dateutil import parser as dateutil_parser
            parsed = dateutil_parser.parse(time_str.strip(), fuzzy=True)
            return parsed.strftime("%H:%M")
        except Exception:
            try:
                parsed_time = datetime.strptime(time_str.strip(), "%I:%M %p")
                return parsed_time.strftime("%H:%M")
            except ValueError:
                return time_str


    def _get_temperature(self, intent: str, repeat_hint: str = "") -> float:
        """Return the right creativity level based on what the user is asking.
        
        - timing / simple_fact  → low (0.2): needs precision, specific dates/planets
        - explanation / strength_check → medium (0.5): factual but needs explanation
        - remedy → medium-low (0.45): grounded but warm tone
        - general → high (0.7): creative, warm, varied
        - repeat_hint → always bumped to 0.8 to force fresh phrasing
        """
        if repeat_hint:
            return 0.8
        temperature_map = {
            "timing":         0.2,
            "simple_fact":    0.2,
            "explanation":    0.5,
            "strength_check": 0.5,
            "remedy":         0.45,
            "general":        0.7,
        }
        return temperature_map.get(intent, 0.7)


    def _run_chain_of_thought(self, message_text: str, kundli_data: str,
                               dasha_info: str, topic: str, language: str) -> str:
        """Step 1 of 2: Ask the LLM to silently reason about the chart BEFORE
        generating the final response. Returns a structured 3-point analysis
        that gets injected into the main prompt as 'grounded facts'.
        
        This forces the 8B model to think first, then speak — dramatically
        reducing hallucinations and improving relevance.
        """
        cot_prompt = (
            f"You are a Vedic astrology analysis engine. Given the data below, "
            f"identify the 3 most relevant astrological facts that directly answer "
            f"the user's question. Be specific: name actual planets, houses, and "
            f"periods. Do NOT write a response to the user — only output 3 bullet points.\n\n"
            f"User's Question: {message_text}\n"
            f"Topic: {topic or 'general'}\n"
            f"Birth Chart Summary: {kundli_data[:800]}\n"
            f"Current Dasha: {dasha_info[:300]}\n\n"
            f"Output format (3 lines only, no extra text):\n"
            f"1. [Most relevant chart fact]\n"
            f"2. [Second relevant fact or timing detail]\n"
            f"3. [Supporting factor or cautionary note]"
        )
        try:
            result = llm_service.generate(prompt=cot_prompt, temperature=0.2)
            logger.info(f"[CoT] Reasoning facts generated: {result[:120]}...")
            return result.strip()
        except Exception as e:
            logger.warning(f"[CoT] Chain-of-thought failed, skipping: {e}")
            return ""

    def _fetch_and_cache_kundli(self, session_id: str, session: Dict) -> str:
        """Fetches Kundli + real Dasha data once, and ALSO pre-computes and
        caches yoga_text right here — since yogas depend only on planets/
        ascendant, not on topic, they never need recomputation for this
        birth chart. This is the single expensive network round-trip;
        everything downstream should read from cache, not refetch."""
        try:
            coords = geocoding_service.geocode(session.get("birth_place"))
            if not coords:
                logger.warning(f"Could not geocode birth_place: {session.get('birth_place')}")
                return "No chart data available."

            lat, lon = coords
            time_24h = self._to_24h(session.get("birth_time", ""))

            kundli_data = kundli_service.fetch_kundli(
                name=session.get("name") or "User",
                date=session.get("dob"), time=time_24h, latitude=lat, longitude=lon,
            )
            if kundli_data:
                dasha_info = kundli_service.get_real_or_calculated_dasha(
                    kundli_data, session.get("dob"), time_24h, lat, lon
                )
                kundli_str = kundli_service.summarize_kundli(kundli_data, dob=session.get("dob"), dasha_info=dasha_info)
                chart_data = kundli_service.extract_chart_data(kundli_data)
                chart_json = json.dumps(chart_data) if chart_data else None
                dasha_json = json.dumps(dasha_info) if dasha_info else None
                full_raw_json = json.dumps(kundli_data, ensure_ascii=False)

                yoga_text = ""
                if chart_data:
                    try:
                        yogas = detect_yogas(chart_data.get("planets", []), chart_data.get("ascendant_sign", ""))
                        yoga_text = format_yogas_for_prompt(yogas)
                    except Exception as yoga_err:
                        logger.error(f"Yoga pre-computation failed: {yoga_err}")

                updates = {
                    "kundli_data": kundli_str,
                    "kundli_raw": chart_json,
                    "kundli_dasha": dasha_json,
                    "kundli_full_raw": full_raw_json,
                    "yoga_text": yoga_text,
                    "latitude": lat,        # saved so _get_dasha_timeline always has coords
                    "longitude": lon,       # without needing to re-geocode
                    "topic_cache": None,        # invalidate any stale per-topic cache
                    "dasha_tree_raw": None,     # invalidate — refetched lazily on next timeline need
                }
                db.update_session(session_id, updates)
                session.update(updates)
                logger.info("Kundli data fetched and cached (summary + chart + dasha + full raw + yoga)")
                return kundli_str
        except Exception as kundli_err:
            logger.error(f"Kundli fetch failed: {kundli_err}")

        return "No chart data available."
    
    def _get_rag_context(self, message_text: str, topic: Optional[str] = None, llm_summary: Optional[str] = None):
        """Retrieves top-K chunks, logs their relevance scores (so retrieval
        quality is actually visible/debuggable), and DROPS chunks below
        settings.MIN_RAG_RELEVANCE instead of silently feeding weak matches
        to the LLM as if they were solid ground truth."""
        try:
            # Use the router's LLM-generated intent summary as the search query
            # when available — it's always clean English and semantically precise,
            # which dramatically improves book retrieval for Hinglish / vague queries.
            # Fall back to raw message if no summary exists.
            base_query = llm_summary if llm_summary else message_text
            search_query = base_query
            if topic:
                bias = get_search_bias(topic)
                if bias:
                    search_query = f"{base_query} {bias}"

            query_vector = self.embeddings_provider.get_embedding(search_query)
            hits = vector_store.hybrid_search(
                query=search_query, query_vector=query_vector,
                top_k=settings.TOP_K_RETRIEVAL, alpha=settings.HYBRID_ALPHA
            )

            logger.info(f"[RAG] query='{search_query}' topic={topic} raw_hits={len(hits)}")
            for i, hit in enumerate(hits):
                logger.info(
                    f"[RAG]   #{i+1} score={hit['score']:.3f} "
                    f"(semantic={hit['semantic_score']:.3f}, lexical={hit['lexical_score']:.3f}) "
                    f"source={hit['metadata'].get('source', 'Unknown')}"
                )

            relevant_hits = [h for h in hits if h["score"] >= settings.MIN_RAG_RELEVANCE]
            dropped = len(hits) - len(relevant_hits)
            if dropped > 0:
                logger.info(
                    f"[RAG] dropped {dropped}/{len(hits)} chunk(s) below "
                    f"relevance threshold {settings.MIN_RAG_RELEVANCE}"
                )

            if not relevant_hits:
                logger.info("[RAG] no sufficiently relevant chunks — proceeding with no book context")
                return "No reference available.", []

            context_chunks = []
            sources = []
            for i, hit in enumerate(relevant_hits):
                source = hit["metadata"].get("source", "Unknown")
                sources.append(source)
                context_chunks.append(
                    f"--- Context {i+1} [Source: {source}, relevance: {hit['score']:.2f}] ---\n{hit['text']}\n"
                )

            return "\n".join(context_chunks), sources
        except Exception as rag_err:
            logger.error(f"RAG failed: {rag_err}")
            return "No reference available.", []
    
    # ------------------------------------------------------------------
    # Per-topic cache — bundles emphasis/divisional/consistency/missing-
    # evidence/timeline/evidence_vote into ONE JSON blob keyed by topic,
    # so asking the SAME topic again in a session reuses everything instantly.
    # ------------------------------------------------------------------
    def _get_topic_cache(self, session: Dict, topic: str) -> Optional[Dict]:
        raw = session.get("topic_cache")
        if not raw:
            return None
        try:
            cache = json.loads(raw)
            return cache.get(topic)
        except Exception:
            return None

    def _save_topic_cache(self, session_id: str, session: Dict, topic: str, bundle: Dict):
        try:
            raw = session.get("topic_cache")
            cache = json.loads(raw) if raw else {}
            cache[topic] = bundle
            cache_json = json.dumps(cache, ensure_ascii=False)
            db.update_session(session_id, {"topic_cache": cache_json})
            session["topic_cache"] = cache_json
        except Exception as e:
            logger.error(f"Failed to save topic cache for '{topic}': {e}")

    def _get_topic_bundle(self, session_id: str, session: Dict, topic: Optional[str], language: str) -> Dict[str, Any]:
        """Returns {emphasis, divisional, consistency, missing_evidence,
        timeline, evidence_vote} for this topic — from cache if already
        computed this session, otherwise computes once and caches."""
        empty = {"emphasis": "", "divisional": "", "consistency": "", "missing_evidence": "", "timeline": "", "evidence_vote": None}
        if not topic:
            return empty

        cached = self._get_topic_cache(session, topic)
        if cached is not None:
            logger.info(f"Using cached topic bundle for '{topic}'")
            if "evidence_vote" not in cached:
                cached["evidence_vote"] = None
            return cached

        bundle = dict(empty)
        try:
            cached_raw = session.get("kundli_raw")
            cached_dasha = session.get("kundli_dasha")
            if cached_raw:
                parsed = json.loads(cached_raw)
                planets = parsed.get("planets", [])
                ascendant_sign = parsed.get("ascendant_sign")
                dasha_info = json.loads(cached_dasha) if cached_dasha else None

                if planets and ascendant_sign:
                    bundle["emphasis"] = build_topic_emphasis(topic, planets, ascendant_sign, None)

                from app.services.topic_service import build_consistency_check, build_consistency_note, build_missing_evidence_note
                check = build_consistency_check(topic, planets, ascendant_sign, dasha_info)
                bundle["consistency"] = build_consistency_note(check, topic)

                yoga_text_for_vote = session.get("yoga_text") or ""
                vote = build_evidence_vote(topic, planets, ascendant_sign, dasha_info, yoga_text=yoga_text_for_vote)
                bundle["evidence_vote"] = vote
                vote_text = format_evidence_vote_for_prompt(vote, topic)
                if vote_text:
                    bundle["consistency"] = (
                        f"{bundle['consistency']}\n\n{vote_text}" if bundle["consistency"] else vote_text
                    )

                config = TOPIC_CHART_FACTORS.get(topic, {})
                chart_code = config.get("divisional_chart")
                if chart_code:
                    kundli_data = self._get_full_kundli_response(session_id, session)
                    if kundli_data:
                        purpose_map = {"D9": "marriage", "D10": "career", "D24": "education", "D7": "children"}
                        bundle["divisional"] = kundli_service.summarize_divisional_chart(
                            kundli_data, chart_code, purpose_map.get(chart_code, chart_code)
                        )

                bundle["missing_evidence"] = build_missing_evidence_note(
                    topic, planets, ascendant_sign, dasha_info, bundle["divisional"]
                )

            bundle["timeline"] = self._get_dasha_timeline(session_id, session, topic, language)
        except Exception as e:
            logger.error(f"Topic bundle build failed for '{topic}': {e}")

        self._save_topic_cache(session_id, session, topic, bundle)
        return bundle

    def _get_dasha_timeline(self, session_id: str, session: Dict, topic: Optional[str], language: str) -> str:
        try:
            cached_tree_raw = session.get("dasha_tree_raw")
            dasha_tree = None
            if cached_tree_raw:
                try:
                    dasha_tree = json.loads(cached_tree_raw)
                except Exception:
                    dasha_tree = None

            if dasha_tree is None:
                time_24h = self._to_24h(session.get("birth_time", ""))
                coords_lat = session.get("latitude")
                coords_lon = session.get("longitude")
                if not (coords_lat and coords_lon):
                    return ""

                kundli_raw_full = self._get_full_kundli_response(session_id, session)
                ascendant_data = kundli_service.get_ascendant_data(kundli_raw_full) if kundli_raw_full else None
                if not ascendant_data:
                    logger.warning("Could not extract ascendant_data — skipping dasha timeline")
                    return ""

                dasha_tree = dasha_api_service.fetch_dasha_tree(
                    date=session.get("dob"), time=time_24h,
                    latitude=coords_lat, longitude=coords_lon,
                    ascendant_data=ascendant_data,
                )
                if not dasha_tree:
                    return ""

                tree_json = json.dumps(dasha_tree, ensure_ascii=False)
                db.update_session(session_id, {"dasha_tree_raw": tree_json})
                session["dasha_tree_raw"] = tree_json

            upcoming = dasha_api_service.get_upcoming_periods(dasha_tree, months_ahead=60)
            favorable = rank_favorable_periods(upcoming, topic)
            timeline_str = format_dasha_timeline_for_prompt(upcoming, favorable, language)
            logger.info(f"Dasha timeline built for topic '{topic}': {len(upcoming)} periods, {len(favorable)} favorable")
            return timeline_str
        except Exception as dasha_err:
            logger.error(f"Dasha timeline fetch failed: {dasha_err}")
            return ""

    def _get_yoga_text(self, session: Dict) -> str:
        return session.get("yoga_text") or ""

    def _build_final_kundli_data(self, kundli_str: str, topic_emphasis: str, divisional_text: str,
                                   yoga_text: str, missing_evidence: str = "") -> str:
        parts = [p for p in [kundli_str, topic_emphasis, divisional_text, yoga_text, missing_evidence] if p]
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Response Novelty Checker
    # ------------------------------------------------------------------
    def _get_recent_assistant_texts(self, session_id: str, limit: int = 5) -> List[str]:
        history = db.get_history(session_id, limit=20)
        assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
        return assistant_msgs[-limit:]

    def _similarity_ratio(self, text_a: str, text_b: str) -> float:
        a = text_a.strip().lower()
        b = text_b.strip().lower()
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _is_too_similar(self, response_text: str, recent_texts: List[str], threshold: float = 0.75) -> Optional[str]:
        for prior in recent_texts:
            if self._similarity_ratio(response_text, prior) >= threshold:
                return prior
        return None

    def _get_repeat_topic_hint(self, session: Dict, topic: Optional[str]) -> str:
        if not topic:
            return ""
        try:
            raw = session.get("topic_memory")
            if not raw:
                return ""
            memory = json.loads(raw)
            prior_summary = memory.get(topic)
            if not prior_summary:
                return ""
            return (
                f"IMPORTANT — Avoid repetition: You already answered a {topic} question earlier "
                f"in this conversation, with reasoning along these lines: \"{prior_summary}\". "
                f"This new question is related but distinct — answer what's SPECIFICALLY being "
                f"asked now. Do not restate the same facts/wording again; build on or add to what "
                f"was already said, or focus on a different angle (timing, specific action, etc.)."
            )
        except Exception as e:
            logger.error(f"Repeat-topic hint build failed: {e}")
            return ""

    def _get_user_memory_block(self, session: Dict, current_topic: Optional[str]) -> str:
        try:
            raw = session.get("topic_memory")
            if not raw:
                return ""
            memory = json.loads(raw)
            if not memory:
                return ""
            lines = []
            for topic, summary in memory.items():
                if topic == current_topic:
                    continue
                lines.append(f"- {topic.capitalize()}: {summary}")
            if not lines:
                return ""
            return "Earlier in this conversation, you already discussed:\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to build user memory block: {e}")
            return ""

    def _update_topic_memory(self, session_id: str, session: Dict, topic: Optional[str], response_text: str):
        if not topic or not response_text:
            return
        try:
            raw = session.get("topic_memory")
            memory = json.loads(raw) if raw else {}
            truncated = response_text.strip().replace("\n", " ")
            if len(truncated) > 150:
                truncated = truncated[:150].rsplit(" ", 1)[0] + "..."
            memory[topic] = truncated
            memory_json = json.dumps(memory, ensure_ascii=False)
            db.update_session(session_id, {"topic_memory": memory_json})
            session["topic_memory"] = memory_json
            logger.info(f"Updated topic_memory['{topic}']")
        except Exception as e:
            logger.error(f"Failed to update topic memory: {e}")

    def _safe_generate_followups(self, response_text: str, language: str) -> List[str]:
        try:
            return llm_service.generate_followups(response_text, language) or []
        except Exception as followup_err:
            logger.error(f"Follow-up suggestion generation failed: {followup_err}")
            return []

    # ------------------------------------------------------------------
    # NON-STREAMING — POST /api/chat
    # ------------------------------------------------------------------
    def process_chat_message(self, session_id: str, message_text: str) -> Dict[str, Any]:
        logger.info(f"Processing chat message for session: {session_id}")
        try:
            session = db.get_or_create_session(session_id)
            history_text = self._get_context_window(session_id, session)

            profile_complete = bool(session.get("dob") and session.get("birth_time") and session.get("birth_place"))

            if profile_complete:
                logger.info("Profile already complete — skipping extraction step")
                is_astrology = True
                language = session.get("language", "Hinglish")
                db.add_message(session_id, "user", message_text)
                missing_fields = []
            else:
                try:
                    extracted = llm_service.extract_profile_details(message_text, history_text)
                except Exception as extract_err:
                    logger.error(f"Profile extraction failed: {extract_err}")
                    extracted = {"dob": None, "birth_time": None, "birth_place": None, "language": "Hinglish", "is_astrology_query": True}

                updates = {}
                for key in ["dob", "birth_time", "birth_place"]:
                    if extracted.get(key):
                        updates[key] = extracted[key]
                        session[key] = extracted[key]
                if extracted.get("language"):
                    updates["language"] = extracted["language"]
                    session["language"] = extracted["language"]
                if updates:
                    db.update_session(session_id, updates)

                language = session.get("language", "Hinglish")
                db.add_message(session_id, "user", message_text)

                is_astrology = extracted.get("is_astrology_query", True)
                missing_fields = []
                if not session.get("dob"):
                    missing_fields.append(("Date of Birth", "dob"))
                if not session.get("birth_time"):
                    missing_fields.append(("Birth Time", "birth_time"))
                if not session.get("birth_place"):
                    missing_fields.append(("Birth Place", "birth_place"))

                if is_astrology and missing_fields:
                    next_missing_name, _ = missing_fields[0]
                    try:
                        prompt = MISSING_INFO_PROMPT.format(missing_detail=next_missing_name, language=language)
                        response_text = llm_service.generate(prompt=prompt, temperature=0.3)
                    except Exception as llm_err:
                        logger.error(f"LLM failed: {llm_err}")
                        response_text = f"Kripya apna {next_missing_name} batayein."
                    db.add_message(session_id, "assistant", response_text)
                    threading.Thread(target=self._update_rolling_summary, args=(session_id, session), daemon=True).start()
                    return {
                        "session_id": session_id, "message": response_text,
                        "dob": session.get("dob"), "birth_time": session.get("birth_time"),
                        "birth_place": session.get("birth_place"), "language": language
                    }

            topic_result = route_topic(message_text, self.embeddings_provider) if (is_astrology and not missing_fields) else None
            topic = topic_result.topic if topic_result else None
            intent = classify_intent(message_text) if (is_astrology and not missing_fields) else "general"
            response_contract = get_response_contract(intent)

            context_str = ""
            rag_sources = []
            if is_astrology and not missing_fields:
                context_str, rag_sources = self._get_rag_context(message_text, topic, llm_summary=topic_result.llm_summary if topic_result else None)



            kundli_str = "No chart data available."
            if is_astrology and not missing_fields:
                cached_kundli = session.get("kundli_data")
                kundli_str = cached_kundli if cached_kundli else self._fetch_and_cache_kundli(session_id, session)

            yoga_text = self._get_yoga_text(session) if (is_astrology and not missing_fields) else ""

            topic_emphasis = divisional_text = consistency_note = missing_evidence = dasha_timeline_str = ""
            evidence_vote = None
            if is_astrology and not missing_fields:
                dasha_timeline_str = self._get_dasha_timeline(session_id, session, topic or "general", language)
                if topic:
                    bundle = self._get_topic_bundle(session_id, session, topic, language)
                    topic_emphasis = bundle["emphasis"]
                    divisional_text = bundle["divisional"]
                    consistency_note = bundle["consistency"]
                    missing_evidence = bundle["missing_evidence"]
                    evidence_vote = bundle.get("evidence_vote")

            final_kundli_data = self._build_final_kundli_data(kundli_str, topic_emphasis, divisional_text, yoga_text, missing_evidence)
            user_memory = self._get_user_memory_block(session, topic) if (is_astrology and not missing_fields) else ""

            try:
                current_date = datetime.now().strftime("%d %B %Y")

                # Chain-of-Thought for non-streaming path
                cot_facts = ""
                if is_astrology and not missing_fields and kundli_str != "No chart data available.":
                    cot_facts = self._run_chain_of_thought(
                        message_text, kundli_str,
                        session.get("kundli_dasha") or "",
                        topic or "general", language
                    )

                cot_injection = ""
                if cot_facts:
                    cot_injection = (
                        f"\n\nPRE-ANALYSED CHART FACTS (use these as your grounded foundation — "
                        f"do NOT ignore them or contradict them in your response):\n{cot_facts}"
                    )

                astrologer_prompt = ASTROLOGER_PROMPT.format(
                    name=session.get("name") or "Friend",
                    language=language, dob=session.get("dob") or "Not provided",
                    birth_time=session.get("birth_time") or "Not provided",
                    birth_place=session.get("birth_place") or "Not provided",
                    current_date=current_date,
                    context=context_str or "No book context.", kundli_data=final_kundli_data,
                    user_memory=user_memory or "No prior topics discussed yet.",
                    consistency_note=consistency_note or "No specific conflict detected.",
                    dasha_timeline=dasha_timeline_str or "No timeline data available.",
                    response_contract=response_contract,
                    history=history_text, query=message_text
                )
                if cot_injection:
                    astrologer_prompt += cot_injection
                _temp = self._get_temperature(intent, "")
                response_text = llm_service.generate(prompt=astrologer_prompt, temperature=_temp)

                if is_astrology and not missing_fields:
                    recent_texts = self._get_recent_assistant_texts(session_id)
                    similar_to = self._is_too_similar(response_text, recent_texts)

                    claim_failures = validate_claims(response_text, dasha_timeline_str, evidence_vote)

                    if similar_to or claim_failures:
                        retry_prompt = astrologer_prompt
                        if similar_to:
                            retry_prompt += (
                                f"\n\nIMPORTANT: Your previous response was very similar to this one:\n"
                                f"\"{similar_to}\"\n"
                                f"Express the same astrological reasoning but do NOT repeat the same wording. "
                                f"Focus specifically on what's different about the CURRENT question."
                            )
                        if claim_failures:
                            logger.info(f"Claim validation found {len(claim_failures)} issue(s) — regenerating with corrections")
                            retry_prompt += "\n\n" + build_claim_correction_instructions(claim_failures)

                        response_text = llm_service.generate(prompt=retry_prompt, temperature=self._get_temperature(intent, 'retry'))

                        # Re-validate once after the fix attempt, log-only —
                        # don't loop indefinitely if the model still misses it.
                        remaining = validate_claims(response_text, dasha_timeline_str, evidence_vote)
                        if remaining:
                            logger.warning(f"Claim validation still found {len(remaining)} issue(s) after regeneration")
            except Exception as gen_err:
                logger.error(f"Generation failed: {gen_err}")
                response_text = "Mujhe samajhne mein kuch pareshani ho gayi."

            db.add_message(session_id, "assistant", response_text)
            threading.Thread(target=self._update_rolling_summary, args=(session_id, session), daemon=True).start()

            if is_astrology and not missing_fields:
                try:
                    trace = self._build_reasoning_trace(session, topic, rag_sources, topic_result=topic_result)
                    db.update_session(session_id, {"last_reasoning_trace": json.dumps(trace)})
                except Exception as trace_err:
                    logger.error(f"Reasoning trace caching failed: {trace_err}")
                self._update_topic_memory(session_id, session, topic, response_text)

            suggestions = []
            if response_text and len(response_text) > 20:
                suggestions = self._safe_generate_followups(response_text, language)

            return {
                "session_id": session_id, "message": response_text,
                "dob": session.get("dob"), "birth_time": session.get("birth_time"),
                "birth_place": session.get("birth_place"), "language": language,
                "suggestions": suggestions
            }

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return {"session_id": session_id, "message": "Kripya dobara koshish karein.",
                    "dob": None, "birth_time": None, "birth_place": None, "language": "Hinglish"}

    # ------------------------------------------------------------------
    # STREAMING — POST /api/chat/stream
    # ------------------------------------------------------------------
    def process_chat_message_stream(self, session_id: str, message_text: str):
        logger.info(f"Processing chat message (stream) for session: {session_id}")
        # Initialize language from session BEFORE try block so the except fallback
        # always uses the correct user language instead of hardcoded Hinglish
        _session_pre = db.get_or_create_session(session_id)
        language = _session_pre.get("language", "English")
        try:
            session = _session_pre
            history_text = self._get_context_window(session_id, session)

            profile_complete = bool(session.get("dob") and session.get("birth_time") and session.get("birth_place"))

            if profile_complete:
                logger.info("Profile already complete — skipping extraction step")
                is_astrology = True
                language = session.get("language", "Hinglish")
                db.add_message(session_id, "user", message_text)
                missing_fields = []
            else:
                try:
                    extracted = llm_service.extract_profile_details(message_text, history_text)
                except Exception as extract_err:
                    logger.error(f"Profile extraction failed: {extract_err}")
                    extracted = {"dob": None, "birth_time": None, "birth_place": None, "language": "Hinglish", "is_astrology_query": True}

                updates = {}
                for key in ["dob", "birth_time", "birth_place"]:
                    if extracted.get(key):
                        updates[key] = extracted[key]
                        session[key] = extracted[key]
                if extracted.get("language"):
                    updates["language"] = extracted["language"]
                    session["language"] = extracted["language"]
                if updates:
                    db.update_session(session_id, updates)

                language = session.get("language", "Hinglish")
                db.add_message(session_id, "user", message_text)

                is_astrology = extracted.get("is_astrology_query", True)
                missing_fields = []
                if not session.get("dob"):
                    missing_fields.append(("Date of Birth", "dob"))
                if not session.get("birth_time"):
                    missing_fields.append(("Birth Time", "birth_time"))
                if not session.get("birth_place"):
                    missing_fields.append(("Birth Place", "birth_place"))

                if is_astrology and missing_fields:
                    next_missing_name, _ = missing_fields[0]
                    prompt = MISSING_INFO_PROMPT.format(missing_detail=next_missing_name, language=language)
                    full_text = ""
                    try:
                        for token in llm_service.generate_stream(prompt=prompt, temperature=0.3):
                            full_text += token
                            yield {"type": "chunk", "text": token}
                    except Exception as llm_err:
                        logger.error(f"LLM stream failed: {llm_err}")
                        full_text = f"Kripya apna {next_missing_name} batayein."
                        yield {"type": "chunk", "text": full_text}

                    db.add_message(session_id, "assistant", full_text)
                    threading.Thread(target=self._update_rolling_summary, args=(session_id, session), daemon=True).start()
                    yield {"type": "done", "session_id": session_id, "message": full_text,
                           "dob": session.get("dob"), "birth_time": session.get("birth_time"),
                           "birth_place": session.get("birth_place"), "language": language}
                    return

            topic_result = route_topic(message_text, self.embeddings_provider) if (is_astrology and not missing_fields) else None
            topic = topic_result.topic if topic_result else None
            intent = classify_intent(message_text) if (is_astrology and not missing_fields) else "general"
            response_contract = get_response_contract(intent)

            context_str = ""
            rag_sources = []
            if is_astrology and not missing_fields:
                context_str, rag_sources = self._get_rag_context(message_text, topic, llm_summary=topic_result.llm_summary if topic_result else None)



            kundli_str = "No chart data available."
            if is_astrology and not missing_fields:
                cached_kundli = session.get("kundli_data")
                kundli_str = cached_kundli if cached_kundli else self._fetch_and_cache_kundli(session_id, session)

            yoga_text = self._get_yoga_text(session) if (is_astrology and not missing_fields) else ""

            topic_emphasis = divisional_text = consistency_note = missing_evidence = dasha_timeline_str = ""
            evidence_vote = None
            if is_astrology and not missing_fields:
                dasha_timeline_str = self._get_dasha_timeline(session_id, session, topic or "general", language)
                if topic:
                    bundle = self._get_topic_bundle(session_id, session, topic, language)
                    topic_emphasis = bundle["emphasis"]
                    divisional_text = bundle["divisional"]
                    consistency_note = bundle["consistency"]
                    missing_evidence = bundle["missing_evidence"]
                    evidence_vote = bundle.get("evidence_vote")

            final_kundli_data = self._build_final_kundli_data(kundli_str, topic_emphasis, divisional_text, yoga_text, missing_evidence)

            user_memory = ""
            repeat_hint = ""
            if is_astrology and not missing_fields:
                user_memory = self._get_user_memory_block(session, topic)
                repeat_hint = self._get_repeat_topic_hint(session, topic)

            current_date = datetime.now().strftime("%d %B %Y")

            # Chain-of-Thought: disabled in streaming path to avoid 30-40s blank screen.
            # CoT is only used in the non-streaming /api/chat endpoint where buffering is acceptable.
            cot_facts = ""  # Streaming always skips CoT for UX reasons

            # Inject CoT facts into the prompt if available
            cot_injection = ""
            if cot_facts:
                cot_injection = (
                    f"\n\nPRE-ANALYSED CHART FACTS (use these as your grounded foundation — "
                    f"do NOT ignore them or contradict them in your response):\n{cot_facts}"
                )

            astrologer_prompt = ASTROLOGER_PROMPT.format(
                name=session.get("name") or "Friend",
                language=language, dob=session.get("dob") or "Not provided",
                birth_time=session.get("birth_time") or "Not provided",
                birth_place=session.get("birth_place") or "Not provided",
                current_date=current_date,
                context=context_str or "No book context.", kundli_data=final_kundli_data,
                user_memory=user_memory or "No prior topics discussed yet.",
                consistency_note=consistency_note or "No specific conflict detected.",
                dasha_timeline=dasha_timeline_str or "No timeline data available.",
                response_contract=response_contract,
                history=history_text, query=message_text
            )
            if repeat_hint:
                astrologer_prompt += f"\n\n{repeat_hint}"
            if cot_injection:
                astrologer_prompt += cot_injection

            gen_temperature = self._get_temperature(intent, repeat_hint)

            full_text = ""
            try:
                for token in llm_service.generate_stream(prompt=astrologer_prompt, temperature=gen_temperature):
                    full_text += token
                    yield {"type": "chunk", "text": token}
            except Exception as gen_err:
                logger.error(f"Streaming generation failed: {gen_err}")
                if language == "English":
                    full_text = "I'm sorry, I had some trouble processing that."
                elif language == "Hindi":
                    full_text = "मुझे समझने में कुछ परेशानी हो गई।"
                else:
                    full_text = "Mujhe samajhne mein kuch pareshani ho gayi."
                yield {"type": "chunk", "text": full_text}

            db.add_message(session_id, "assistant", full_text)
            threading.Thread(target=self._update_rolling_summary, args=(session_id, session), daemon=True).start()

            # Claim validation is LOG-ONLY here — tokens are already streamed
            # to the user, so there's nothing left to regenerate cleanly.
            # This still gives you visibility into hallucination rate for
            # streamed responses without breaking the streaming UX.
            if is_astrology and not missing_fields:
                try:
                    claim_failures = validate_claims(full_text, dasha_timeline_str, evidence_vote)
                    if claim_failures:
                        logger.warning(f"Claim validation found {len(claim_failures)} issue(s) in streamed response (not corrected — log only): {claim_failures}")
                except Exception as validate_err:
                    logger.error(f"Claim validation failed: {validate_err}")

            if is_astrology and not missing_fields:
                try:
                    trace = self._build_reasoning_trace(session, topic, rag_sources, topic_result=topic_result)
                    db.update_session(session_id, {"last_reasoning_trace": json.dumps(trace)})
                except Exception as trace_err:
                    logger.error(f"Reasoning trace caching failed: {trace_err}")
                self._update_topic_memory(session_id, session, topic, full_text)

            suggestions = get_instant_suggestions(topic, language)

            yield {"type": "done", "session_id": session_id, "message": full_text,
                   "dob": session.get("dob"), "birth_time": session.get("birth_time"),
                   "birth_place": session.get("birth_place"), "language": language,
                   "suggestions": suggestions}

        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            if language == "English":
                fallback = "I'm sorry, please try again."
            elif language == "Hindi":
                fallback = "कृपया दोबारा कोशिश करें।"
            else:
                fallback = "Kripya dobara koshish karein."
            yield {"type": "chunk", "text": fallback}
            yield {"type": "done", "session_id": session_id, "message": fallback,
                   "dob": None, "birth_time": None, "birth_place": None, "language": language}

    def _build_reasoning_trace(self, session: Dict, topic: Optional[str], rag_hits_sources: Optional[List[str]] = None, topic_result=None) -> list:
        try:
            cached_raw = session.get("kundli_raw")
            cached_dasha = session.get("kundli_dasha")
            if not cached_raw:
                return []
            parsed = json.loads(cached_raw)
            planets = parsed.get("planets", [])
            ascendant_sign = parsed.get("ascendant_sign")
            dasha_info = json.loads(cached_dasha) if cached_dasha else None

            from app.services.topic_service import build_consistency_check, build_reasoning_trace
            consistency_check = build_consistency_check(topic, planets, ascendant_sign, dasha_info)

            topic_cache = self._get_topic_cache(session, topic)
            evidence_vote = topic_cache.get("evidence_vote") if topic_cache else None

            return build_reasoning_trace(
                topic, ascendant_sign, planets, dasha_info, consistency_check,
                rag_hits_sources, evidence_vote, topic_result=topic_result
            )
        except Exception as e:
            logger.error(f"Reasoning trace build failed: {e}")
            return []

    def _get_full_kundli_response(self, session_id: str, session: Dict) -> Optional[Dict]:
        cached_full_raw = session.get("kundli_full_raw")
        if cached_full_raw:
            try:
                return json.loads(cached_full_raw)
            except Exception as e:
                logger.error(f"Failed to parse cached kundli_full_raw: {e}")

        self._fetch_and_cache_kundli(session_id, session)
        cached_full_raw = session.get("kundli_full_raw")
        if cached_full_raw:
            try:
                return json.loads(cached_full_raw)
            except Exception:
                return None
        return None


chat_service = ChatService()
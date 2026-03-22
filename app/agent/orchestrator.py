import json
import logging
import time
from typing import Optional

import anthropic

from app.models.session import (
    AgentSession,
    AgentState,
    Intent,
    IntentResult,
    LatencyBreakdown,
    Turn,
)
from app.services.llm.claude_llm import ClaudeLLMService
from app.services.rag.retriever import RAGRetriever
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

NLU_SYSTEM_PROMPT = """You are an NLU component for a fintech voice support system.
Given a user utterance and optional conversation history, extract:
- intent: one of [loan_query, portfolio_query, policy_faq, account_query, otp_verify, escalate, smalltalk, unknown]
- sub_intent: specific sub-category string or null
- entities: dict of extracted slots (loan_id, account_id, portfolio_id, date_range, amount, otp_code)
- confidence: float 0.0-1.0
- clarification_needed: bool
- clarification_question: string or null

Return ONLY valid JSON. No explanation. No markdown."""

INTENTS_REQUIRING_AUTH = {Intent.LOAN_QUERY, Intent.PORTFOLIO_QUERY, Intent.ACCOUNT_QUERY}


class Orchestrator:
    def __init__(self) -> None:
        self._llm = ClaudeLLMService()
        self._rag = RAGRetriever()
        self._nlu_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def process_turn(
        self,
        session: AgentSession,
        transcript: str,
    ) -> tuple[str, LatencyBreakdown]:
        """
        Core agent loop: classify intent → route → generate response.
        Returns (response_text, latency_breakdown).
        """
        latency = LatencyBreakdown()
        user_turn = Turn(role="user", transcript=transcript)

        # 1. Intent classification
        t0 = time.monotonic()
        intent_result = await self._classify_intent(transcript, session)
        latency.intent_ms = int((time.monotonic() - t0) * 1000)
        user_turn.intent = intent_result
        session.add_turn(user_turn)

        logger.info(
            "intent session=%s intent=%s confidence=%.2f",
            session.session_id,
            intent_result.intent,
            intent_result.confidence,
        )

        # 2. Handle clarification
        if intent_result.clarification_needed and intent_result.clarification_question:
            return self._assistant_turn(session, intent_result.clarification_question, latency)

        # 3. Check if auth is required
        if intent_result.intent in INTENTS_REQUIRING_AUTH and not session.is_authenticated:
            session.state = AgentState.AUTH_REQUIRED
            msg = (
                "I need to verify your identity before I can share account details. "
                "Please say 'send OTP' and I'll send a one-time password to your registered number."
            )
            return self._assistant_turn(session, msg, latency)

        # 4. Route to handler
        response_text = await self._route(session, intent_result, latency)
        return self._assistant_turn(session, response_text, latency)

    async def _classify_intent(
        self,
        transcript: str,
        session: AgentSession,
    ) -> IntentResult:
        history_snippet = "\n".join(
            f"{t.role}: {t.transcript or t.response_text}"
            for t in session.recent_turns(3)
        )
        user_content = f"History:\n{history_snippet}\n\nUtterance: {transcript}" if history_snippet else f"Utterance: {transcript}"

        response = await self._nlu_client.messages.create(
            model=settings.llm_model,
            max_tokens=256,
            temperature=0.0,
            system=NLU_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        try:
            raw = json.loads(response.content[0].text)
            return IntentResult(
                intent=Intent(raw.get("intent", "unknown")),
                sub_intent=raw.get("sub_intent"),
                entities=raw.get("entities", {}),
                confidence=float(raw.get("confidence", 0.5)),
                clarification_needed=raw.get("clarification_needed", False),
                clarification_question=raw.get("clarification_question"),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("NLU parse failed: %s", exc)
            return IntentResult(
                intent=Intent.UNKNOWN,
                sub_intent=None,
                entities={},
                confidence=0.0,
            )

    async def _route(
        self,
        session: AgentSession,
        intent: IntentResult,
        latency: LatencyBreakdown,
    ) -> str:
        if intent.intent == Intent.SMALLTALK:
            return await self._handle_smalltalk(session, intent)

        if intent.intent == Intent.ESCALATE:
            session.state = AgentState.ESCALATE
            return "Of course. Let me connect you to one of our specialists right away. Please hold."

        if intent.intent == Intent.POLICY_FAQ:
            return await self._handle_rag(session, intent, latency)

        if intent.intent in INTENTS_REQUIRING_AUTH:
            return await self._handle_api_intent(session, intent, latency)

        if intent.intent == Intent.OTP_VERIFY:
            return await self._handle_otp(session, intent)

        # Fallback — try RAG
        return await self._handle_rag(session, intent, latency)

    async def _handle_rag(
        self,
        session: AgentSession,
        intent: IntentResult,
        latency: LatencyBreakdown,
    ) -> str:
        session.state = AgentState.RAG_QUERY
        query = session.conversation_history[-1].transcript

        t0 = time.monotonic()
        rag_result = await self._rag.retrieve(query)
        latency.rag_ms = int((time.monotonic() - t0) * 1000)

        context = self._rag.format_context(rag_result)
        messages = session.history_for_llm()

        t1 = time.monotonic()
        response = await self._llm.complete(messages, context=context)
        latency.llm_ms = int((time.monotonic() - t1) * 1000)

        return response

    async def _handle_api_intent(
        self,
        session: AgentSession,
        intent: IntentResult,
        latency: LatencyBreakdown,
    ) -> str:
        # Phase 1: stub — live API integration in Phase 2
        session.state = AgentState.API_CALL
        messages = session.history_for_llm()

        t0 = time.monotonic()
        response = await self._llm.complete(
            messages,
            context="[Live account data is not yet available in this demo environment.]",
        )
        latency.llm_ms = int((time.monotonic() - t0) * 1000)
        return response

    async def _handle_smalltalk(
        self,
        session: AgentSession,
        intent: IntentResult,
    ) -> str:
        messages = session.history_for_llm()
        return await self._llm.complete(messages)

    async def _handle_otp(
        self,
        session: AgentSession,
        intent: IntentResult,
    ) -> str:
        # Phase 2 will wire in real OTP service
        if intent.sub_intent == "request_otp" or not intent.entities.get("otp_code"):
            session.state = AgentState.OTP_SENT
            return "I've sent a 6-digit OTP to your registered mobile number. Please tell me the code when you receive it."

        session.state = AgentState.OTP_VERIFY
        # Stub: accept any 6-digit code in Phase 1
        otp_code = str(intent.entities.get("otp_code", ""))
        if len(otp_code) == 6 and otp_code.isdigit():
            session.is_authenticated = True
            session.state = AgentState.LISTENING
            return "Identity verified successfully. How can I help you with your account?"

        session.otp_attempts += 1
        if session.otp_attempts >= settings.otp_max_attempts:
            session.state = AgentState.FAILED
            return "I'm sorry, I was unable to verify your identity. Please call our support line for assistance."

        return f"That code doesn't seem right. You have {settings.otp_max_attempts - session.otp_attempts} attempt(s) remaining. Please try again."

    @staticmethod
    def _assistant_turn(
        session: AgentSession,
        response_text: str,
        latency: LatencyBreakdown,
    ) -> tuple[str, LatencyBreakdown]:
        assistant_turn = Turn(
            role="assistant",
            response_text=response_text,
            latency=latency,
        )
        session.add_turn(assistant_turn)
        session.state = AgentState.LISTENING
        return response_text, latency

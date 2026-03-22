from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class AgentState(str, Enum):
    GREETING = "greeting"
    LISTENING = "listening"
    PROCESSING = "processing"
    AUTH_REQUIRED = "auth_required"
    OTP_SENT = "otp_sent"
    OTP_VERIFY = "otp_verify"
    RAG_QUERY = "rag_query"
    API_CALL = "api_call"
    RESPONDING = "responding"
    ESCALATE = "escalate"
    FAILED = "failed"
    ENDED = "ended"


class Intent(str, Enum):
    LOAN_QUERY = "loan_query"
    PORTFOLIO_QUERY = "portfolio_query"
    POLICY_FAQ = "policy_faq"
    ACCOUNT_QUERY = "account_query"
    OTP_VERIFY = "otp_verify"
    ESCALATE = "escalate"
    SMALLTALK = "smalltalk"
    UNKNOWN = "unknown"


@dataclass
class WordTimestamp:
    word: str
    start_ms: int
    end_ms: int
    confidence: float


@dataclass
class TranscriptEvent:
    session_id: str
    text: str
    is_final: bool
    confidence: float
    timestamp_ms: int
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class IntentResult:
    intent: Intent
    sub_intent: Optional[str]
    entities: dict
    confidence: float
    clarification_needed: bool = False
    clarification_question: Optional[str] = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    source_url: str = ""


@dataclass
class RAGResult:
    chunks: list[RetrievedChunk]
    retrieval_latency_ms: int = 0
    rerank_latency_ms: int = 0


@dataclass
class APICallLog:
    service: str
    endpoint: str
    status_code: int
    latency_ms: int
    success: bool


@dataclass
class LatencyBreakdown:
    stt_ms: int = 0
    intent_ms: int = 0
    rag_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0

    @property
    def total_ms(self) -> int:
        return self.stt_ms + self.rag_ms + self.llm_ms + self.tts_ms


@dataclass
class Turn:
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # "user" | "assistant"
    transcript: str = ""
    intent: Optional[IntentResult] = None
    rag_chunks: list[RetrievedChunk] = field(default_factory=list)
    api_calls: list[APICallLog] = field(default_factory=list)
    response_text: str = ""
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phone_number: str = ""
    user_id: Optional[str] = None
    state: AgentState = AgentState.GREETING
    conversation_history: list[Turn] = field(default_factory=list)
    is_authenticated: bool = False
    otp_attempts: int = 0
    channel: str = "web"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    def add_turn(self, turn: Turn) -> None:
        self.conversation_history.append(turn)
        self.last_activity = datetime.utcnow()

    def recent_turns(self, n: int = 10) -> list[Turn]:
        return self.conversation_history[-n:]

    def history_for_llm(self) -> list[dict]:
        return [
            {"role": t.role, "content": t.transcript or t.response_text}
            for t in self.recent_turns()
            if (t.role == "user" and t.transcript) or (t.role == "assistant" and t.response_text)
        ]

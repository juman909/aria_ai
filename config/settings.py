from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "change-me"

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1

    # STT
    deepgram_api_key: str = ""
    stt_model: str = "nova-2"
    stt_language: str = "en-IN"

    # TTS
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    tts_model: str = "eleven_turbo_v2"

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # Vector DB
    pinecone_api_key: str = ""
    pinecone_index_name: str = "finance-voice-agent"
    pinecone_environment: str = "us-east-1-aws"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/finance_agent"
    database_pool_size: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # OTP
    otp_expiry_seconds: int = 300
    otp_max_attempts: int = 3

    # Observability
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    log_level: str = "INFO"

    # RAG
    rag_top_k_retrieval: int = 20
    rag_top_k_rerank: int = 5
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_hybrid_alpha: float = 0.7
    rag_score_threshold: float = 0.6

    # Agent
    conversation_window: int = 10
    session_timeout_seconds: int = 1800  # 30 min


@lru_cache
def get_settings() -> Settings:
    return Settings()

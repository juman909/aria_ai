# Aria AI — Voice Agent

A real-time voice AI support agent for fintech customer support. Users speak to the agent via WebSocket; their speech is transcribed (Deepgram), understood and answered by Claude (Anthropic), and the response is spoken back (ElevenLabs). A RAG pipeline backed by Pinecone answers policy/FAQ questions with grounded context.

## Architecture

```
Browser / Client
      │  WebSocket (audio in / audio out)
      ▼
FastAPI  ──► Orchestrator
              ├── NLU  (Claude – intent classification)
              ├── RAG  (OpenAI embeddings + Pinecone + BM25 rerank)
              ├── LLM  (Claude – response generation)
              ├── STT  (Deepgram Nova-2)
              └── TTS  (ElevenLabs Turbo)
```

**Supported intents:** `loan_query`, `portfolio_query`, `policy_faq`, `account_query`, `otp_verify`, `escalate`, `smalltalk`

Authentication is enforced for account/loan/portfolio queries via OTP verification.

---

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- API keys for: Anthropic, Deepgram, ElevenLabs, OpenAI, Pinecone

---

## Local Setup

### 1. Clone & create a virtual environment

```bash
git clone git@github.com:juman909/aria_ai.git
cd aria_ai
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys and service URLs:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key |
| `DEEPGRAM_API_KEY` | Deepgram STT key |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS key |
| `OPENAI_API_KEY` | OpenAI embeddings key |
| `PINECONE_API_KEY` | Pinecone vector DB key |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `finance-voice-agent`) |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |

### 4. Start PostgreSQL and Redis

```bash
# Using Docker
docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
docker run -d --name redis -p 6379:6379 redis:7
```

### 5. Run the server

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

- Health check: `GET /health`
- Voice WebSocket: `WS /ws/{session_id}`

---

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Project Structure

```
├── app/
│   ├── agent/          # Orchestrator — intent routing and conversation loop
│   ├── api/            # FastAPI routes (health, WebSocket)
│   ├── core/           # Logging configuration
│   ├── models/         # Session and turn data models
│   ├── services/
│   │   ├── llm/        # Claude LLM service
│   │   ├── rag/        # Embeddings and Pinecone retriever
│   │   ├── stt/        # Deepgram speech-to-text
│   │   └── tts/        # ElevenLabs text-to-speech
│   └── utils/
├── config/             # Pydantic settings
├── tests/
├── docker/
├── main.py
└── requirements.txt
```

---

## Environment Variables Reference

See [`.env.example`](.env.example) for the full list with defaults.

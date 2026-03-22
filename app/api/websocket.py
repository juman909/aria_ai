import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.orchestrator import Orchestrator
from app.models.session import AgentSession, AgentState
from app.services.stt.deepgram_stt import DeepgramSTTService
from app.services.tts.elevenlabs_tts import ElevenLabsTTSService

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session store (Phase 1). Replace with Redis in Phase 3.
_sessions: dict[str, AgentSession] = {}


async def _audio_source(websocket: WebSocket):
    """Async generator that yields raw audio bytes from the WebSocket."""
    while True:
        try:
            data = await websocket.receive_bytes()
            yield data
        except WebSocketDisconnect:
            return


@router.websocket("/ws/voice/{session_id}")
async def voice_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("ws_connect session=%s", session_id)

    session = _sessions.setdefault(session_id, AgentSession(session_id=session_id))
    stt = DeepgramSTTService()
    tts = ElevenLabsTTSService()
    orchestrator = Orchestrator()

    # Send greeting
    greeting = "Hello! I'm Aria, your FinEdge Bank voice assistant. How can I help you today?"
    await _send_voice_response(websocket, tts, greeting)
    session.state = AgentState.LISTENING

    try:
        async for transcript_event in stt.stream_transcribe(_audio_source(websocket), session_id):
            if not transcript_event.is_final:
                continue  # Skip partials — use final transcripts only

            transcript = transcript_event.text.strip()
            if not transcript:
                continue

            logger.info("transcript session=%s text=%r", session_id, transcript)

            # Process through agent
            t0 = time.monotonic()
            response_text, latency = await orchestrator.process_turn(session, transcript)
            total_ms = int((time.monotonic() - t0) * 1000)

            logger.info(
                "turn_complete session=%s total_ms=%d intent=%s",
                session_id,
                total_ms,
                session.conversation_history[-2].intent.intent if len(session.conversation_history) >= 2 else "n/a",
            )

            # Stream TTS audio back to client
            await _send_voice_response(websocket, tts, response_text)

            if session.state in {AgentState.ESCALATE, AgentState.FAILED, AgentState.ENDED}:
                await websocket.close()
                break

    except WebSocketDisconnect:
        logger.info("ws_disconnect session=%s", session_id)
    except Exception as exc:
        logger.exception("ws_error session=%s", session_id, exc_info=exc)
        await websocket.close()
    finally:
        _sessions.pop(session_id, None)


async def _send_voice_response(
    websocket: WebSocket,
    tts: ElevenLabsTTSService,
    text: str,
) -> None:
    """Synthesize text to speech and stream audio chunks over WebSocket."""
    try:
        async for audio_chunk in tts.synthesize_stream(text):
            await websocket.send_bytes(audio_chunk)
        # Send end-of-audio sentinel
        await websocket.send_json({"event": "tts_end"})
    except Exception as exc:
        logger.warning("TTS stream error: %s", exc)

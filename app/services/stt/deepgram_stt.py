import asyncio
import logging
from collections.abc import AsyncIterator

from deepgram import AsyncDeepgramClient
from deepgram.listen import ListenV1Results

from app.models.session import TranscriptEvent, WordTimestamp
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DeepgramSTTService:
    def __init__(self) -> None:
        self._client = AsyncDeepgramClient(api_key=settings.deepgram_api_key)

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes],
        session_id: str,
    ) -> AsyncIterator[TranscriptEvent]:
        """
        Yields TranscriptEvent objects (partial + final) as audio arrives.
        Uses Deepgram SDK v6 async WebSocket API.
        """
        queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

        async with self._client.listen.v1.raw.connect(
            model=settings.stt_model,
            language=settings.stt_language,
            punctuate="true",
            smart_format="true",
            interim_results="true",
            endpointing="1500",
            utterance_end_ms="1500",
        ) as connection:
            async def _send_audio():
                try:
                    async for chunk in audio_stream:
                        await connection.send_media(chunk)
                    await connection.send_finalize()
                    await connection.send_close_stream()
                except Exception as exc:
                    logger.exception("Audio send error session=%s", session_id, exc_info=exc)
                    await queue.put(None)

            send_task = asyncio.create_task(_send_audio())

            async for message in connection:
                if isinstance(message, bytes):
                    continue

                if not hasattr(message, "type"):
                    continue

                # Results message
                if isinstance(message, ListenV1Results):
                    try:
                        alt = message.channel.alternatives[0]
                        words = [
                            WordTimestamp(
                                word=w.word,
                                start_ms=int(w.start * 1000),
                                end_ms=int(w.end * 1000),
                                confidence=w.confidence,
                            )
                            for w in (alt.words or [])
                        ]
                        event = TranscriptEvent(
                            session_id=session_id,
                            text=alt.transcript or "",
                            is_final=message.is_final or False,
                            confidence=alt.confidence or 0.0,
                            timestamp_ms=int((message.start or 0) * 1000),
                            words=words,
                        )
                        if event.text.strip():
                            await queue.put(event)
                    except (AttributeError, IndexError) as exc:
                        logger.warning("STT parse error: %s", exc)

            await send_task
            await queue.put(None)  # signal end

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

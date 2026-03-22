import logging
import re
from collections.abc import AsyncIterator

from elevenlabs import AsyncElevenLabs, VoiceSettings

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ElevenLabsTTSService:
    def __init__(self) -> None:
        self._client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio bytes for a given text string."""
        text = self._clean_for_voice(text)
        async for chunk in await self._client.generate(
            text=text,
            voice=settings.elevenlabs_voice_id,
            model=settings.tts_model,
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.8,
                style=0.0,
                use_speaker_boost=True,
            ),
            stream=True,
        ):
            if chunk:
                yield chunk

    async def synthesize(self, text: str) -> bytes:
        """Return complete audio bytes (non-streaming). Useful for short responses."""
        audio_chunks = []
        async for chunk in self.synthesize_stream(text):
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)

    @staticmethod
    def _clean_for_voice(text: str) -> str:
        """Remove markdown artifacts that would sound bad when spoken."""
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"`+", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

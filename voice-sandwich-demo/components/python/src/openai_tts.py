"""
OpenAI Text-to-Speech Streaming

Converts text to PCM audio using OpenAI's TTS API with HTTP streaming.
Bridges the HTTP stream to an async queue so it fits the same
send_text / receive_events interface used by CartesiaTTS.

Input: Text strings
Output: TTS events (tts_chunk for audio chunks)
"""

import asyncio
import os
from typing import AsyncIterator, Optional

import httpx

from events import TTSChunkEvent


class OpenAITTS:
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "alloy",
        model: str = "gpt-4o-mini-tts",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.voice = voice
        self.model = model
        self._queue: asyncio.Queue[Optional[TTSChunkEvent]] = asyncio.Queue()
        self._close_signal = asyncio.Event()

    async def send_text(self, text: Optional[str]) -> None:
        if not text or not text.strip():
            return
        asyncio.create_task(self._stream_to_queue(text))

    async def _stream_to_queue(self, text: str) -> None:
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": text,
                        "voice": self.voice,
                        "response_format": "pcm",  # raw PCM16 at 24kHz, mono
                    },
                    timeout=30.0,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if chunk:
                            await self._queue.put(TTSChunkEvent.create(chunk))
        except Exception as e:
            print(f"OpenAITTS error: {e}")

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        while not self._close_signal.is_set():
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                yield event
            except asyncio.TimeoutError:
                continue

    async def close(self) -> None:
        self._close_signal.set()

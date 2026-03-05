"""
OpenAI Real-Time Streaming STT Transform

Connects to OpenAI's Realtime API WebSocket for streaming speech-to-text.
Uses server-side VAD for automatic turn detection.

Input: PCM 16-bit audio buffer (bytes)
Output: STT events (stt_chunk for partials, stt_output for final transcripts)
"""

import asyncio
import base64
import contextlib
import json
import os
from typing import AsyncIterator, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from events import STTChunkEvent, STTEvent, STTOutputEvent


class OpenAISTT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        model: str = "gpt-4o-realtime-preview",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.sample_rate = sample_rate
        self.model = model
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connection_signal = asyncio.Event()
        self._close_signal = asyncio.Event()

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        while not self._close_signal.is_set():
            _, pending = await asyncio.wait(
                [
                    asyncio.create_task(self._close_signal.wait()),
                    asyncio.create_task(self._connection_signal.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            with contextlib.suppress(asyncio.CancelledError):
                for task in pending:
                    task.cancel()

            if self._close_signal.is_set():
                break

            if self._ws and self._ws.close_code is None:
                self._connection_signal.clear()
                try:
                    async for raw_message in self._ws:
                        try:
                            message = json.loads(raw_message)
                            event_type = message.get("type")

                            if event_type == "conversation.item.input_audio_transcription.delta":
                                delta = message.get("delta", "")
                                if delta:
                                    yield STTChunkEvent.create(delta)

                            elif event_type == "conversation.item.input_audio_transcription.completed":
                                transcript = message.get("transcript", "")
                                if transcript:
                                    yield STTOutputEvent.create(transcript)

                            elif event_type == "error":
                                print(f"OpenAISTT error: {message.get('error', {})}")
                                break

                        except json.JSONDecodeError as e:
                            print(f"[DEBUG] OpenAISTT JSON decode error: {e}")
                            continue
                except websockets.exceptions.ConnectionClosed:
                    print("OpenAISTT: WebSocket connection closed")

    async def send_audio(self, audio_chunk: bytes) -> None:
        ws = await self._ensure_connection()
        audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))

    async def close(self) -> None:
        if self._ws and self._ws.close_code is None:
            await self._ws.close()
        self._ws = None
        self._close_signal.set()

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._close_signal.is_set():
            raise RuntimeError(
                "OpenAISTT tried establishing a connection after it was closed"
            )
        if self._ws and self._ws.close_code is None:
            return self._ws

        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        self._ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

        # Configure session: enable transcription with server VAD for turn detection.
        # Responses from the Realtime API are ignored; the LangChain agent handles replies.
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-transcribe",
                },
                "turn_detection": {
                    "type": "server_vad",
                },
            },
        }))

        self._connection_signal.set()
        return self._ws

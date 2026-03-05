import asyncio
import audioop
import base64
import contextlib
import json
import os
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from fi_instrumentation import register, using_prompt_template, using_session, using_user
from fi_instrumentation.fi_types import ProjectType, FiSpanKindValues, SpanAttributes
from fi.prompt import Prompt
from opentelemetry import trace
from traceai_langchain import LangChainInstrumentor

trace_provider = register(
    project_type=ProjectType.OBSERVE,
    project_name="voice-sandwich-demo",
)
LangChainInstrumentor().instrument(tracer_provider=trace_provider)
tracer = trace.get_tracer(__name__)

# Fetch system prompt from FutureAGI Prompt Workbench
PROMPT_TEMPLATE_NAME = "sandwich-shop-assistant"
PROMPT_TEMPLATE_LABEL = "Production"
PROMPT_TEMPLATE_VERSION = ""
try:
    prompt_client = Prompt.get_template_by_name(
        PROMPT_TEMPLATE_NAME, label=PROMPT_TEMPLATE_LABEL
    )
    # Access template messages directly — .compile() stringifies the content list
    content_blocks = prompt_client.template.messages[0].content
    if isinstance(content_blocks, list):
        system_prompt = "\n".join(b["text"] for b in content_blocks if "text" in b).strip()
    else:
        system_prompt = str(content_blocks).strip()
    PROMPT_TEMPLATE_VERSION = getattr(prompt_client.template, "version", "")
except Exception:
    system_prompt = None
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_aws import ChatBedrockConverse
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles

from openai_stt import OpenAISTT
from openai_tts import OpenAITTS
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from utils import merge_async_iters

# Static files are served from the shared web build output
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


FALLBACK_SYSTEM_PROMPT = """\
You are a friendly sandwich shop voice assistant taking orders over the phone.

## Voice Rules
- Keep responses SHORT (1-2 sentences max). This is a phone call, not a chat.
- Never say "I didn't catch that" or ask the user to repeat. Instead, make your best guess from context and confirm it.
- Speak naturally like a real person. Use contractions (I'll, we've, what'd).
- Don't list all options at once. Guide the customer step by step.
- If the transcript seems garbled, infer intent from keywords and context.

## Order Flow
1. Greet briefly: "Hey! What can I get for you today?"
2. Ask for bread choice (white, wheat, Italian, wrap)
3. Ask for meat (turkey, ham, roast beef) — or veggie
4. Ask for cheese (swiss, cheddar, provolone) — or none
5. Ask for toppings (lettuce, tomato, onion, pickles)
6. Ask for sauce (mayo, mustard, oil & vinegar) — or none
7. Confirm the full order, then use the confirm_order tool

## Tools
- Use add_to_order whenever the customer picks an item.
- Use confirm_order once the full sandwich is confirmed.

## Handling Unclear Input
- If someone says something unclear like "hamchz" → assume "ham and cheese" and confirm: "Ham and cheese, got it! What bread?"
- If you hear a number, treat it as quantity.
- Always keep the conversation moving forward. Don't stall.
"""

if not system_prompt:
    system_prompt = FALLBACK_SYSTEM_PROMPT

bedrock_model = ChatBedrockConverse(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region_name="us-east-1",
)

agent = create_agent(
    model=bedrock_model,
    tools=[add_to_order, confirm_order],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)

    This function takes a stream of audio chunks and sends them to AssemblyAI for STT.

    It uses a producer-consumer pattern where:
    - Producer: A background task reads audio chunks from audio_stream and sends
      them to AssemblyAI via WebSocket. This runs concurrently with the consumer,
      allowing transcription to begin before all audio has arrived.
    - Consumer: The main coroutine receives transcription events from AssemblyAI
      and yields them downstream. Events include both partial results (stt_chunk)
      and final transcripts (stt_output).

    Args:
        audio_stream: Async iterator of PCM audio bytes (16-bit, mono, 16kHz)

    Yields:
        STT events (stt_chunk for partials, stt_output for final transcripts)
    """
    stt = OpenAISTT(sample_rate=16000)

    async def send_audio():
        """
        Background task that pumps audio chunks to AssemblyAI.

        This runs concurrently with the main coroutine, continuously reading
        audio chunks from the input stream and forwarding them to AssemblyAI.
        When the input stream ends, it signals completion by closing the
        WebSocket connection.
        """
        try:
            # Stream each audio chunk to AssemblyAI as it arrives
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            # Signal to AssemblyAI that audio streaming is complete
            await stt.close()

    # Launch the audio sending task in the background
    # This allows us to simultaneously receive transcripts in the main coroutine
    send_task = asyncio.create_task(send_audio())

    try:
        # Consumer loop: receive and yield transcription events as they arrive
        # from AssemblyAI. The receive_events() method listens on the WebSocket
        # for transcript events and yields them as they become available.
        async for event in stt.receive_events():
            if event.type == "stt_output":
                with tracer.start_as_current_span("stt") as span:
                    span.set_attribute("fi.span.kind", FiSpanKindValues.TOOL)
                    span.set_attribute("audio.transcript", event.transcript)
                    span.set_attribute("tool.name", "speech-to-text")
            yield event
    finally:
        # Cleanup: ensure the background task is cancelled and awaited
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        # Ensure the WebSocket connection is closed
        await stt.close()


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Agent Responses)

    This function takes a stream of upstream voice agent events and processes them.
    When an stt_output event arrives, it passes the transcript to the LangChain agent.
    The agent streams back its response tokens as agent_chunk events.
    Tool calls and results are also emitted as separate events.
    All other upstream events are passed through unchanged.

    The passthrough pattern ensures downstream stages (like TTS) can observe all
    events in the pipeline, not just the ones this stage produces. This enables
    features like displaying partial transcripts while the agent is thinking.

    Args:
        event_stream: An async iterator of upstream voice agent events

    Yields:
        All upstream events plus agent_chunk, tool_call, and tool_result events
    """
    # Generate a unique thread ID for this conversation session
    # This allows the agent to maintain conversation context across multiple turns
    # using the checkpointer (InMemorySaver) configured in the agent
    thread_id = str(uuid4())

    # Process each event as it arrives from the upstream STT stage
    async for event in event_stream:
        # Pass through all events to downstream consumers
        yield event

        # When we receive a final transcript, invoke the agent
        if event.type == "stt_output":
            # Stream the agent's response using LangChain's astream method.
            # stream_mode="messages" yields message chunks as they're generated.
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            # Iterate through the agent's streaming response. The stream yields
            # tuples of (message, metadata), but we only need the message.
            prev_text_len = 0
            async for message, metadata in stream:
                # Emit agent chunks (AI messages)
                if isinstance(message, AIMessage):
                    # Extract only the NEW text delta from each chunk
                    full_text = message.text
                    delta = full_text[prev_text_len:]
                    prev_text_len = len(full_text)
                    if delta:
                        yield AgentChunkEvent.create(delta)
                    # Emit tool calls if present
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                # Emit tool results (tool messages)
                if isinstance(message, ToolMessage):
                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=getattr(message, "name", "unknown"),
                        result=str(message.content) if message.content else "",
                    )

            # Signal that the agent has finished responding for this turn
            yield AgentEndEvent.create()


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Audio)

    This function takes a stream of upstream voice agent events and processes them.
    When agent_chunk events arrive, it sends the text to Cartesia for TTS synthesis.
    Audio is streamed back as tts_chunk events as it's generated.
    All upstream events are passed through unchanged.

    It uses merge_async_iters to combine two concurrent streams:
    - process_upstream(): Iterates through incoming events, yields them for
      passthrough, and sends agent text chunks to Cartesia for synthesis.
    - tts.receive_events(): Yields audio chunks from Cartesia as they are
      synthesized.

    The merge utility runs both iterators concurrently, yielding items from
    either stream as they become available. This allows audio generation to
    begin before the agent has finished generating all text, minimizing latency.

    Args:
        event_stream: An async iterator of upstream voice agent events

    Yields:
        All upstream events plus tts_chunk events for synthesized audio
    """
    tts = OpenAITTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        """
        Process upstream events, yielding them while sending text to Cartesia.

        This async generator serves two purposes:
        1. Pass through all upstream events (stt_chunk, stt_output, agent_chunk)
           so downstream consumers can observe the full event stream.
        2. Buffer agent_chunk text and send to Cartesia when agent_end arrives.
           This ensures the full response is sent at once for better TTS quality.
        """
        buffer: list[str] = []
        async for event in event_stream:
            # Pass through all events to downstream consumers
            yield event
            # User finished speaking — interrupt TTS playback
            # We use stt_output (final transcript) instead of stt_chunk (partials)
            # to avoid false interrupts from echo / background noise on phone calls.
            if event.type == "stt_output":
                tts.interrupt()
                buffer = []
            # Agent starts responding — resume TTS
            if event.type == "agent_chunk" and not buffer:
                tts.resume()
            # Buffer agent text chunks, flush on sentence boundaries
            if event.type == "agent_chunk":
                buffer.append(event.text)
                text_so_far = "".join(buffer)
                if any(text_so_far.rstrip().endswith(p) for p in (".", "!", "?", ":")):
                    with tracer.start_as_current_span("tts") as span:
                        span.set_attribute("fi.span.kind", FiSpanKindValues.TOOL)
                        span.set_attribute("audio.transcript", text_so_far)
                        span.set_attribute("tool.name", "text-to-speech")
                    await tts.send_text(text_so_far)
                    buffer = []
            # Flush any remaining text when agent finishes
            if event.type == "agent_end":
                remaining = "".join(buffer).strip()
                if remaining:
                    with tracer.start_as_current_span("tts") as span:
                        span.set_attribute("fi.span.kind", FiSpanKindValues.TOOL)
                        span.set_attribute("audio.transcript", remaining)
                        span.set_attribute("tool.name", "text-to-speech")
                    await tts.send_text(remaining)
                buffer = []

    try:
        # Merge the processed upstream events with TTS audio events
        # Both streams run concurrently, yielding events as they arrive
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        # Cleanup: close the WebSocket connection to Cartesia
        await tts.close()


def pipeline(audio_stream: AsyncIterator[bytes]) -> AsyncIterator[VoiceAgentEvent]:
    """Chain STT → Agent → TTS as plain async generators (no RunnableSequence)."""
    return _tts_stream(_agent_stream(_stt_stream(audio_stream)))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid4())
    user_id = websocket.query_params.get("user_id", "anonymous")

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        while True:
            data = await websocket.receive_bytes()
            yield data

    with tracer.start_as_current_span("voice-agent-session") as root_span:
        root_span.set_attribute("fi.span.kind", FiSpanKindValues.AGENT)
        with (
            using_session(session_id),
            using_user(user_id),
            using_prompt_template(
                template=system_prompt,
                label=PROMPT_TEMPLATE_LABEL,
                version=PROMPT_TEMPLATE_VERSION,
            ),
        ):
            try:
                output_stream = pipeline(websocket_audio_stream())

                async for event in output_stream:
                    await websocket.send_json(event_to_dict(event))
            except Exception as e:
                print(f"[WS] Session ended: {e}")
                root_span.set_attribute("session.end_reason", str(type(e).__name__))


@app.post("/twiml")
async def twiml_endpoint(request: Request):
    """Return TwiML that tells Twilio to stream audio to our WebSocket."""
    host = request.headers.get("host", "localhost")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/ws/twilio" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws/twilio")
async def twilio_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid4())
    stream_sid = None

    async def twilio_audio_stream() -> AsyncIterator[bytes]:
        """Decode Twilio mulaw base64 media into PCM bytes at 24kHz."""
        while True:
            try:
                raw = await websocket.receive_text()
            except Exception as e:
                print(f"[Twilio] WebSocket receive error: {e}")
                return
            msg = json.loads(raw)
            event_type = msg.get("event")
            if event_type == "media":
                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                # Upsample 8kHz -> 24kHz for OpenAI Realtime API
                pcm_24k = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)[0]
                yield pcm_24k
            elif event_type == "start":
                nonlocal stream_sid
                stream_sid = msg["start"].get("streamSid")
                print(f"[Twilio] Stream started: {stream_sid}")
            elif event_type == "stop":
                print("[Twilio] Stream stopped")
                return
            elif event_type == "connected":
                print("[Twilio] Connected")

    with tracer.start_as_current_span("voice-agent-session") as root_span:
        root_span.set_attribute("fi.span.kind", FiSpanKindValues.AGENT)
        with (
            using_session(session_id),
            using_user("twilio-caller"),
            using_prompt_template(
                template=system_prompt,
                label=PROMPT_TEMPLATE_LABEL,
                version=PROMPT_TEMPLATE_VERSION,
            ),
        ):
            try:
                output_stream = pipeline(twilio_audio_stream())

                async for event in output_stream:
                    # User finished speaking — clear Twilio's audio buffer
                    if event.type == "stt_output" and stream_sid:
                        await websocket.send_text(json.dumps({
                            "event": "clear",
                            "streamSid": stream_sid,
                        }))
                    if event.type == "tts_chunk" and stream_sid:
                        # TTS outputs 24kHz PCM16 mono, Twilio expects 8kHz mulaw
                        pcm_8k = audioop.ratecv(event.audio, 2, 1, 24000, 8000, None)[0]
                        mulaw_bytes = audioop.lin2ulaw(pcm_8k, 2)
                        payload = base64.b64encode(mulaw_bytes).decode("ascii")
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload},
                        }))
            except Exception as e:
                print(f"[Twilio] Session ended: {e}")
                root_span.set_attribute("session.end_reason", str(type(e).__name__))


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8081, reload=True)

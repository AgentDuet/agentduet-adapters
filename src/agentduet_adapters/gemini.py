"""GeminiLive VoiceModel adapter.

Wraps a Gemini Live session as an `agentduet.VoiceModel`, so:
    VoiceAgent.from_env().run(GeminiLive(instruction="...", tools=[...]))

Requires the `gemini` extra:  pip install "agentduet-adapters[gemini]"  (google-genai)
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Optional

from agentduet import (
    AudioOut,
    Interrupted,
    ModelEvent,
    ModelSession,
    ToolCall,
    TranscriptDelta,
    Usage,
    VoiceModel,
)

try:
    from google import genai
    from google.genai import types
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'GeminiLive needs google-genai — install with: pip install "agentduet-adapters[gemini]"'
    ) from e

DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"
SAMPLE_RATE = 24000
GREETING = "The caller has just connected. Greet them."
DEFAULT_INSTRUCTION = "You are a helpful, friendly voice assistant. Keep replies short."

logger = logging.getLogger(__name__)


def _fmt_usage(um) -> str:
    """Compact one-line summary of Gemini Live cumulative token usage (per session)."""

    def _brk(details) -> str:
        return " ".join(
            f"{getattr(d.modality, 'name', d.modality)}={d.token_count}" for d in (details or [])
        )

    parts = [
        f"total={um.total_token_count}",
        f"in={um.prompt_token_count}",
        f"out={um.response_token_count}",
    ]
    if pin := _brk(um.prompt_tokens_details):
        parts.append(f"in[{pin}]")
    if pout := _brk(um.response_tokens_details):
        parts.append(f"out[{pout}]")
    return "  ".join(parts)


def _usage_detail(um) -> dict:
    """Per-modality token breakdown, e.g. {'in_audio': .., 'in_text': .., 'out_audio': ..}."""
    d: dict = {}
    for pref, details in (("in", um.prompt_tokens_details), ("out", um.response_tokens_details)):
        for x in details or []:
            mod = getattr(x.modality, "name", str(x.modality)).lower()
            d[f"{pref}_{mod}"] = x.token_count
    return d


class GeminiLive(VoiceModel):
    """A reusable Gemini Live spec; opens a fresh session per call."""

    def __init__(
        self,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        persona: Optional[str] = None,  # deprecated alias for `instruction`
        tools: Optional[list] = None,
        voice: str = "Zephyr",
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        greeting: str = GREETING,
    ):
        self._instruction = persona if persona is not None else instruction
        self._tool_declarations = tools or []
        self._voice = voice
        self._model = model
        self._greeting = greeting
        self._client = genai.Client(vertexai=False, api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def _live_config(self):
        # Pass named args directly (not **kwargs) so the type checker can validate each
        # against LiveConnectConfig's typed params.
        tools: Optional[list] = (
            [types.Tool(function_declarations=self._tool_declarations)]
            if self._tool_declarations
            else None
        )
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._voice)
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=self._instruction)]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            tools=tools,
        )

    async def open(self) -> ModelSession:
        cm = self._client.aio.live.connect(model=self._model, config=self._live_config())
        session = await cm.__aenter__()
        return _GeminiLiveSession(cm, session, self._greeting)


class _GeminiLiveSession(ModelSession):
    def __init__(self, cm, session, greeting: str):
        self._cm = cm
        self._session = session
        self._greeting = greeting
        self._names: dict[str, str] = {}  # tool call_id -> name (Gemini needs it back)

    async def push_audio(self, pcm: bytes) -> None:
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
        )

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        await self._session.send_tool_response(
            function_responses=[
                types.FunctionResponse(id=call_id, name=self._names.get(call_id), response=result)
            ]
        )

    async def events(self) -> AsyncIterator[ModelEvent]:
        # Gemini Live is input-driven — nudge it to greet first.
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=self._greeting)]),
            turn_complete=True,
        )
        last_tokens = 0
        while True:  # receive() ends each turn — re-enter so the agent stays responsive
            async for resp in self._session.receive():
                um = getattr(resp, "usage_metadata", None)
                if um and (um.total_token_count or 0) != last_tokens:
                    last_tokens = um.total_token_count or 0
                    logger.info("gemini tokens — %s", _fmt_usage(um))
                    yield Usage(
                        total=um.total_token_count or 0,
                        input=um.prompt_token_count or 0,
                        output=um.response_token_count or 0,
                        detail=_usage_detail(um),
                    )
                if resp.tool_call:
                    for fc in resp.tool_call.function_calls:
                        self._names[fc.id] = fc.name
                        yield ToolCall(id=fc.id, name=fc.name, args=dict(fc.args or {}))
                    continue
                sc = resp.server_content
                if not sc:
                    continue
                if sc.interrupted:
                    yield Interrupted()
                if sc.input_transcription and sc.input_transcription.text:
                    yield TranscriptDelta(text=sc.input_transcription.text, role="user")
                if sc.output_transcription and sc.output_transcription.text:
                    yield TranscriptDelta(text=sc.output_transcription.text, role="agent")
                if sc.model_turn:
                    for p in sc.model_turn.parts:
                        if p.inline_data and isinstance(p.inline_data.data, bytes):
                            yield AudioOut(pcm=p.inline_data.data)

    async def close(self) -> None:
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception:
            pass

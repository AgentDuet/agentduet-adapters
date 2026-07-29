"""GrokVoice VoiceModel adapter — xAI Grok Voice (native speech-to-speech).

Wraps xAI's realtime Voice API (grok-voice-think-fast-1.0) as an `agentduet.VoiceModel`,
so it drops in exactly like GeminiLive / NovaSonic:

    VoiceAgent.from_env().run(GrokVoice(instruction="..."))

Grok Voice is a native speech-to-speech model (audio in → audio out, no STT/TTS cascade),
driven over an OpenAI-Realtime-style WebSocket. This adapter uses the "bring your own audio
client" mode (raw PCM in/out) so agentDuet owns the carrier/transport and xAI hosts the model.

Auth: an xAI API key (XAI_API_KEY or api_key=). Needs only `websockets` (a core SDK dep) —
no extra provider SDK. Docs: https://docs.x.ai/developers/model-capabilities/audio/voice-agent

Live-validated against xAI 2026-07-13 (grok-voice-think-fast-1.0): connect + session config,
greeting, audio-out, agent + caller transcripts, server_vad barge-in, and function tool-calling
all confirmed against a real session.
"""

from __future__ import annotations

import base64
import json
import os
from typing import AsyncIterator, Optional

import websockets

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

WS_ENDPOINT = "wss://api.x.ai/v1/realtime"
DEFAULT_MODEL = "grok-voice-think-fast-1.0"
SAMPLE_RATE = 24000
DEFAULT_INSTRUCTION = "You are a helpful, friendly voice assistant. Keep replies short."


def _fold_usage(msg: dict, acc: dict) -> Optional[Usage]:
    """Fold an OpenAI-realtime `response.done` usage block into running per-session totals,
    returning cumulative Usage (None when the message carries no usage)."""
    u = msg.get("usage") or (msg.get("response") or {}).get(
        "usage"
    )  # Grok: top-level; others: nested
    if not u:
        return None
    acc["total"] += u.get("total_tokens") or u.get("total") or 0
    acc["input"] += u.get("input_tokens") or u.get("prompt_tokens") or 0
    acc["output"] += u.get("output_tokens") or u.get("completion_tokens") or 0
    for pref, key in (("in", "input_token_details"), ("out", "output_token_details")):
        for k, v in (u.get(key) or {}).items():
            if isinstance(v, int):
                dk = f"{pref}_{k.replace('_tokens', '')}"
                acc["detail"][dk] = acc["detail"].get(dk, 0) + v
    return Usage(
        total=acc["total"], input=acc["input"], output=acc["output"], detail=dict(acc["detail"])
    )


def _read_xai_key() -> Optional[str]:
    """XAI_API_KEY env, else the raw key in ~/.x.ai, else None."""
    key = os.getenv("XAI_API_KEY")
    if key:
        return key.strip()
    path = os.path.expanduser("~/.x.ai")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


class GrokVoice(VoiceModel):
    """A reusable Grok Voice spec; opens a fresh realtime WS session per call."""

    def __init__(
        self,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        persona: Optional[str] = None,  # deprecated alias for `instruction`
        tools: Optional[list[dict]] = None,
        voice: str = "eve",
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        greet_on_connect: bool = True,
    ):
        # tools: neutral declarations [{"name", "description", "input_schema": {...}}],
        # e.g. straight from MCP list_tools — mapped to xAI function tools on session.update.
        self._instruction = persona if persona is not None else instruction
        self._tools = tools or []
        self._voice = voice
        self._model = model
        self._sample_rate = sample_rate
        self._greet_on_connect = greet_on_connect
        self._api_key = api_key or _read_xai_key()
        if not self._api_key:
            raise RuntimeError(
                "no xAI API key (set XAI_API_KEY, put it in ~/.x.ai, or pass api_key=)"
            )

    def _session_update(self) -> dict:
        fmt = {"type": "audio/pcm", "rate": self._sample_rate}
        session: dict = {
            "voice": self._voice,
            "instructions": self._instruction,
            "turn_detection": {"type": "server_vad"},  # server handles VAD + barge-in
            "audio": {"input": {"format": fmt}, "output": {"format": fmt}},
        }
        if self._tools:
            session["tools"] = [
                {
                    "type": "function",
                    "name": t["name"],
                    "description": (t.get("description") or "").strip(),
                    "parameters": t.get("input_schema") or t.get("parameters") or {},
                }
                for t in self._tools
            ]
        return {"type": "session.update", "session": session}

    async def open(self) -> ModelSession:
        url = f"{WS_ENDPOINT}?model={self._model}"
        ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
        )
        await ws.send(json.dumps(self._session_update()))
        return _GrokVoiceSession(ws, greet=self._greet_on_connect)


class _GrokVoiceSession(ModelSession):
    """One live Grok Voice realtime conversation over a WebSocket."""

    def __init__(self, ws, *, greet: bool = True):
        self._ws = ws
        self._greet = greet

    async def push_audio(self, pcm: bytes) -> None:
        # caller PCM16 (little-endian) at the configured rate → base64 append
        await self._ws.send(
            json.dumps(
                {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
            )
        )

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    },
                }
            )
        )
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def events(self) -> AsyncIterator[ModelEvent]:
        # server_vad is input-driven; nudge a first turn so Grok greets before the caller speaks.
        if self._greet:
            await self._ws.send(json.dumps({"type": "response.create"}))
        acc = {"total": 0, "input": 0, "output": 0, "detail": {}}
        async for raw in self._ws:
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "response.output_audio.delta":
                b64 = msg.get("delta") or msg.get("audio")  # xAI puts audio in `delta`
                if b64:
                    yield AudioOut(pcm=base64.b64decode(b64))
            elif t == "response.output_audio_transcript.delta":
                yield TranscriptDelta(text=msg.get("delta") or msg.get("text", ""), role="agent")
            elif t == "conversation.item.input_audio_transcription.completed":
                yield TranscriptDelta(text=msg.get("transcript", ""), role="user")
            elif t == "response.function_call_arguments.done":
                yield ToolCall(
                    id=msg["call_id"],
                    name=msg["name"],
                    args=json.loads(msg.get("arguments") or "{}"),
                )
            elif t == "input_audio_buffer.speech_started":
                yield Interrupted()  # caller barged in (server_vad)
            elif t == "response.done":
                us = _fold_usage(msg, acc)
                if us:
                    yield us

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass

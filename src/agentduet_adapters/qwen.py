"""QwenVoice VoiceModel adapter — Alibaba Qwen-Omni Realtime (DashScope Model Studio).

Wraps Qwen-Omni realtime as an `agentduet.VoiceModel`, so it drops in exactly like
GeminiLive / NovaSonic / GrokVoice:

    VoiceAgent.from_env().run(QwenVoice(instruction="..."))

Qwen-Omni realtime is a native speech-to-speech model driven over an OpenAI-Realtime-style
WebSocket. Auth: a DashScope / Alibaba Cloud Model Studio API key (DASHSCOPE_API_KEY env, or
the raw key in ~/.qwen). Region: intl (default) or cn via DASHSCOPE_REGION.

Audio rates differ from the other adapters: Qwen expects **16 kHz PCM16 input** and emits
**24 kHz output**. AgentDuet runs the call at 24 kHz, so this adapter resamples the caller's
audio 24 -> 16 kHz on the way in (soxr); the model's 24 kHz output passes straight through.

Needs `websockets` (core) plus `soxr` + `numpy` for resampling (the `[qwen]` extra).

Live-validated 2026-07-15 against DashScope intl (qwen3-omni-flash-realtime): connect,
English + Mandarin understanding, in-language replies, and audio out.
"""

from __future__ import annotations

import base64
import json
import logging
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

try:
    import numpy as np  # ty: ignore[unresolved-import]
    import soxr  # ty: ignore[unresolved-import]
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'QwenVoice needs soxr + numpy for resampling — install with: pip install "agentduet-adapters[qwen]"'
    ) from e

_REGION = os.getenv("DASHSCOPE_REGION", "intl")
_BASE = "dashscope.aliyuncs.com" if _REGION == "cn" else "dashscope-intl.aliyuncs.com"
WS_ENDPOINT = f"wss://{_BASE}/api-ws/v1/realtime"
DEFAULT_MODEL = (
    "qwen3.5-omni-flash-realtime"  # 3.5 supports tool calling; the older qwen3-omni-* does not
)
SAMPLE_RATE = 24000  # AgentDuet call audio (in + out)
QWEN_INPUT_RATE = 16000  # Qwen expects 16 kHz input; its output is 24 kHz
DEFAULT_INSTRUCTION = "You are a helpful, friendly voice assistant. Keep replies short."

logger = logging.getLogger(__name__)


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


def _read_qwen_key() -> Optional[str]:
    """DASHSCOPE_API_KEY env, else the raw key in ~/.qwen, else None."""
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key.strip()
    path = os.path.expanduser("~/.qwen")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


class QwenVoice(VoiceModel):
    """A reusable Qwen-Omni realtime spec; opens a fresh realtime WS session per call."""

    def __init__(
        self,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        persona: Optional[str] = None,  # deprecated alias for `instruction`
        tools: Optional[list[dict]] = None,
        voice: str = "Jennifer",
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        greet_on_connect: bool = True,
    ):
        # tools: neutral declarations [{"name", "description", "input_schema": {...}}],
        # e.g. straight from MCP list_tools — mapped to OpenAI-style function tools on session.update.
        self._instruction = persona if persona is not None else instruction
        self._tools = tools or []
        self._voice = voice
        self._model = model
        self._sample_rate = sample_rate
        self._greet_on_connect = greet_on_connect
        self._api_key = api_key or _read_qwen_key()
        if not self._api_key:
            raise RuntimeError(
                "no DashScope API key (set DASHSCOPE_API_KEY, put it in ~/.qwen, or pass api_key=)"
            )

    def _session_update(self) -> dict:
        session: dict = {
            "modalities": ["text", "audio"],
            "voice": self._voice,
            "instructions": self._instruction,
            "input_audio_format": "pcm16",  # 16 kHz mono in
            "output_audio_format": "pcm24",  # 24 kHz mono out
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
            "input_audio_transcription": {"model": "gummy-realtime-v1"},
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
        return _QwenVoiceSession(ws, in_rate=self._sample_rate, greet=self._greet_on_connect)


class _QwenVoiceSession(ModelSession):
    """One live Qwen-Omni realtime conversation over a WebSocket."""

    def __init__(self, ws, *, in_rate: int = SAMPLE_RATE, greet: bool = True):
        self._ws = ws
        self._in_rate = in_rate
        self._greet = greet

    def _to_qwen_rate(self, pcm: bytes) -> bytes:
        """Resample caller PCM16 from the call rate down to Qwen's 16 kHz input."""
        if self._in_rate == QWEN_INPUT_RATE or not pcm:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        resampled = soxr.resample(samples, self._in_rate, QWEN_INPUT_RATE)
        return resampled.astype(np.int16).tobytes()

    async def push_audio(self, pcm: bytes) -> None:
        await self._ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(self._to_qwen_rate(pcm)).decode(),
                }
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
        # server_vad is input-driven; nudge a first turn so Qwen greets before the caller speaks.
        # qwen3.5 rejects a bare response.create on an empty conversation ("input messages do not
        # contain elements with the role of user"), so seed a user turn first.
        if self._greet:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "The call just connected. Greet the caller.",
                                }
                            ],
                        },
                    }
                )
            )
            await self._ws.send(json.dumps({"type": "response.create"}))
        seen: set = set()
        acc = {"total": 0, "input": 0, "output": 0, "detail": {}}
        async for raw in self._ws:
            msg = json.loads(raw)
            t = msg.get("type")
            # --- diagnostics: surface Qwen's event vocabulary + any tool/function frames ---
            if t not in seen:
                seen.add(t)
                logger.info("qwen event type: %s", t)
            if t and ("function" in t or "tool" in t):
                logger.info("qwen TOOL frame: %s", json.dumps(msg)[:1000])
            # -------------------------------------------------------------------------------
            if t == "response.audio.delta":
                b64 = msg.get("delta")
                if b64:
                    yield AudioOut(pcm=base64.b64decode(b64))  # 24 kHz, matches the call
            elif t == "response.audio_transcript.done":
                # full turn transcript (clean line); avoids mid-word truncation from deltas
                yield TranscriptDelta(text=msg.get("transcript") or "", role="agent")
            elif t == "conversation.item.input_audio_transcription.completed":
                yield TranscriptDelta(text=msg.get("transcript", ""), role="user")
            elif t == "response.function_call_arguments.done":
                yield ToolCall(
                    id=msg.get("call_id"),
                    name=msg.get("name"),
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

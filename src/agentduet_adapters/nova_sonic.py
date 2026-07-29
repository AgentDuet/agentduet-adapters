"""NovaSonic VoiceModel adapter.

Wraps an Amazon Nova Sonic bidirectional-stream session as an `agentduet.VoiceModel`, so:
    VoiceAgent.from_env().run(NovaSonic(instruction="..."))

Requires the `nova-sonic` extra:  pip install "agentduet-adapters[nova-sonic]"
(aws-sdk-bedrock-runtime + smithy-aws-core)

Auth is the standard AWS credential chain. The default resolver reads the AWS_*
environment variables only; if your creds live in ~/.aws/credentials, either export
them (`eval "$(aws configure export-credentials --format env)"`) or pass your own
`credentials_resolver=`. Nova Sonic is region-limited (e.g. us-east-1, ap-northeast-1) —
set AWS_REGION or pass region=.

Greeting (`greet_on_connect`) & Nova's VAD
------------------------------------------
Nova is speech-driven — it only responds after an audio turn ends, so it will NOT greet
from a text nudge. To greet first, `greet_on_connect` pushes a bundled "hello" word (an
OSS, non-Amazon voice) as inaudible USER audio; Nova then takes the first turn and greets
per the persona, in its own voice.

The non-obvious part is Nova's VAD: it needs a real NOISE floor — NOT digital silence —
around the audio to register turn boundaries. Zero-padding fails; low-level background
noise works. Hence:
  * Live calls: the caller's real line noise (flowing in right after connect) supplies the
    trailing noise floor that marks end-of-turn, so the adapter adds NO synthetic padding.
  * Tests: inject synthetic comfort noise to stand in for the live line
    (see tests/smoke/test_nova_greeting_smoke.py).
The trigger's voice/engine and onset-trimming don't matter — the noise floor does. For a
good greeting, describe it in your persona (e.g. "the caller greets you with 'hello' —
greet them, introduce yourself, and ask how you can help").

WATCH UPSTREAM: the "hello" trigger is a workaround for Nova having no assistant-initiated
turn. AWS will likely add native "assistant speaks first" (an initial-response / kickoff
API) — when they do, drop the trigger and use it. Revisit `greet_on_connect` then.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from functools import lru_cache
from importlib.resources import files
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
    from aws_sdk_bedrock_runtime.client import (  # ty: ignore[unresolved-import]
        BedrockRuntimeClient,
        InvokeModelWithBidirectionalStreamOperationInput,
    )
    from aws_sdk_bedrock_runtime.config import Config  # ty: ignore[unresolved-import]
    from aws_sdk_bedrock_runtime.models import (  # ty: ignore[unresolved-import]
        BidirectionalInputPayloadPart,
        InvokeModelWithBidirectionalStreamInputChunk,
    )
    from smithy_aws_core.identity.environment import (  # ty: ignore[unresolved-import]
        EnvironmentCredentialsResolver,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "NovaSonic needs aws-sdk-bedrock-runtime + smithy-aws-core — "
        'install with: pip install "agentduet-adapters[nova-sonic]"'
    ) from e

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "amazon.nova-2-sonic-v1:0"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "matthew"
DEFAULT_INSTRUCTION = "You are a helpful, friendly voice assistant. Keep replies short."
# Nova signals barge-in with this marker inside a textOutput content payload.
_INTERRUPT_MARKER = '"interrupted" : true'


@lru_cache(maxsize=1)
def _greeting_trigger() -> bytes:
    """Bundled 'hello' word (Kokoro, OSS) — 24kHz mono 16-bit PCM, just the word (no padding).

    Pushed to Nova as inaudible USER audio on connect so it takes the first turn and greets
    per the persona, in its own voice. Nova's VAD needs a NOISE floor after the word (not
    digital silence) to register end-of-turn — in a live call the caller's real background
    noise supplies that trailing floor; headless tests inject synthetic comfort noise.
    (Voice/engine and onset-trim don't matter — the noise floor does.)
    """
    return (files("agentduet_adapters") / "_assets" / "nova_hello.pcm").read_bytes()


class NovaSonic(VoiceModel):
    """A reusable Nova Sonic spec; opens a fresh bidirectional stream per call."""

    def __init__(
        self,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        persona: Optional[str] = None,  # deprecated alias for `instruction`
        tools: Optional[list[dict]] = None,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        region: Optional[str] = None,
        greet_on_connect: bool = True,  # push a bundled "hello" so Nova greets first
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        credentials_resolver: Optional[object] = None,
    ):
        # tools: neutral declarations [{"name", "description", "input_schema": {...}}],
        # e.g. straight from MCP list_tools — mapped to Nova toolSpec on promptStart.
        self._instruction = persona if persona is not None else instruction
        self._tools = tools or []
        self._voice = voice
        self._model = model
        self._greet_on_connect = greet_on_connect
        self._inference = {"maxTokens": max_tokens, "topP": top_p, "temperature": temperature}

        region = region or os.getenv("AWS_REGION", "us-east-1")
        self._client = BedrockRuntimeClient(
            config=Config(
                endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
                region=region,
                aws_credentials_identity_resolver=credentials_resolver
                or EnvironmentCredentialsResolver(),
            )
        )

    async def open(self) -> ModelSession:
        stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model)
        )
        session = _NovaSonicSession(
            stream,
            instruction=self._instruction,
            voice=self._voice,
            tools=self._tools,
            greet_on_connect=self._greet_on_connect,
            inference=self._inference,
        )
        await session._start()
        return session


class _NovaSonicSession(ModelSession):
    """One live Nova Sonic conversation over a bidirectional stream."""

    def __init__(self, stream, *, instruction, voice, tools, greet_on_connect, inference):
        self._stream = stream
        self._instruction = instruction
        self._voice = voice
        self._tools = tools
        self._greet_on_connect = greet_on_connect
        self._inference = inference
        self._active = False

        self._prompt = str(uuid.uuid4())
        self._sys_content = str(uuid.uuid4())
        self._audio_content = str(uuid.uuid4())

    async def _send(self, event: dict) -> None:
        # json.dumps (not f-strings) so quotes/newlines in the persona can't break the JSON.
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps(event).encode("utf-8"))
        )
        await self._stream.input_stream.send(chunk)  # ty: ignore[unresolved-attribute]

    def _tool_config(self) -> Optional[dict]:
        if not self._tools:
            return None
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        # Bedrock expects the JSON Schema as a string under "json".
                        "inputSchema": {"json": json.dumps(t.get("input_schema", {}))},
                    }
                }
                for t in self._tools
            ]
        }

    async def _start(self) -> None:
        self._active = True
        await self._send({"event": {"sessionStart": {"inferenceConfiguration": self._inference}}})

        prompt_start: dict = {
            "promptName": self._prompt,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": SAMPLE_RATE,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": self._voice,
                "encoding": "base64",
                "audioType": "SPEECH",
            },
        }
        tool_config = self._tool_config()
        if tool_config:
            prompt_start["toolConfiguration"] = tool_config
        await self._send({"event": {"promptStart": prompt_start}})

        # Persona as a SYSTEM text turn.
        await self._text_turn(
            self._sys_content, self._instruction, role="SYSTEM", interactive=False
        )

        # Open the interactive USER audio channel (stays open for the call).
        await self._send(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt,
                        "contentName": self._audio_content,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": SAMPLE_RATE,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            }
        )

        # Greet first (optional): push the bundled "hello" word so Nova takes the first turn
        # and greets per the persona, in its own voice — inaudible to the caller (fed as USER
        # audio). The live mic's real background noise that follows supplies the trailing
        # end-of-turn (Nova's VAD needs a noise floor, not digital silence). No synthetic
        # padding here — headless tests inject their own comfort noise to stand in.
        if self._greet_on_connect:
            trigger = _greeting_trigger()
            for i in range(0, len(trigger), 3200):
                await self.push_audio(trigger[i : i + 3200])

    async def _text_turn(
        self, content_name: str, text: str, *, role: str, interactive: bool
    ) -> None:
        await self._send(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": interactive,
                        "role": role,
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self._send(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt,
                        "contentName": content_name,
                        "content": text,
                    }
                }
            }
        )
        await self._send(
            {"event": {"contentEnd": {"promptName": self._prompt, "contentName": content_name}}}
        )

    async def push_audio(self, pcm: bytes) -> None:
        await self._send(
            {
                "event": {
                    "audioInput": {
                        "promptName": self._prompt,
                        "contentName": self._audio_content,
                        "content": base64.b64encode(pcm).decode("utf-8"),
                    }
                }
            }
        )

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        # NOTE: Nova's tool-result shape is best-effort here — verify against a live
        # tool-calling run before relying on it.
        content_name = str(uuid.uuid4())
        await self._send(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt,
                        "contentName": content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": call_id,
                            "type": "TEXT",
                            "textInputConfiguration": {"mediaType": "text/plain"},
                        },
                    }
                }
            }
        )
        await self._send(
            {
                "event": {
                    "toolResult": {
                        "promptName": self._prompt,
                        "contentName": content_name,
                        "content": json.dumps(result),
                    }
                }
            }
        )
        await self._send(
            {"event": {"contentEnd": {"promptName": self._prompt, "contentName": content_name}}}
        )

    async def events(self) -> AsyncIterator[ModelEvent]:
        # The greeting (if any) was already sent in _start(), before the audio channel.
        try:
            while self._active:
                output = await self._stream.await_output()  # ty: ignore[unresolved-attribute]
                result = await output[1].receive()
                if not (result.value and result.value.bytes_):
                    continue
                event = json.loads(result.value.bytes_.decode("utf-8")).get("event")
                if not event:
                    continue

                if "audioOutput" in event:
                    yield AudioOut(pcm=base64.b64decode(event["audioOutput"]["content"]))
                elif "textOutput" in event:
                    content = event["textOutput"].get("content", "")
                    if _INTERRUPT_MARKER in content:
                        yield Interrupted()
                    else:
                        role = "user" if event["textOutput"].get("role") == "USER" else "agent"
                        yield TranscriptDelta(text=content, role=role)
                elif "toolUse" in event:
                    tu = event["toolUse"]
                    raw = tu.get("content")
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    yield ToolCall(
                        id=tu.get("toolUseId", ""), name=tu.get("toolName", ""), args=args
                    )
                elif "usageEvent" in event:
                    # Nova reports cumulative totals (per session) + a speech/text split under details.total.
                    ue = event["usageEvent"]
                    tot = (ue.get("details") or {}).get("total") or {}
                    detail = {}
                    for pref, sub in (
                        ("in", tot.get("input") or {}),
                        ("out", tot.get("output") or {}),
                    ):
                        for k, v in sub.items():  # speechTokens / textTokens
                            if isinstance(v, int):
                                detail[f"{pref}_{k.replace('Tokens', '').lower()}"] = v
                    yield Usage(
                        total=ue.get("totalTokens") or 0,
                        input=ue.get("totalInputTokens") or 0,
                        output=ue.get("totalOutputTokens") or 0,
                        detail=detail,
                    )
        except Exception:
            logger.debug("Nova receive loop ended", exc_info=True)

    async def close(self) -> None:
        self._active = False
        if not self._stream:
            return
        try:
            await self._send(
                {
                    "event": {
                        "contentEnd": {
                            "promptName": self._prompt,
                            "contentName": self._audio_content,
                        }
                    }
                }
            )
            await self._send({"event": {"promptEnd": {"promptName": self._prompt}}})
            await self._send({"event": {"sessionEnd": {}}})
        except Exception:
            pass
        finally:
            # Close both directions so a pending output receive() ends cleanly — reduces an
            # intermittent aws-crt cancellation race seen on teardown. Best-effort.
            for closer in (self._stream.input_stream.close, self._stream.close):
                try:
                    await closer()
                except Exception:
                    pass
            self._stream = None

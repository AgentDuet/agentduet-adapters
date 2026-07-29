# agentduet-adapters

Model adapters for the [AgentDuet](https://agentduet.com) Python SDK's `VoiceAgent` layer.

AgentDuet connects an AI agent to real phone calls: it owns the carrier side (SIP trunking,
numbers, WhatsApp, web chat) and hands your code a bidirectional PCM audio stream. These
adapters connect that stream to a speech-to-speech model, so a working voice agent is three
lines:

```python
from agentduet import VoiceAgent
from agentduet_adapters.gemini import GeminiLive

VoiceAgent.from_env().run(
    GeminiLive(instruction="You are May, a warm phone concierge. Keep replies short.")
)
```

Call your number and the agent picks up.

## Install

The SDK comes as a dependency; pick the provider you actually call:

```bash
pip install "agentduet-adapters[gemini]"      # Google Gemini Live
pip install "agentduet-adapters[grok]"        # xAI Grok Voice
pip install "agentduet-adapters[qwen]"        # Alibaba Qwen-Omni Realtime
pip install "agentduet-adapters[nova-sonic]"  # Amazon Nova Sonic
```

Nothing vendor-specific installs by default, so a Nova Sonic deployment never ships
`google-genai`. Each adapter module raises `ImportError` naming its extra if you import it
without one.

## The adapters

| Import | Class | Default model | Credentials |
|---|---|---|---|
| `agentduet_adapters.gemini` | `GeminiLive` | `models/gemini-3.1-flash-live-preview` | `GEMINI_API_KEY` |
| `agentduet_adapters.grok_voice` | `GrokVoice` | `grok-voice-think-fast-1.0` | `XAI_API_KEY`, or the raw key in `~/.x.ai` |
| `agentduet_adapters.qwen` | `QwenVoice` | `qwen3.5-omni-flash-realtime` | `DASHSCOPE_API_KEY`, or the raw key in `~/.qwen`; `DASHSCOPE_REGION` selects `intl` (default) or `cn` |
| `agentduet_adapters.nova_sonic` | `NovaSonic` | `amazon.nova-2-sonic-v1:0` | standard AWS environment credentials, `AWS_REGION` |

Every adapter takes the same four arguments: `instruction=`, `tools=`, `voice=` and `model=`.
Swapping providers is a one-line change. Credentials are the one deliberate exception:
Gemini, Grok and Qwen take `api_key=`, while Nova Sonic follows the AWS credential chain and
takes `region=` (plus an optional `credentials_resolver=`).

Two provider quirks worth knowing, because they are not bugs in the adapters:

- **Qwen** runs at 16 kHz input and 24 kHz output while the call runs at 24 kHz, so the
  adapter resamples inbound audio (that is what the `soxr` and `numpy` dependencies are for).
- **Qwen tool calling needs a `qwen3.5-omni-*` model.** The older `qwen3-omni-*` models do
  not support it, which is why the default is 3.5.

## Tools, transcripts and usage

`VoiceAgent` handles these, not the adapters: pass callbacks and every adapter reports through
the same path.

```python
async def tools(name: str, args: dict) -> dict:
    if name == "get_balance":
        return {"balance": 402.15, "currency": "SGD"}
    return {"error": f"unknown tool {name}"}

async def on_transcript(ev):   # ev.role is "user" or "agent"
    print(f"{ev.role}: {ev.text}")

async def on_usage(ev):        # cumulative for this call
    print(f"tokens: {ev.total} (in {ev.input} / out {ev.output})")

VoiceAgent.from_env(tools=tools, on_transcript=on_transcript, on_usage=on_usage).run(
    GrokVoice(instruction="You are a bank concierge.", tools=[
        {"name": "get_balance", "description": "Balance for the caller",
         "input_schema": {"type": "object", "properties": {}}},
    ])
)
```

Tool declarations are passed in a neutral shape (`name`, `description`, `input_schema`) and
each adapter translates it to its provider's format. A tool that raises, or one the model
calls with no handler registered, returns an error result to the model rather than ending the
call.

## Writing your own adapter

You do not need this package for that. The seam is in the SDK: implement `VoiceModel` and
`ModelSession` from `agentduet` and pass your object to `VoiceAgent`.

```python
from agentduet import AudioOut, Interrupted, ToolCall, TranscriptDelta, Usage

class MyModel:
    async def open(self):                              # once per call
        return MySession(await connect_to_my_provider())

class MySession:
    async def push_audio(self, pcm: bytes) -> None:    # caller audio, 24 kHz PCM16
        ...
    def events(self):                                  # async iterator
        # yield AudioOut(pcm=...) to speak, Interrupted() on barge-in,
        # ToolCall(id=..., name=..., args={...}), TranscriptDelta(text=..., role=...),
        # Usage(total=..., input=..., output=...)
        ...
    async def send_tool_result(self, call_id: str, result: dict) -> None:
        ...
    async def close(self) -> None:
        ...
```

The four adapters here are the worked examples: `gemini.py` is the shortest, `nova_sonic.py`
the most involved. `VoiceAgent`'s side of the contract (what it guarantees to call, and when)
is specified in the SDK's `specs/wire-protocol-spec.md`, section 10.5.

## Why this is a separate package

The SDK is a stable transport library: its only dependencies are `websockets`, `httpx` and
`abxbus`. Adapters are the opposite kind of code. They track fast-moving provider APIs, they
each drag in a vendor SDK, and they are the part you are most likely to want to read, fork or
fix yourself. Splitting them out means a provider's breaking change ships as a patch here
instead of forcing a core SDK release, and these can be Apache-2.0 while the SDK is not.

## Examples

`examples/` has a runnable quickstart per provider. Each needs `AGENTDUET_API_KEY` and
`AGENTDUET_CONNECTOR_UUID` (from B3) plus that provider's key:

```bash
pip install "agentduet-adapters[gemini,examples]"
python examples/voice_agent_gemini.py     # then call your number
```

## License

Apache-2.0. The `agentduet` SDK itself is separately licensed.

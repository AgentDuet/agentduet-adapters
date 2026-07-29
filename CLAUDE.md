# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Open-source (Apache-2.0) model adapters for the `agentduet` Python SDK's `VoiceAgent` layer.
Four adapters, one per speech-to-speech provider: `gemini.py`, `grok_voice.py`, `qwen.py`,
`nova_sonic.py`. Each is a `VoiceModel` that opens one `ModelSession` per phone call.

The SDK (`B3Networks/wss-sdk-python`, private, proprietary) owns the seam and the runner. This
repo owns only the vendor half. That split is the point: provider APIs move weekly and each
drags in a vendor SDK, so a Grok protocol change ships as a patch here instead of forcing a
core SDK release.

## Commands

```bash
uv venv && uv pip install -e . --no-deps      # the package itself
uv pip install "google-genai" soxr numpy aws-sdk-bedrock-runtime smithy-aws-core  # vendor SDKs
uv run pytest -q                              # conformance suite (no network, no credentials)
uv run ruff format src tests examples && uv run ruff check src tests examples
uv build                                      # wheel + sdist
```

**The published SDK floor (`agentduet>=1.0.0b10`) may not exist on PyPI yet.** The
`VoiceAgent` layer landed on the SDK's `main` after `1.0.0b9` was published, so until the next
SDK release, install the SDK from a local checkout instead:

```bash
uv pip install -e ../wss-sdk-python           # sibling clone, on main or later
```

Do not "fix" this by lowering the floor: `1.0.0b9` does not export the seam, and an adapter
installed against it fails at import with a confusing name error.

## Rules

- **Never import from a private SDK module.** Everything an adapter needs is exported from
  `agentduet` itself (`VoiceModel`, `ModelSession`, `AudioOut`, `Interrupted`, `ToolCall`,
  `TranscriptDelta`, `Usage`). A relative or deep import worked when these files lived inside
  the SDK; from here it is a break waiting for the next SDK refactor.
- **Never re-export an adapter from `__init__.py`.** Each module raises `ImportError` if its
  vendor SDK is absent, so a package-level re-export would make `import agentduet_adapters`
  require every provider's SDK at once. The test suite asserts this.
- **Keep the constructor vocabulary identical** across adapters: `instruction`, `tools`,
  `voice`, `model`, then provider-specific credentials (`api_key`, or `region` for Nova
  Sonic's AWS chain). Swapping providers must stay a one-line change. Enforced by
  `test_adapters_share_one_constructor_vocabulary`.
- **Tool declarations are neutral** (`{"name", "description", "input_schema"}`, e.g. straight
  from MCP `list_tools`); each adapter translates to its provider's format. Do not leak a
  provider's tool schema into the public argument.
- **Pin ruff below 0.16** (matches the SDK repo; 0.16 widens the default rule set).
- Every adapter change should say in the docstring what was live-validated and when, as
  `gemini.py`, `grok_voice.py` and `qwen.py` already do. "It imports" is not validation.

## Cross-repo changes

A change to the seam is a change to *two* repos. Land the SDK side first (it is the
dependency), then this side. If a rename is unavoidable, the adapters break at import time,
which the conformance suite catches only when both are installed together, so run it against
the local SDK checkout before pushing either.

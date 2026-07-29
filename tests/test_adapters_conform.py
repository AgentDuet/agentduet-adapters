"""Every adapter must satisfy the SDK's seam, and say so when its extra is missing.

These are the two things that break when this package and the SDK move independently:

1. A rename on either side of the seam. In one repo the type checker caught it; across two
   repos nothing does until someone installs both, which is what this suite is.
2. The install story. An adapter imported without its extra must fail with a message naming
   the extra. Get that wrong and the user sees `ModuleNotFoundError: google` and has no idea
   which extra they were supposed to install.

No network, no credentials, and no adapter is constructed (several open a provider client in
__init__). Protocol conformance is checked with issubclass, which works because VoiceModel
and ModelSession are method-only protocols.
"""

import importlib

import pytest
from agentduet import ModelSession, VoiceModel

# module, public adapter class, private session class, the extra that installs it
ADAPTERS = [
    ("agentduet_adapters.gemini", "GeminiLive", "_GeminiLiveSession", "gemini"),
    ("agentduet_adapters.grok_voice", "GrokVoice", "_GrokVoiceSession", "grok"),
    ("agentduet_adapters.qwen", "QwenVoice", "_QwenVoiceSession", "qwen"),
    ("agentduet_adapters.nova_sonic", "NovaSonic", "_NovaSonicSession", "nova-sonic"),
]
IDS = [a[1] for a in ADAPTERS]


def _import_or_skip(module: str, extra: str):
    """Import an adapter, or skip if its vendor SDK is absent.

    A missing extra must announce itself: the adapters raise ImportError naming the extra.
    Asserting on that message is what stops this from also skipping a *real* breakage, such
    as the seam import at the top of the module failing.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        assert f"agentduet-adapters[{extra}]" in str(e), (
            f"{module} failed to import for a reason other than its missing extra: {e}"
        )
        pytest.skip(f"{extra} extra not installed")


@pytest.mark.parametrize("module,cls_name,session_name,extra", ADAPTERS, ids=IDS)
def test_adapter_is_a_voice_model(module, cls_name, session_name, extra):
    mod = _import_or_skip(module, extra)
    cls = getattr(mod, cls_name)
    assert issubclass(cls, VoiceModel)
    assert callable(getattr(cls, "open"))


@pytest.mark.parametrize("module,cls_name,session_name,extra", ADAPTERS, ids=IDS)
def test_adapter_session_is_a_model_session(module, cls_name, session_name, extra):
    mod = _import_or_skip(module, extra)
    session_cls = getattr(mod, session_name)
    assert issubclass(session_cls, ModelSession)
    for method in ("push_audio", "events", "send_tool_result", "close"):
        assert callable(getattr(session_cls, method)), f"{session_name}.{method} is missing"


@pytest.mark.parametrize("module,cls_name,session_name,extra", ADAPTERS, ids=IDS)
def test_adapters_share_one_constructor_vocabulary(module, cls_name, session_name, extra):
    # The README promises swapping providers is a one-line change. That is only true while
    # every adapter takes the same words for the same things.
    import inspect

    mod = _import_or_skip(module, extra)
    params = inspect.signature(getattr(mod, cls_name).__init__).parameters
    for name in ("instruction", "tools", "voice", "model"):
        assert name in params, f"{cls_name} is missing the common '{name}' argument"
    assert params["instruction"].default, f"{cls_name}.instruction needs a usable default"
    # Credentials are the one place the vocabulary legitimately splits: an API-key provider
    # takes `api_key`, while Nova Sonic uses the AWS credential chain (`region` +
    # `credentials_resolver`). Both must be reachable without editing the adapter.
    assert {"api_key", "credentials_resolver"} & set(params), (
        f"{cls_name} offers no way to pass credentials explicitly"
    )


@pytest.mark.parametrize("module,cls_name,session_name,extra", ADAPTERS, ids=IDS)
def test_adapter_declares_a_default_model(module, cls_name, session_name, extra):
    mod = _import_or_skip(module, extra)
    assert isinstance(mod.DEFAULT_MODEL, str) and mod.DEFAULT_MODEL


def test_the_seam_lives_in_the_sdk_not_here():
    # If the SDK ever stops exporting one of these, every adapter breaks at import time.
    # Naming them here turns that into one clear failure instead of four confusing ones.
    import agentduet

    for name in (
        "VoiceAgent",
        "VoiceModel",
        "ModelSession",
        "ModelEvent",
        "AudioOut",
        "Interrupted",
        "ToolCall",
        "TranscriptDelta",
        "Usage",
    ):
        assert hasattr(agentduet, name), f"the installed agentduet SDK does not export {name}"


def test_importing_the_package_does_not_need_any_vendor_sdk():
    # The package __init__ deliberately re-exports nothing: importing it must not drag in
    # google-genai, soxr, or the AWS preview SDKs.
    import agentduet_adapters

    assert agentduet_adapters.__doc__
    for attr in ("GeminiLive", "GrokVoice", "QwenVoice", "NovaSonic"):
        assert not hasattr(agentduet_adapters, attr), (
            f"{attr} is re-exported at package level, which forces every vendor SDK to install"
        )

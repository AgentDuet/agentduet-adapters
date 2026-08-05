"""GeminiLive-specific tests.

Guards a real bug found empirically (2026-08-05): audio generation and the
output_audio_transcription text draw from separate token budgets on Gemini Live. Leaving
max_output_tokens unset silently truncates the TRANSCRIPT partway through a longer reply
(confirmed ~20-25 words in) while the AUDIO the caller hears keeps playing in full — so the
bug is invisible in the call itself and only shows up in anything that reads TranscriptDelta
(logging, guardrails). Config-only fix, so a config-only test is what catches a regression.
"""

from agentduet_adapters.gemini import GeminiLive


def test_max_output_tokens_is_set_generously_by_default():
    # Must be set to SOMETHING non-trivial, or the transcript-truncation bug is back.
    live_config = GeminiLive()._live_config()
    assert live_config.max_output_tokens is not None
    assert live_config.max_output_tokens >= 4096


def test_max_output_tokens_is_configurable():
    live_config = GeminiLive(max_output_tokens=2048)._live_config()
    assert live_config.max_output_tokens == 2048

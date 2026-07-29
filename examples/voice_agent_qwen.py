"""Qwen-Omni quickstart — agentDuet VoiceAgent driving Alibaba Qwen-Omni Realtime.

agentDuet owns the carrier/transport; Alibaba hosts the model. Needs:
    DASHSCOPE_API_KEY                            (Alibaba Cloud Model Studio; or ~/.qwen)
    DASHSCOPE_REGION                             (optional: intl by default, or cn)
    AGENTDUET_API_KEY / AGENTDUET_CONNECTOR_UUID (from B3)

Install:  pip install "agentduet-adapters[qwen]"
Run:      python examples/voice_agent_qwen.py   (then call your number)

The adapter resamples the caller's 24 kHz audio down to the 16 kHz Qwen expects; its
24 kHz output passes straight through.
"""

from agentduet import VoiceAgent

from agentduet_adapters.qwen import QwenVoice

if __name__ == "__main__":
    VoiceAgent.from_env().run(
        QwenVoice(
            instruction="You are May, a warm phone concierge for OmniBank. Keep replies short."
        )
    )

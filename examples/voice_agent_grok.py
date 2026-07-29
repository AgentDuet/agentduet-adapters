"""Grok Voice quickstart — agentDuet VoiceAgent driving xAI Grok Voice (speech-to-speech).

agentDuet owns the carrier/transport; xAI hosts the model. Needs:
    XAI_API_KEY                                  (xAI console)
    AGENTDUET_API_KEY / AGENTDUET_CONNECTOR_UUID (from B3)

Run:  uv run python examples/voice_agent_grok.py   (then call your number)
"""

from agentduet import VoiceAgent
from agentduet_adapters.grok_voice import GrokVoice

if __name__ == "__main__":
    VoiceAgent.from_env().run(
        GrokVoice(
            instruction="You are May, a warm phone concierge for OmniBank. Keep replies short."
        )
    )

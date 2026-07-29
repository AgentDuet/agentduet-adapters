"""Gemini Live quickstart — agentDuet VoiceAgent driving Google Gemini Live (speech-to-speech).

agentDuet owns the carrier/transport; Google hosts the model. Needs:
    GEMINI_API_KEY                               (Google AI Studio)
    AGENTDUET_API_KEY / AGENTDUET_CONNECTOR_UUID (from B3)

Install:  pip install "agentduet-adapters[gemini]"
Run:      python examples/voice_agent_gemini.py   (then call your number)
"""

from agentduet import VoiceAgent

from agentduet_adapters.gemini import GeminiLive

if __name__ == "__main__":
    VoiceAgent.from_env().run(
        GeminiLive(
            instruction="You are May, a warm phone concierge for OmniBank. Keep replies short."
        )
    )

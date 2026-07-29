"""Model adapters for the AgentDuet `VoiceAgent` layer.

Each adapter implements `agentduet.VoiceModel`, so any of them drops into the same runner:

    from agentduet import VoiceAgent
    from agentduet_adapters.gemini import GeminiLive

    VoiceAgent.from_env().run(GeminiLive(instruction="You are a friendly assistant."))

Import the adapter you want from its own module, never from this package. Each one raises
ImportError at import time if its provider's SDK is missing, so a package-level re-export
here would make `import agentduet_adapters` require *every* provider's SDK at once:

    agentduet_adapters.gemini      GeminiLive   pip install "agentduet-adapters[gemini]"
    agentduet_adapters.grok_voice  GrokVoice    pip install "agentduet-adapters[grok]"
    agentduet_adapters.qwen        QwenVoice    pip install "agentduet-adapters[qwen]"
    agentduet_adapters.nova_sonic  NovaSonic    pip install "agentduet-adapters[nova-sonic]"

Writing your own adapter needs nothing from this package: implement `VoiceModel` and
`ModelSession` from `agentduet` and pass it to `VoiceAgent`. These four are worked examples
as much as they are a product.
"""

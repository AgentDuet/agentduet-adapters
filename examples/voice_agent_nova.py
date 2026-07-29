"""Nova Sonic voice agent via the high-level helper — the whole thing in a few lines.

Prereqs:
    pip install "agentduet-adapters[nova-sonic]"
    export AGENTDUET_API_KEY=...          # B3 connector
    export AGENTDUET_CONNECTOR_UUID=...
    # AWS creds + a region where Nova Sonic is available (e.g. ap-northeast-1, us-east-1):
    eval "$(aws configure export-credentials --format env)"   # if creds live in ~/.aws
    export AWS_REGION=ap-northeast-1
"""

from agentduet import VoiceAgent
from agentduet_adapters.nova_sonic import NovaSonic

if __name__ == "__main__":
    VoiceAgent.from_env().run(
        NovaSonic(
            instruction=(
                "You are May, a warm phone concierge for OmniBank. The caller greets you "
                "with 'hello' — greet them back, introduce yourself, and ask how you can "
                "help. Keep replies short."
            )
        )
    )

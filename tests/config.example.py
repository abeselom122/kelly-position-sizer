# Copy this file to config.py (NOT committed — it's in .gitignore) and fill
# in your real values from Exness's Personal Area API page.

API_KEY = "exnsk-live-your-key-here"

# The Ed25519 private key Exness gives you, as raw 32 bytes.
# If Exness gives it to you as base64:
#   import base64
#   PRIVATE_KEY_BYTES = base64.b64decode("your-base64-key-here")
# If it's PEM, load it with cryptography's serialization module and export
# raw bytes from that instead of pasting PEM text here directly.
PRIVATE_KEY_BYTES = b"\x00" * 32  # placeholder — replace with your real key

ACCOUNT_ID = "123456789"  # your Exness trading account number

# TODO: confirm the exact REST and WebSocket hostnames from your Exness API
# dashboard, or the OpenAPI spec at https://www.exness-api.com/exness_api.yaml
# — they weren't published anywhere this file's contents could confirm.
API_HOST = "https://REPLACE-WITH-YOUR-EXNESS-API-HOST"
WS_HOST = "wss://REPLACE-WITH-YOUR-EXNESS-WS-HOST"

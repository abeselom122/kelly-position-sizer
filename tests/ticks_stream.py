"""
Live tick stream over the Exness WebSocket API.

The WS connection handshake is signed exactly like a GET request (same
Ed25519 scheme as exness_client.py) — the signed `path` must match the WS
connection path exactly.

CONFIRMED from the docs: the path itself, and that ticks are throttled to
at most one update per 500ms per instrument.

NOT confirmed: whether the `["all"]` wildcard (confirmed for the
`instruments` subscription on the separate /ws/events stream) also works
for ticks, and the exact subscribe-message JSON shape below is a reasonable
guess from the unsubscribe description in the docs, not something I could
see directly. If subscribing with "all" gets rejected, pull the real
instrument list via ExnessClient.get_instruments() and subscribe to that
list instead.

Requires: pip install websockets
"""

import asyncio
import json
from typing import AsyncIterator, Iterable

import websockets

from exness_client import ExnessClient


class ExnessTicksStream:
    def __init__(self, client: ExnessClient, ws_host: str):
        """
        ws_host: the WebSocket host, e.g. "wss://<host>" — confirm this from
        your Exness API dashboard. It may or may not be the same host as
        the REST api_host.
        """
        self.client = client
        self.ws_host = ws_host.rstrip("/")

    async def stream_ticks(self, instruments: Iterable[str] = ("all",)) -> AsyncIterator[dict]:
        path = f"/v1/server-events/accounts/{self.client.account_id}/ws/ticks"
        headers = self.client.sign("GET", path, b"", "")
        uri = f"{self.ws_host}{path}"

        async with websockets.connect(uri, extra_headers=headers) as ws:
            # Newer `websockets` versions renamed this to `additional_headers`.
            # If you get a TypeError on `extra_headers`, swap it for that.
            subscribe_msg = {"event": "subscribe", "instruments": list(instruments)}
            await ws.send(json.dumps(subscribe_msg))

            async for raw in ws:
                yield json.loads(raw)


async def _demo():
    from config import ACCOUNT_ID, API_HOST, API_KEY, PRIVATE_KEY_BYTES, WS_HOST

    client = ExnessClient(API_KEY, PRIVATE_KEY_BYTES, ACCOUNT_ID, API_HOST)
    stream = ExnessTicksStream(client, WS_HOST)

    async for tick in stream.stream_ticks(["all"]):
        print(tick)


if __name__ == "__main__":
    asyncio.run(_demo())

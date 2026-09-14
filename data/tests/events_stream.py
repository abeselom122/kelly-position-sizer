"""
Live account/transaction event stream — this is how you find out a
position actually closed (and its realized PnL), separate from the raw
price ticks in ticks_stream.py.

CONFIRMED from the docs: the WS path below is a distinct stream from
ticks, signed the same Ed25519 way, covering transaction/account-state/
instrument-condition channels.

NOT CONFIRMED: the exact JSON field names inside a transaction-close
event (which key holds realized PnL, position id, etc.) — the docs
described event *types*, not a full payload example, in what I could
fetch. Print raw events the first time you run this against a real
account and adjust extract_close_pnl() to match what actually comes
through.

Requires: pip install websockets
"""

import json
from typing import AsyncIterator, Iterable, Optional

import websockets

from exness_client import ExnessClient


class ExnessEventsStream:
    def __init__(self, client: ExnessClient, ws_host: str):
        self.client = client
        self.ws_host = ws_host.rstrip("/")

    async def stream_events(
        self, channels: Iterable[str] = ("transactions", "account_state")
    ) -> AsyncIterator[dict]:
        path = f"/v1/server-events/accounts/{self.client.account_id}/ws/events"
        headers = self.client.sign("GET", path, b"", "")
        uri = f"{self.ws_host}{path}"

        async with websockets.connect(uri, extra_headers=headers) as ws:
            subscribe_msg = {"event": "subscribe", "channels": list(channels)}
            await ws.send(json.dumps(subscribe_msg))

            async for raw in ws:
                yield json.loads(raw)


def extract_close_pnl(event: dict) -> Optional[float]:
    """
    Best-effort extraction of realized PnL from a transaction-close event.
    UNVERIFIED field names — this checks a few plausible keys. Print raw
    events during your first live test and adjust to match reality.
    """
    for key in ("realized_pnl", "profit", "pnl", "realizedProfit"):
        if key in event:
            try:
                return float(event[key])
            except (TypeError, ValueError):
                pass
    return None

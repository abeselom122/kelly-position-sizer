"""
Exness Trader API client.

Implements the Ed25519 request-signing scheme from the official Exness
Public Trader API docs:
https://www.exness-api.com/reference/exness-trader-api

CONFIRMED from the official docs (safe to rely on):
  - the signing scheme itself (headers, EXN-DATA payload shape, Ed25519 sig)
  - these endpoint paths:
      GET  /v1/configuration/accounts/{account_id}/account
      GET  /v1/configuration/accounts/{account_id}/instruments
      GET  /v1/configuration/accounts/{account_id}/instruments/{instrument}/conditions
      GET  /v1/configuration/accounts/{account_id}/limits
      POST /v1/trading/accounts/{account_id}/positions   (open position)
      GET  /v1/trading/accounts/{account_id}/operations/{operation_id}
  - trading operations are ASYNC: a POST returns an ACK with operation_id;
    the real fill/rejection arrives later via the Server Events WebSocket
    stream (see ticks_stream.py), not in the POST response itself.

NOT confirmed (the request body schema didn't render in what I could fetch)
  - the exact field names/values for open_position's body below are my best
    reasonable guess (instrument/side/volume/stop_loss_price/take_profit_price).
    Verify against https://www.exness-api.com/reference/open-position in
    your own account (or the OpenAPI YAML at
    https://www.exness-api.com/exness_api.yaml) before trading real money.
  - the REST/WebSocket hostnames — not published anywhere I could reach.
    Get them from your Exness API dashboard after signup.

Requires: pip install requests cryptography
"""

import base64
import hashlib
import json
import time
import uuid
from typing import Any, Dict, Optional

import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _b64url_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class ExnessAPIError(Exception):
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self.payload = payload
        super().__init__(f"Exness API error {status_code}: {payload}")


class ExnessClient:
    """
    Signed REST client for the Exness Trader API.

    private_key_bytes: the raw 32-byte Ed25519 private key seed Exness gives
    you. If yours comes as base64, `base64.b64decode()` it first. If it's
    PEM, load it with `cryptography`'s serialization module and export raw
    bytes from that instead of passing PEM text in here.
    """

    def __init__(
        self,
        api_key: str,
        private_key_bytes: bytes,
        account_id: str,
        api_host: str,
        timeout: float = 10.0,
    ):
        self.api_key = api_key
        self.account_id = account_id
        self.api_host = api_host.rstrip("/")
        self.timeout = timeout
        self._private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        self._session = requests.Session()

    def sign(self, method: str, path: str, body_bytes: bytes, idempotency_key: str) -> Dict[str, str]:
        """
        Builds the EXN-* signed headers for one request. Exposed (not
        prefixed with _) because ticks_stream.py reuses it to sign the
        WebSocket handshake.
        """
        timestamp_ms = int(time.time() * 1000)
        body_hash = _b64url_no_pad(hashlib.sha256(body_bytes).digest())

        payload = {
            "api_key": self.api_key,
            "idempotency_key": idempotency_key,
            "timestamp": timestamp_ms,
            "sign_version": 1,
            "method": method.upper(),
            "path": path,
            "body_hash": body_hash,
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        data_b64 = _b64url_no_pad(payload_bytes)
        signature = self._private_key.sign(payload_bytes)
        sign_b64 = _b64url_no_pad(signature)

        return {
            "EXN-API-KEY": self.api_key,
            "EXN-IDEMPOTENCY-KEY": idempotency_key,
            "EXN-TIMESTAMP": str(timestamp_ms),
            "EXN-SIGN-VERSION": "1",
            "EXN-DATA": data_b64,
            "EXN-SIGN": sign_b64,
        }

    def _request(self, method: str, path: str, body: Optional[dict] = None, mutating: bool = False) -> Any:
        body_bytes = json.dumps(body, separators=(",", ":")).encode("utf-8") if body else b""
        # Signed GET requests must still send an empty EXN-IDEMPOTENCY-KEY header.
        idempotency_key = f"req-{uuid.uuid4().hex}" if mutating else ""

        headers = self.sign(method, path, body_bytes, idempotency_key)
        if body_bytes:
            headers["Content-Type"] = "application/json"

        url = f"{self.api_host}{path}"
        response = self._session.request(method, url, headers=headers, data=body_bytes or None, timeout=self.timeout)

        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            raise ExnessAPIError(response.status_code, payload)

        return response.json() if response.content else None

    # ---- Configuration ----

    def get_account_info(self) -> dict:
        return self._request("GET", f"/v1/configuration/accounts/{self.account_id}/account")

    def get_instruments(self) -> list:
        return self._request("GET", f"/v1/configuration/accounts/{self.account_id}/instruments")

    def get_instrument_condition(self, instrument: str) -> dict:
        path = f"/v1/configuration/accounts/{self.account_id}/instruments/{instrument}/conditions"
        return self._request("GET", path)

    def get_limits(self) -> dict:
        return self._request("GET", f"/v1/configuration/accounts/{self.account_id}/limits")

    # ---- Trading ----

    def open_position(
        self,
        instrument: str,
        side: str,
        volume: str,
        stop_loss_price: Optional[str] = None,
        take_profit_price: Optional[str] = None,
    ) -> dict:
        """
        Returns an ACK containing operation_id — NOT a confirmed fill.
        Listen on the Server Events WebSocket stream (ticks_stream.py's
        sibling, the /ws/events path) for the real transaction_event, or
        poll get_operation_status(operation_id).

        UNVERIFIED body schema — see module docstring. Confirm field names
        before using this with real money.
        """
        body = {"instrument": instrument, "side": side, "volume": volume}
        if stop_loss_price is not None:
            body["stop_loss_price"] = stop_loss_price
        if take_profit_price is not None:
            body["take_profit_price"] = take_profit_price
        return self._request(
            "POST", f"/v1/trading/accounts/{self.account_id}/positions", body=body, mutating=True
        )

    def get_operation_status(self, operation_id: str) -> dict:
        return self._request("GET", f"/v1/trading/accounts/{self.account_id}/operations/{operation_id}")


if __name__ == "__main__":
    # Self-test: proves the signing logic produces a valid Ed25519 signature,
    # without making any real network call. Run this after pasting the file
    # in to confirm nothing's broken before you plug in real credentials.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as _Gen

    test_key = _Gen.generate()
    test_key_bytes = test_key.private_bytes_raw()

    client = ExnessClient(
        api_key="test-key",
        private_key_bytes=test_key_bytes,
        account_id="123456789",
        api_host="https://example.invalid",
    )
    headers = client.sign("GET", "/v1/configuration/accounts/123456789/account", b"", "")

    payload_bytes = base64.urlsafe_b64decode(headers["EXN-DATA"] + "==")
    signature = base64.urlsafe_b64decode(headers["EXN-SIGN"] + "==")
    test_key.public_key().verify(signature, payload_bytes)  # raises if invalid

    print("Signing self-test passed.")
    print("Sample headers:", json.dumps({k: v for k, v in headers.items() if k != "EXN-API-KEY"}, indent=2))

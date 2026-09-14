"""
Crypto social-sentiment signal, built on LunarCrush's real REST API.
https://lunarcrush.com/developers/api/overview

CONFIRMED from LunarCrush's own docs:
  - Base URL: https://lunarcrush.com/api4
  - Auth: `Authorization: Bearer <API_KEY>` header
  - Example endpoint used below: /public/topic/<topic>/posts/v1 (top
    social posts for a topic)
  - Free "Hobby" plan: market-data endpoints only, 4 requests/min, 100/day.
    Social, creator, and AI endpoints — which Posts falls under — unlock
    on paid plans (Individual and up).

NOT FULLY CONFIRMED: I tested LunarCrush's data through its Claude
connector (not this raw REST endpoint) on a Hobby-tier account and got
back real post TEXT, with only follower/engagement numbers redacted.
Whether calling /public/topic/<topic>/posts/v1 directly with a free API
key behaves the same way, I haven't verified — test it yourself before
relying on it. If it's blocked outright, you'll need at least the
Individual plan.

CONFIRMED SEPARATELY, directly tested: LunarCrush's historical time-series
endpoints return every value blanked out on the Hobby tier — not
partially limited, fully unusable for backtesting. That means this
signal can be tried LIVE today, but can't be backtested against real
history without paying. Either upgrade, or start logging
get_recent_sentiment() output daily starting now and build your own
history going forward.

The scoring in _score_post_text is a naive keyword counter — a
placeholder to prove the pipeline works end to end, not a real sentiment
model. Swap it for something better (a proper NLP model, or an LLM call
reading the actual post text) once you've confirmed the API access works
for your plan.

Requires: pip install requests
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import requests

from backtest import SignalCall

LUNARCRUSH_BASE_URL = "https://lunarcrush.com/api4"

_POSITIVE_WORDS = {
    "bullish", "buy", "moon", "surge", "rally", "greed", "pump", "strong",
    "breakout", "gain", "bull", "up", "rising", "high",
}
_NEGATIVE_WORDS = {
    "bearish", "sell", "crash", "dump", "fear", "plunge", "weak",
    "breakdown", "loss", "bear", "down", "falling", "low", "outflow",
}


@dataclass
class SentimentReading:
    topic: str
    post_count: int
    positive_hits: int
    negative_hits: int
    score: float  # -1.0 (very negative) to +1.0 (very positive)


def fetch_recent_posts(topic: str, api_key: str) -> List[dict]:
    """
    Hits LunarCrush's real API. Raises requests.HTTPError on a 4xx/5xx —
    including, plausibly, a 402/403 if Posts turns out to be gated on
    your plan the same way historical time-series was.
    """
    url = f"{LUNARCRUSH_BASE_URL}/public/topic/{topic}/posts/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", [])


def _score_post_text(posts: List[dict], topic: str) -> SentimentReading:
    positive_hits = 0
    negative_hits = 0

    for post in posts:
        text = (post.get("text") or post.get("title") or "").lower()
        words = set(re.findall(r"[a-z']+", text))
        positive_hits += len(words & _POSITIVE_WORDS)
        negative_hits += len(words & _NEGATIVE_WORDS)

    total_hits = positive_hits + negative_hits
    score = (positive_hits - negative_hits) / total_hits if total_hits > 0 else 0.0

    return SentimentReading(
        topic=topic,
        post_count=len(posts),
        positive_hits=positive_hits,
        negative_hits=negative_hits,
        score=score,
    )


def get_recent_sentiment(topic: str, api_key: str) -> SentimentReading:
    posts = fetch_recent_posts(topic, api_key)
    return _score_post_text(posts, topic)


def sentiment_to_signal(
    reading: SentimentReading,
    current_price: float,
    stop_distance_pct: float = 0.03,
    reward_risk_ratio: float = 1.5,
    min_posts: int = 5,
    score_threshold: float = 0.15,
) -> Optional[SignalCall]:
    """
    Made-up mapping, same honesty caveat as the forex signal: this
    converts a keyword-count score into a probability by a formula I
    picked, not one anyone validated. That's what backtesting — once you
    can actually get historical data for it — is supposed to interrogate.
    """
    if reading.post_count < min_posts:
        return None  # not enough posts to trust a reading either way

    if reading.score >= score_threshold:
        direction = "buy"
    elif reading.score <= -score_threshold:
        direction = "sell"
    else:
        return None

    win_probability = min(0.70, 0.50 + abs(reading.score) * 0.20)
    stop_distance = current_price * stop_distance_pct

    return SignalCall(
        direction=direction,
        win_probability=win_probability,
        reward_risk_ratio=reward_risk_ratio,
        stop_distance=stop_distance,
    )


if __name__ == "__main__":
    # Can't hit the real network from wherever this demo runs without a
    # real API key, so this proves _score_post_text and sentiment_to_signal
    # work correctly using text shaped like what LunarCrush actually
    # returns (real phrasing, paraphrased here rather than reproduced).
    mock_posts = [
        {"text": "Bitcoin sentiment remains firmly bullish, greed index holding high, are we about to see another leg up"},
        {"text": "Bitcoin sentiment just hit its highest level since March, extreme greed zone, rallying hard"},
        {"text": "Bitcoin struggles near support as bond yields rise and rate hike fears loom, sentiment cooling"},
        {"text": "Bitcoin sentiment shifted from sell the rally to buy the dip as traders bet on a rebound"},
        {"text": "Bitcoin ETFs continue to see outflows, adding to market pressure, sentiment weak"},
        {"text": "Another all time high in bitcoin sentiment, feels like a strong pump is building"},
    ]

    reading = _score_post_text(mock_posts, topic="bitcoin")
    print("Mock sentiment reading:", reading)

    signal = sentiment_to_signal(reading, current_price=80000.0)
    print("Resulting signal:", signal)

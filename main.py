import os
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from pprint import pprint
from typing import Any

import anthropic
import requests
import yfinance as yf
from apify_client import ApifyClient
from dotenv import load_dotenv

# Load variables from local .env file.
ENV_PATH = Path(__file__).resolve().parent / ".env"
HISTORY_PATH = Path(__file__).resolve().parent / "stocks_history.json"
load_dotenv(dotenv_path=ENV_PATH, override=True, encoding="utf-8")

# Ensure Unicode console output works on Windows terminals (emoji, symbols).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def fetch_tweets() -> list[dict[str, Any]]:
    """
    Fetch recent tweets for the configured accounts and return normalized output.
    """
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_API_TOKEN in environment variables.")

    client = ApifyClient(token)

    handles = ["aleabitoreddit", "zephyr_z9", "Julianpetroulas"]
    per_account_limit = 5
    search_query = " OR ".join([f"from:{handle}" for handle in handles])

    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=24)

    run_input = {
        "searchQueries": [search_query],
        "searchTerms": [search_query],
        "queries": [search_query],
        "query": search_query,
        "type": "Latest",
        "searchType": "Latest",
        "sort": "Latest",
        "tweetsDesired": per_account_limit,
        "maxItems": per_account_limit,
        "maxTweets": per_account_limit,
        "since": since_utc.isoformat(),
        "until": now_utc.isoformat(),
    }

    print("Trying actor: shanes/tweet-flash")
    run = client.actor("shanes/tweet-flash").call(run_input=run_input)
    dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

    if not dataset_items:
        print("No items from shanes/tweet-flash, trying fallback actor.")
        fallback_input = {
            "searchQueries": [search_query],
            "searchTerms": [search_query],
            "searches": [search_query],
            "query": search_query,
            "searchType": "Latest",
            "sort": "Latest",
            "sortType": "Latest",
            "mode": "Latest",
            "maxItems": per_account_limit,
            "tweetsDesired": per_account_limit,
            "since": since_utc.isoformat(),
            "until": now_utc.isoformat(),
        }
        fallback_run = client.actor("khadinakbar/x-tweet-scraper").call(
            run_input=fallback_input
        )
        dataset_items = list(client.dataset(fallback_run["defaultDatasetId"]).iterate_items())

    if dataset_items:
        print(f"DEBUG RAW DATA: {dataset_items[0]}")
        print(f"DEBUG FIRST ITEM KEYS: {list(dataset_items[0].keys())}")

    tweets: list[dict[str, Any]] = []
    for item in dataset_items:
        author_data = item.get("author") or item.get("user") or {}
        author_name: Any = None
        if isinstance(author_data, dict):
            author_name = (
                author_data.get("userName")
                or author_data.get("username")
                or author_data.get("screen_name")
            )
        elif isinstance(author_data, str):
            author_name = author_data

        tweets.append(
            {
                "full_text": item.get("full_text") or item.get("text") or item.get("fullText"),
                "author": author_name
                or item.get("author_handle")
                or item.get("author_name")
                or item.get("authorName")
                or item.get("screen_name")
                or item.get("username"),
                "created_at": item.get("created_at") or item.get("createdAt"),
            }
        )

    return tweets


def analyze_tweets_with_claude(tweets_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Analyze tweets with Claude and return ticker sentiment data.
    """
    system_prompt = (
        "You are a professional stock analyst. Extract stock tickers (e.g. $AAPL) "
        "from the tweets, determine the sentiment (Bullish, Bearish, or Neutral), "
        "and provide a 1-sentence reasoning for each."
    )

    user_prompt = (
        "Analyze these tweets and return ONLY a valid JSON list of dictionaries. "
        "Each dictionary must contain exactly these keys: ticker, sentiment, reasoning. "
        "Do not include markdown, explanations, or any extra text.\n\n"
        f"Tweets data:\n{json.dumps(tweets_data, ensure_ascii=False)}"
    )

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if (not anthropic_api_key) or (anthropic_api_key == "твоят_нов_токен_тук"):
        print("⚠️ ANTHROPIC_API_KEY not found. Running in MOCK MODE.")
        time.sleep(1.5)
        return [
            {
                "ticker": "ARM",
                "sentiment": "Bullish",
                "reasoning": "Strong earnings growth and AI demand.",
            }
        ]

    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1200,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

    try:
        raw_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        parsed = json.loads(raw_text)
    except Exception as exc:
        raise ValueError(f"Failed to parse Claude response as JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError("Claude response must be a JSON list.")

    required_keys = {"ticker", "sentiment", "reasoning"}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each item in Claude response must be a dictionary.")
        if set(item.keys()) != required_keys:
            raise ValueError(
                "Each dictionary must contain exactly these keys: "
                "ticker, sentiment, reasoning."
            )

    return parsed


def get_current_prices(tickers_list: list[str]) -> dict[str, float | None]:
    """
    Fetch current market prices for a list of stock tickers.
    Returns a dictionary with ticker -> current price (or None if unavailable).
    """
    prices: dict[str, float | None] = {}

    for ticker in tickers_list:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            continue

        try:
            ticker_data = yf.Ticker(normalized_ticker)
            info = ticker_data.fast_info

            current_price: Any = None
            if info:
                current_price = info.get("lastPrice") or info.get("regularMarketPrice")

            if current_price is None:
                history = ticker_data.history(period="1d")
                if not history.empty and "Close" in history:
                    close_price = history["Close"].iloc[-1]
                    current_price = float(close_price) if close_price is not None else None

            prices[normalized_ticker] = (
                float(current_price) if current_price is not None else None
            )
        except Exception:
            prices[normalized_ticker] = None

    return prices


def send_telegram_notification(text: str) -> bool:
    """
    Send a Telegram message using bot token and chat ID from environment variables.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token or not chat_id:
        print(
            "Telegram notification failed: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env."
        )
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    try:
        response = requests.post(url, json=payload, timeout=15)
        response_data = response.json()
    except requests.RequestException as exc:
        print(f"Telegram notification failed: network/API request error: {exc}")
        return False
    except ValueError:
        print("Telegram notification failed: could not parse Telegram response as JSON.")
        return False

    if response.status_code == 200 and response_data.get("ok"):
        print("Telegram notification sent successfully.")
        return True

    error_description = response_data.get("description", "Unknown Telegram API error.")
    print(
        f"Telegram notification failed: {error_description} "
        f"(HTTP {response.status_code})."
    )
    return False


def _load_history() -> dict[str, dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return {}

    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _write_history(history_data: dict[str, dict[str, Any]]) -> None:
    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history_data, file, ensure_ascii=False, indent=2)


def save_to_history(ticker: str, price: float | None, sentiment: str) -> None:
    """
    Save or update ticker data in stocks_history.json.
    """
    if price is None:
        return

    normalized_ticker = ticker.replace("$", "").strip().upper()
    if not normalized_ticker:
        return

    history_data = _load_history()
    timestamp = datetime.now(timezone.utc).isoformat()

    if normalized_ticker not in history_data:
        history_data[normalized_ticker] = {
            "initial_price": float(price),
            "current_price": float(price),
            "sentiment": sentiment,
            "timestamp": timestamp,
        }
    else:
        history_data[normalized_ticker]["current_price"] = float(price)
        history_data[normalized_ticker]["sentiment"] = sentiment

    _write_history(history_data)


def get_performance_report() -> list[dict[str, Any]]:
    """
    Calculate percentage change for all tracked stocks in history.
    """
    history_data = _load_history()
    report: list[dict[str, Any]] = []

    for ticker, values in history_data.items():
        initial_price = values.get("initial_price")
        current_price = values.get("current_price")
        sentiment = values.get("sentiment", "Neutral")
        timestamp = values.get("timestamp")

        if (
            not isinstance(initial_price, (int, float))
            or not isinstance(current_price, (int, float))
            or initial_price == 0
        ):
            continue

        change_pct = ((current_price - initial_price) / initial_price) * 100
        report.append(
            {
                "ticker": ticker,
                "initial_price": float(initial_price),
                "current_price": float(current_price),
                "change_pct": change_pct,
                "sentiment": sentiment,
                "timestamp": timestamp,
            }
        )

    report.sort(key=lambda item: item["ticker"])
    return report


def main() -> None:
    tweets = fetch_tweets()
    if not tweets:
        print("No tweets found from source. Using mock tweet for QA pipeline run.")
        tweets = [
            {
                "full_text": "ARM looks strong with AI demand and momentum.",
                "author": "mock_source",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

    analysis = analyze_tweets_with_claude(tweets)
    tickers = [
        str(item.get("ticker", "")).replace("$", "").strip().upper()
        for item in analysis
        if str(item.get("ticker", "")).strip()
    ]
    prices = get_current_prices(tickers)

    for item in analysis:
        ticker = str(item.get("ticker", "")).strip()
        normalized_ticker = ticker.replace("$", "").strip().upper()
        sentiment = str(item.get("sentiment", "Neutral")).strip()
        current_price = prices.get(normalized_ticker)
        save_to_history(normalized_ticker, current_price, sentiment)

    performance = get_performance_report()

    lines = ["🏦 MARKET INSIGHT REPORT 🏦", ""]
    for item in analysis:
        ticker = str(item.get("ticker", "")).replace("$", "").strip().upper()
        sentiment = str(item.get("sentiment", "Neutral")).strip()
        reasoning = str(item.get("reasoning", "")).strip()
        current_price = prices.get(ticker)
        price_text = f"${current_price:.2f}" if current_price is not None else "N/A"

        perf_match = next((entry for entry in performance if entry["ticker"] == ticker), None)
        if perf_match is not None:
            change_value = perf_match["change_pct"]
            trend_emoji = "🟢" if change_value >= 0 else "🔴"
            change_text = f"{trend_emoji} {change_value:+.2f}%"
        else:
            change_text = "⚪ N/A"

        lines.extend(
            [
                f"🚀 ${ticker} | {sentiment}",
                f"💰 Price: {price_text}",
                f"📈 Trend since first discovery: {change_text}",
                f"🧠 AI Analysis: {reasoning}",
                "Generated by AlphaFlow AI",
                "",
            ]
        )

    report_text = "\n".join(lines)
    sent_ok = send_telegram_notification(report_text)
    if sent_ok:
        print("Test Successful")
    else:
        print("Test Failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline failed: {exc}")

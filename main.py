import html
import json
import logging
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
import requests
import yfinance as yf
from apify_client import ApifyClient
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
HISTORY_PATH = Path(__file__).resolve().parent / "stocks_history.json"
LOG_PATH = Path(__file__).resolve().parent / "production.log"

load_dotenv(dotenv_path=ENV_PATH, override=True, encoding="utf-8")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

MAX_TELEGRAM_MESSAGE_LENGTH = 4096
APIFY_TIMEOUT_SECONDS = int(os.getenv("APIFY_TIMEOUT_SECONDS", "180"))
PRICE_FETCH_RETRIES = 2


def _parse_target_accounts() -> list[str]:
    """
    Parse TARGET_ACCOUNTS from .env as a comma-separated list.
    """
    raw_value = os.getenv("TARGET_ACCOUNTS", "").strip()
    if raw_value:
        parsed = [handle.strip().lstrip("@") for handle in raw_value.split(",")]
        unique_handles: list[str] = []
        for handle in parsed:
            if handle and handle not in unique_handles:
                unique_handles.append(handle)
        if unique_handles:
            return unique_handles

    return ["aleabitoreddit", "zephyr_z9", "Julianpetroulas"]


def _normalize_handle(value: str) -> str:
    return value.strip().lstrip("@").lower()


def _is_mock_mode_enabled() -> bool:
    """
    Resolve mock mode from .env.
    If ANTHROPIC_API_KEY exists, force real API mode.
    """
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_api_key:
        return False

    mock_raw = os.getenv("MOCK_MODE", "true").strip().lower()
    return mock_raw in {"1", "true", "yes", "on"}


def _validate_environment() -> None:
    """
    Validate required environment variables and fail fast with clear feedback.
    """
    errors: list[str] = []
    required_keys = [
        "APIFY_API_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TARGET_ACCOUNTS",
    ]
    for key in required_keys:
        value = os.getenv(key, "").strip()
        if not value:
            errors.append(f"{key} is missing or empty.")

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if bot_token and ":" not in bot_token:
        errors.append("TELEGRAM_BOT_TOKEN looks malformed (expected '<id>:<token>').")

    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if chat_id and not re.fullmatch(r"-?\d+", chat_id):
        errors.append("TELEGRAM_CHAT_ID must be numeric.")

    apify_token = os.getenv("APIFY_API_TOKEN", "").strip()
    if apify_token and len(apify_token) < 20:
        errors.append("APIFY_API_TOKEN looks too short / malformed.")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not anthropic_key and not _is_mock_mode_enabled():
        errors.append("ANTHROPIC_API_KEY is required when MOCK_MODE is disabled.")
    if anthropic_key and not anthropic_key.startswith("sk-ant-"):
        errors.append("ANTHROPIC_API_KEY looks malformed (expected prefix 'sk-ant-').")

    target_accounts = _parse_target_accounts()
    if len(target_accounts) != 3:
        errors.append("TARGET_ACCOUNTS must contain exactly 3 handles.")

    if errors:
        logger.error("Startup validation failed:")
        for err in errors:
            logger.error(" - %s", err)
        raise SystemExit(1)


def _get_last_known_price(ticker: str) -> float | None:
    """
    Return the last saved current price for ticker from history.
    """
    normalized_ticker = ticker.replace("$", "").strip().upper()
    if not normalized_ticker:
        return None

    history_data = _load_history()
    ticker_data = history_data.get(normalized_ticker, {})
    previous_price = ticker_data.get("current_price")
    return float(previous_price) if isinstance(previous_price, (int, float)) else None


def _run_with_timeout(func: Any, timeout_seconds: int, description: str) -> Any:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"{description} timed out after {timeout_seconds} seconds."
            ) from exc


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _extract_json_array(raw_text: str) -> list[dict[str, str]]:
    """
    Extract first JSON array from Claude output, even with extra text.
    """
    stripped = raw_text.strip()
    candidates = [stripped]

    fenced_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", stripped, re.IGNORECASE)
    if fenced_match:
        candidates.append(fenced_match.group(1))

    array_match = re.search(r"(\[[\s\S]*\])", stripped)
    if array_match:
        candidates.append(array_match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Unable to extract valid JSON array from Claude response.")


def _fallback_neutral_analysis(tweets_data: list[dict[str, Any]], reason: str) -> list[dict[str, str]]:
    """
    Create safe neutral analysis if Claude refuses or fails.
    """
    logger.info("Using neutral fallback analysis: %s", reason)
    fallback: list[dict[str, str]] = []
    seen: set[str] = set()
    ticker_pattern = re.compile(r"\$([A-Za-z]{1,5})\b")
    for tweet in tweets_data:
        text = str(tweet.get("full_text") or "")
        for symbol in ticker_pattern.findall(text):
            ticker = symbol.upper()
            if ticker not in seen:
                seen.add(ticker)
                fallback.append(
                    {
                        "ticker": ticker,
                        "sentiment": "Neutral",
                        "reasoning": "Automated neutral fallback due to unavailable AI analysis.",
                    }
                )

    if fallback:
        return fallback
    return [
        {
            "ticker": "SPY",
            "sentiment": "Neutral",
            "reasoning": "No explicit ticker extracted; neutral fallback applied.",
        }
    ]


def _escape_html_telegram(text: str) -> str:
    return html.escape(text, quote=False)


def _truncate_telegram_message(text: str, max_length: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    suffix = "\n\n... (truncated)"
    return text[: max_length - len(suffix)] + suffix


def _calculate_change_pct(initial_price: float, current_price: float) -> float:
    return ((current_price - initial_price) / initial_price) * 100.0

def fetch_tweets() -> list[dict[str, Any]]:
    """
    Fetch recent tweets for the configured accounts and return normalized output.
    """
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_API_TOKEN in environment variables.")

    client = ApifyClient(token)

    handles = _parse_target_accounts()
    per_account_limit = int(os.getenv("TWEETS_PER_ACCOUNT", "5"))
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

    logger.info("Starting scrape from configured X accounts.")
    run = _run_with_timeout(
        lambda: client.actor("shanes/tweet-flash").call(run_input=run_input),
        APIFY_TIMEOUT_SECONDS,
        "Apify actor shanes/tweet-flash call",
    )
    dataset_items = _run_with_timeout(
        lambda: list(client.dataset(run["defaultDatasetId"]).iterate_items()),
        APIFY_TIMEOUT_SECONDS,
        "Apify dataset iteration",
    )

    if not dataset_items:
        logger.info("Primary Apify actor returned 0 items. Trying fallback actor.")
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
        fallback_run = _run_with_timeout(
            lambda: client.actor("khadinakbar/x-tweet-scraper").call(
                run_input=fallback_input
            ),
            APIFY_TIMEOUT_SECONDS,
            "Apify actor khadinakbar/x-tweet-scraper call",
        )
        dataset_items = _run_with_timeout(
            lambda: list(client.dataset(fallback_run["defaultDatasetId"]).iterate_items()),
            APIFY_TIMEOUT_SECONDS,
            "Apify fallback dataset iteration",
        )

    tweets: list[dict[str, Any]] = []
    allowed_handles = {_normalize_handle(handle) for handle in handles}
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

        resolved_author = (
            author_name
            or item.get("author_handle")
            or item.get("author_name")
            or item.get("authorName")
            or item.get("screen_name")
            or item.get("username")
        )
        if resolved_author and _normalize_handle(str(resolved_author)) not in allowed_handles:
            continue

        tweets.append(
            {
                "full_text": item.get("full_text") or item.get("text") or item.get("fullText"),
                "author": resolved_author,
                "created_at": item.get("created_at") or item.get("createdAt"),
            }
        )

    logger.info("Scrape completed with %d normalized tweets.", len(tweets))
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
    if _is_mock_mode_enabled():
        logger.warning("MOCK_MODE enabled; skipping Claude API and returning mock sentiment.")
        return [
            {
                "ticker": "ARM",
                "sentiment": "Bullish",
                "reasoning": "Strong earnings growth and AI demand.",
            }
        ]
    if not anthropic_api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY in .env while MOCK_MODE is disabled.")

    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1200,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        return _fallback_neutral_analysis(tweets_data, f"Claude API unavailable: {exc}")

    try:
        raw_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        parsed = _extract_json_array(raw_text)
    except Exception as exc:
        return _fallback_neutral_analysis(tweets_data, f"Claude parse issue: {exc}")

    if not isinstance(parsed, list):
        raise ValueError("Claude response must be a JSON list.")

    required_keys = {"ticker", "sentiment", "reasoning"}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each item in Claude response must be a dictionary.")
        if set(item.keys()) != required_keys:
            return _fallback_neutral_analysis(
                tweets_data,
                "Claude response schema mismatch; neutral fallback used.",
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

        attempts = PRICE_FETCH_RETRIES + 1
        current_price: float | None = None

        for attempt in range(1, attempts + 1):
            try:
                ticker_data = yf.Ticker(normalized_ticker)
                info = ticker_data.fast_info
                if info:
                    current_price = _safe_float(
                        info.get("lastPrice") or info.get("regularMarketPrice")
                    )

                if current_price is None:
                    history = ticker_data.history(period="1d")
                    if not history.empty and "Close" in history:
                        current_price = _safe_float(history["Close"].iloc[-1])

                if current_price is not None:
                    break
            except Exception as exc:
                logger.warning(
                    "Price fetch failed for %s (attempt %d/%d): %s",
                    normalized_ticker,
                    attempt,
                    attempts,
                    exc,
                )

        if current_price is None:
            logger.info("No valid market price available for %s.", normalized_ticker)
        prices[normalized_ticker] = current_price

    return prices


def send_telegram_notification(text: str) -> bool:
    """
    Send a Telegram message using bot token and chat ID from environment variables.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token or not chat_id:
        logger.error("Telegram send skipped: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    safe_text = _truncate_telegram_message(_escape_html_telegram(text))
    payload = {"chat_id": chat_id, "text": safe_text, "parse_mode": "HTML"}

    try:
        response = requests.post(url, json=payload, timeout=15)
        response_data = response.json()
    except requests.RequestException as exc:
        logger.error("Telegram notification failed: network/API request error: %s", exc)
        return False
    except ValueError:
        logger.error("Telegram notification failed: could not parse response as JSON.")
        return False

    if response.status_code == 200 and response_data.get("ok"):
        logger.info("Telegram notification sent successfully.")
        return True

    error_description = response_data.get("description", "Unknown Telegram API error.")
    logger.error(
        "Telegram notification failed: %s (HTTP %s).",
        error_description,
        response.status_code,
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
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(HISTORY_PATH.parent),
        delete=False,
    ) as temp_file:
        json.dump(history_data, temp_file, ensure_ascii=False, indent=2)
        temp_path = Path(temp_file.name)
    temp_path.replace(HISTORY_PATH)


def save_to_history(ticker: str, price: float | None, sentiment: str) -> None:
    """
    Save or update ticker data in stocks_history.json.
    """
    normalized_ticker = ticker.replace("$", "").strip().upper()
    if not normalized_ticker:
        return

    history_data = _load_history()
    timestamp = datetime.now(timezone.utc).isoformat()

    if normalized_ticker not in history_data:
        if price is None:
            return
        history_data[normalized_ticker] = {
            "initial_price": float(price),
            "current_price": float(price),
            "sentiment": sentiment,
            "timestamp": timestamp,  # first seen
            "last_updated": timestamp,
        }
    else:
        if price is not None:
            history_data[normalized_ticker]["current_price"] = float(price)
        history_data[normalized_ticker]["sentiment"] = sentiment
        history_data[normalized_ticker]["last_updated"] = timestamp

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
        last_updated = values.get("last_updated")

        if (
            not isinstance(initial_price, (int, float))
            or not isinstance(current_price, (int, float))
            or initial_price == 0
        ):
            continue

        change_pct = _calculate_change_pct(float(initial_price), float(current_price))
        report.append(
            {
                "ticker": ticker,
                "initial_price": float(initial_price),
                "current_price": float(current_price),
                "change_pct": change_pct,
                "sentiment": sentiment,
                "timestamp": timestamp,
                "last_updated": last_updated,
            }
        )

    report.sort(key=lambda item: item["ticker"])
    return report


def main() -> None:
    try:
        _validate_environment()
        run_timestamp = datetime.now(timezone.utc)
        target_accounts = _parse_target_accounts()
        tweets = fetch_tweets()
        if not tweets:
            logger.info("No new data from X accounts. Skipping this run.")
            return

        analysis = analyze_tweets_with_claude(tweets)
        tickers = sorted(
            {
            str(item.get("ticker", "")).replace("$", "").strip().upper()
            for item in analysis
            if str(item.get("ticker", "")).strip()
            }
        )
        prices = get_current_prices(tickers)

        for item in analysis:
            ticker = str(item.get("ticker", "")).strip()
            normalized_ticker = ticker.replace("$", "").strip().upper()
            sentiment = str(item.get("sentiment", "Neutral")).strip()
            market_price = prices.get(normalized_ticker)
            # If market is closed/unavailable, keep the previous tracked price.
            current_price = market_price if market_price is not None else _get_last_known_price(normalized_ticker)
            prices[normalized_ticker] = current_price
            save_to_history(normalized_ticker, current_price, sentiment)

        performance = get_performance_report()

        timestamp_label = run_timestamp.strftime("%Y-%m-%d %H:%M UTC")
        accounts_label = ", ".join([f"@{handle}" for handle in target_accounts])
        lines = [
            "🏦 MARKET INSIGHT REPORT 🏦",
            f"🕒 Generated: {timestamp_label}",
            f"🐦 Sources: {accounts_label}",
            f"📊 Tweets analyzed: {len(tweets)}",
            "",
        ]
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
            logger.info("Pipeline run completed successfully.")
            logger.info("✅ PIPELINE TEST SUCCESSFUL - MESSAGE SENT")
        else:
            logger.error("Pipeline run completed, but Telegram delivery failed.")
    except Exception as exc:
        logger.exception("Pipeline failed unexpectedly: %s", exc)


if __name__ == "__main__":
    main()

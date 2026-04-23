import os
from datetime import datetime, timedelta, timezone
from pprint import pprint
from typing import Any

from apify_client import ApifyClient
from dotenv import load_dotenv

# Load variables from local .env file.
load_dotenv()

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


if __name__ == "__main__":
    print("Script has started.")

    try:
        result = fetch_tweets()
        if result:
            print("Tweets found:")
            pprint(result, sort_dicts=False)
        else:
            print("No tweets found for the last 24h")
    except Exception as exc:
        print(f"Failed to fetch tweets: {exc}")

# 🚀 AlphaFlow AI: X-to-Signal Intelligence

A high-frequency sentiment analysis engine that converts social media noise into actionable market signals.

## 🛠️ Core Features (Day 1 Progress)

- 🐦 **X (Twitter) Scraper:** High-speed extraction via Apify.
- 🤖 **Intelligence Layer:** Advanced sentiment analysis using Claude 3.5 Sonnet (with automatic Mock-mode fallback).
- 💹 **Market Integration:** Real-time price fetching via Yahoo Finance (`yfinance`).
- 📂 **Smart Persistence:** JSON-based history tracking to monitor performance over time.
- 📱 **Telegram Delivery:** Instant professional reports sent directly to the user.

## 💎 Pro Features

- 🤖 **Daily Automation:** GitHub Actions runs the pipeline every day at 08:00 UTC (plus manual trigger support).
- 🗄️ **Self-Persisting Data:** No external database is required. Your GitHub repository acts as the persistent data layer by automatically saving updates to `stocks_history.json`.
- 🔁 **Auto-Commit History:** When data changes, the workflow commits and pushes updates back to `main` automatically.

### 🔑 Setting up GitHub Secrets

Add these 5 repository secrets in GitHub:
`Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

- 🧩 `APIFY_API_TOKEN` - API token for Apify scraping access.
- 🤖 `ANTHROPIC_API_KEY` - Claude API key for sentiment analysis.
- 📬 `TELEGRAM_BOT_TOKEN` - Telegram bot token used to send messages.
- 🆔 `TELEGRAM_CHAT_ID` - Target Telegram chat/user ID that receives reports.
- 🐦 `TARGET_ACCOUNTS` - Comma-separated list of exactly 3 X handles (example: `aleabitoreddit,zephyr_z9,Julianpetroulas`).

## ⚙️ Tech Stack

**Python 3.11 | Anthropic API | Apify | yfinance | Requests**

## ▶️ How to Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Configure `.env` with your production credentials:
   - `APIFY_API_TOKEN`
   - `ANTHROPIC_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `TARGET_ACCOUNTS` (exactly 3 X handles, comma-separated)
   - Optional: `MOCK_MODE`, `APIFY_TIMEOUT_SECONDS`, `TWEETS_PER_ACCOUNT`
3. Run the bot:
   - `python main.py`
4. Review outputs:
   - Runtime logs: `production.log`
   - Performance history: `stocks_history.json`

## ⚠️ Disclaimer

This software is a data processing and reporting tool. It is not a financial advice engine, does not provide investment recommendations, and should not be used as the sole basis for trading or portfolio decisions.

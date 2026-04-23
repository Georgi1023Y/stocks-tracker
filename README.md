# William Stock Bot

Automated investment research pipeline using Apify for X scraping, Claude 3.5 Sonnet for analysis, and Telegram for delivery.

## Tech Stack

- Python 3.10+
- Apify (`apify-client`) for X/Twitter data collection
- Claude 3.5 Sonnet (`anthropic`) for market and sentiment analysis
- `yfinance` for market data enrichment
- Telegram Bot API for report delivery
- `python-dotenv` for environment variable management

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Georgi1023Y/William-Stock-Bot.git
   cd William-Stock-Bot
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file and add your secrets:
   ```env
   APIFY_API_TOKEN=your_apify_token
   ANTHROPIC_API_KEY=your_anthropic_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```
5. Run the project:
   ```bash
   python main.py
   ```

## Project Roadmap

- Improve tweet ingestion reliability and add multi-actor fallback strategy.
- Add robust data validation and deduplication before analysis.
- Implement Claude-based scoring for sentiment, risk, and catalyst classification.
- Integrate `yfinance` for price context, performance snapshots, and ticker metadata.
- Build scheduled Telegram delivery with retries, alert thresholds, and summaries.
- Add tests, structured logging, and CI checks for production readiness.

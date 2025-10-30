# EMA Fib Telegram Bot

This is a Python trading bot that uses EMA and Fibonacci confluence for Ethereum (ETH/USDT) and sends signals and news to Telegram. News sentiment is enhanced by Google Gemini AI.

## Features
- EMA 21/34/40/50/200 and Fibonacci 0.54, 0.67 zone detection
- Buy/sell/confluence alerts
- Crypto news and AI-powered news analysis
- Telegram notifications
- Works with Binance exchange via CCXT

## Setup

1. **Clone or Upload files to GitHub**
2. **Install dependencies:**
   ```
pip install -r requirements.txt
   ```
3. **Set Environment Variables** (strongly recommended):
   - `BINANCE_API_KEY` – Your Binance API key
   - `BINANCE_API_SECRET` – Your Binance API secret
   - `TELEGRAM_BOT_TOKEN` – Your Telegram bot token
   - `TELEGRAM_CHAT_ID` – Your Telegram chat ID
   - `GEMINI_API_KEY` – (Optional) Google Gemini API Key for news analysis

4. **Run locally:**
   ```
python ema_fib_telegram_bot.py
   ```

## Deploy on Railway

1. **Create a GitHub repository** and push all files (`ema_fib_telegram_bot.py`, `requirements.txt`, `Procfile`, etc.)
2. Go to [Railway](https://railway.app) > New Project > Deploy from GitHub
3. Set the following variables in Railway’s "Variables" dashboard:
   - `BINANCE_API_KEY` 
   - `BINANCE_API_SECRET` 
   - `TELEGRAM_BOT_TOKEN` 
   - `TELEGRAM_CHAT_ID` 
   - `GEMINI_API_KEY` (optional)
4. Deploy the project. Railway will use the `Procfile` to run the bot automatically.

### Notes
- Keep your API keys and tokens secure! Use Railway’s environment variable system.
- By default, the bot trades on ETH/USDT 1hr timeframe - you can adjust in the code.
- For changes or support, edit code or open issues in your GitHub repo.

---

**Disclaimer:** This bot is for educational purposes and not financial advice. Trade carefully!

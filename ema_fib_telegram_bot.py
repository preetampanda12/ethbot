import ccxt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime
import json

# ============= CONFIGURATION =============
# Binance API Configuration
BINANCE_API_KEY = 'WSPoE9L04b8xukvqh7li4WT1TEqNnGob4ZCanZ4p2lTy2AbjYxzjV8q9lPcARRNI'
BINANCE_API_SECRET = 'J4PoDOoyBhpMTKIBBHGa1eCz8gNG19N85yTxfwFGG5tNcPUVgbrwKA51lox451Xg'

# Telegram Configuration
TELEGRAM_BOT_TOKEN = '8459911523:AAHXBnaNywpvSNU59MhD6G0RqJn993VKhUU'
TELEGRAM_CHAT_ID = '6080078099'

# Google Gemini AI Configuration (for news analysis)
GEMINI_API_KEY = 'AIzaSyCg8X9b-TskQhsDfigU-Q71r5jq-U9G2kk'  # Get from https://aistudio.google.com/app/apikey

# Trading Configuration
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'
LIMIT = 250  # Number of candles to fetch

# EMA Periods
EMA_21 = 21
EMA_34 = 34
EMA_40 = 40
EMA_50 = 50
EMA_200 = 200

# Fibonacci Settings
FIB_LOOKBACK = 100
FIB_LEVEL_1 = 0.54
FIB_LEVEL_2 = 0.67
MIN_EMAS_IN_ZONE = 4

# News Configuration
SEND_NEWS_ON_CANDLE_CLOSE = True  # Send news when new 1hr candle completes
SEND_NEWS_WITH_SIGNALS = True  # Send news summary with trading signals
NEWS_EVERY_N_CANDLES = 1  # Send news every N candles (1 = every hour, 4 = every 4 hours)

# ============= TELEGRAM FUNCTIONS =============
def send_telegram_message(message):
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload, timeout=10)
        try:
            print("[Telegram API response]", response.status_code, response.json())
        except Exception as log_err:
            print("[Telegram API log error]", log_err)
        return response.json()
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return None

# ============= NEWS & AI FUNCTIONS =============
def fetch_crypto_news():
    """Fetch latest Ethereum news from CryptoPanic API"""
    try:
        # Using CryptoPanic free API (no key required for basic usage)
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': 'free',  # Use 'free' or get your own key from cryptopanic.com
            'currencies': 'ETH',
            'kind': 'news',
            'filter': 'hot'
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])[:10]  # Get top 10 news items
        return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_news_with_gemini(news_items, current_price=None, trend=None):
    """Use Google Gemini AI to deeply analyze and summarize top 3 Ethereum news"""
    try:
        if not news_items:
            return None
        
        # Prepare detailed news data for AI
        news_text = "\n\n".join([
            f"Article {idx + 1}:\n"
            f"Title: {item.get('title', 'N/A')}\n"
            f"Source: {item.get('source', {}).get('title', 'Unknown')}\n"
            f"Published: {item.get('published_at', 'N/A')}\n"
            f"Votes: {item.get('votes', {}).get('positive', 0)} positive, {item.get('votes', {}).get('negative', 0)} negative"
            for idx, item in enumerate(news_items)
        ])
        
        # Add market context if available
        market_context = ""
        if current_price and trend:
            market_context = f"\nCurrent Market Context:\n- ETH Price: ${current_price:.2f}\n- Technical Trend: {trend}\n"
        
        # Create comprehensive analysis prompt
        prompt = f"""You are a professional cryptocurrency analyst specializing in Ethereum. Analyze these Ethereum news articles with deep market insight and provide a comprehensive summary.
{market_context}
News Articles to Analyze:
{news_text}

Please provide a thorough analysis with the following structure:

1. **TOP 3 MOST IMPACTFUL NEWS** (ranked by importance):
   For each news item, provide:
   - Clear headline/title
   - Detailed summary (3-4 sentences explaining what happened and why it matters)
   - Market Impact Assessment: BULLISH 🟢 / BEARISH 🔴 / NEUTRAL 🟡
   - Impact Magnitude: HIGH / MEDIUM / LOW
   - Time Horizon: SHORT-TERM (hours-days) / MEDIUM-TERM (weeks) / LONG-TERM (months+)
   - Key takeaway for traders

2. **OVERALL MARKET SENTIMENT**:
   - Current sentiment (Bullish/Bearish/Mixed/Uncertain)
   - Confidence level (High/Medium/Low)
   - Brief reasoning (2-3 sentences)

3. **TRADING IMPLICATIONS**:
   - Should traders watch for specific levels or events?
   - Any immediate risks or opportunities?
   - Confluence with technical analysis (if applicable based on current trend)

4. **KEY RISK FACTORS**:
   - What could change the narrative?
   - Any contradicting signals?

Be specific, analytical, and avoid generic statements. Focus on actionable insights that help traders make informed decisions. If news is not significant, say so clearly.

Format your response with clear sections and bullet points for readability."""

        # Google Gemini API Call
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            'contents': [{
                'parts': [{
                    'text': prompt
                }]
            }],
            'generationConfig': {
                'temperature': 0.3,
                'maxOutputTokens': 2048,
                'topP': 0.8,
                'topK': 10
            }
        }
        
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}'
        
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract text from Gemini response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    analysis = candidate['content']['parts'][0]['text']
                    
                    # Validate analysis quality
                    if len(analysis) < 200:
                        print("Warning: Analysis seems too short, might be low quality")
                        return None
                    
                    return analysis
            
            print("Unexpected Gemini response format")
            return None
        else:
            print(f"Gemini API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error analyzing news with Gemini AI: {e}")
        return None

def send_news_summary(current_price=None, trend=None):
    """Fetch news, analyze with AI, and send to Telegram"""
    try:
        print(f"\n[{datetime.now()}] Fetching Ethereum news...")
        
        # Fetch news
        news_items = fetch_crypto_news()
        
        if not news_items:
            print("No news found.")
            price_str = f"{current_price:.2f}" if current_price is not None else "N/A"
            message = f"""
📰 <b>ETHEREUM NEWS UPDATE</b> 📰
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ No significant Ethereum news found at this time.

Current Market Status:
💰 Price: ${price_str}
📊 Trend: {trend if trend else 'N/A'}
            """
            send_telegram_message(message)
            return False
        
        print(f"Found {len(news_items)} news items. Analyzing with Google Gemini AI...")
        
        # Analyze with Gemini AI
        analysis = analyze_news_with_gemini(news_items, current_price, trend)
        
        if not analysis:
            print("Failed to analyze news.")
            return False
        
        # Format message for Telegram with enhanced layout
        price_str = f"{current_price:.2f}" if current_price is not None else "N/A"
        message = f"""
📰 <b>ETHEREUM NEWS ANALYSIS</b> 📰
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'═' * 35}
<b>MARKET CONTEXT</b>
💰 ETH Price: ${price_str}
📊 Technical Trend: {trend if trend else 'Analyzing...'}
{'═' * 35}

🤖 <b>AI DEEP ANALYSIS BY GOOGLE GEMINI:</b>

{analysis}

{'═' * 35}
💡 <b>Disclaimer:</b> This is AI-generated analysis based on news sentiment. Always do your own research and consider multiple factors before trading.

📊 Combine this fundamental analysis with your technical signals for optimal decision-making!
        """
        
        # Send to Telegram
        result = send_telegram_message(message)
        
        if result and result.get('ok'):
            print("✅ News summary sent successfully!")
            return True
        else:
            print("❌ Failed to send news summary")
            return False
        
    except Exception as e:
        print(f"Error in news summary: {e}")
        return False

# ============= BINANCE FUNCTIONS =============
def initialize_exchange():
    """Initialize Binance exchange connection"""
    try:
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        return exchange
    except Exception as e:
        print(f"Error initializing exchange: {e}")
        return None

def fetch_ohlcv(exchange, symbol, timeframe, limit):
    """Fetch OHLCV data from Binance"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ============= INDICATOR CALCULATIONS =============
def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_fibonacci_levels(data, lookback):
    """Calculate Fibonacci retracement levels"""
    highest = data['high'].rolling(window=lookback).max()
    lowest = data['low'].rolling(window=lookback).min()
    diff = highest - lowest
    
    fib_54 = highest - (diff * FIB_LEVEL_1)
    fib_67 = highest - (diff * FIB_LEVEL_2)
    
    return fib_54, fib_67

def check_ema_in_zone(ema_value, fib_67, fib_54):
    """Check if EMA is within Fibonacci zone"""
    return fib_67 <= ema_value <= fib_54

def calculate_indicators(df):
    """Calculate all indicators"""
    # Calculate EMAs
    df['ema21'] = calculate_ema(df, EMA_21)
    df['ema34'] = calculate_ema(df, EMA_34)
    df['ema40'] = calculate_ema(df, EMA_40)
    df['ema50'] = calculate_ema(df, EMA_50)
    df['ema200'] = calculate_ema(df, EMA_200)
    
    # Calculate Fibonacci levels
    df['fib_54'], df['fib_67'] = calculate_fibonacci_levels(df, FIB_LOOKBACK)
    
    # Check EMAs in Fibonacci zone
    df['ema21_in_zone'] = df.apply(lambda row: check_ema_in_zone(row['ema21'], row['fib_67'], row['fib_54']), axis=1)
    df['ema34_in_zone'] = df.apply(lambda row: check_ema_in_zone(row['ema34'], row['fib_67'], row['fib_54']), axis=1)
    df['ema40_in_zone'] = df.apply(lambda row: check_ema_in_zone(row['ema40'], row['fib_67'], row['fib_54']), axis=1)
    
    # Count EMAs in zone
    df['emas_in_zone'] = (df['ema21_in_zone'].astype(int) + 
                          df['ema34_in_zone'].astype(int) + 
                          df['ema40_in_zone'].astype(int))
    
    # Maximum confluence condition
    df['max_confluence'] = df['emas_in_zone'] >= MIN_EMAS_IN_ZONE
    
    # EMA 50/200 crossover signals
    df['ema50_above_200'] = df['ema50'] > df['ema200']
    df['ema50_above_200_prev'] = df['ema50_above_200'].shift(1)
    
    df['buy_signal'] = (df['ema50_above_200'] == True) & (df['ema50_above_200_prev'] == False)
    df['sell_signal'] = (df['ema50_above_200'] == False) & (df['ema50_above_200_prev'] == True)
    
    return df

# ============= SIGNAL DETECTION =============
def check_price_in_fib_zone(price, fib_67, fib_54):
    """Check if price is within Fibonacci zone"""
    return fib_67 <= price <= fib_54

def check_for_signals(df):
    """Check for trading signals in the latest candle"""
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    signals = []
    
    # Check for BUY signal (EMA 50 crosses above EMA 200)
    if latest['buy_signal']:
        message = f"""
🟢 <b>BUY SIGNAL - {SYMBOL}</b> 🟢

📊 <b>Signal:</b> EMA 50 crossed above EMA 200
⏰ <b>Time:</b> {latest['timestamp']}
💰 <b>Price:</b> ${latest['close']:.2f}

<b>EMAs:</b>
  • EMA 21: ${latest['ema21']:.2f}
  • EMA 34: ${latest['ema34']:.2f}
  • EMA 40: ${latest['ema40']:.2f}
  • EMA 50: ${latest['ema50']:.2f}
  • EMA 200: ${latest['ema200']:.2f}

<b>Fibonacci Zone:</b>
  • Fib 0.54: ${latest['fib_54']:.2f}
  • Fib 0.67: ${latest['fib_67']:.2f}
  • EMAs in Zone: {int(latest['emas_in_zone'])}

🎯 <b>Trend:</b> BULLISH
        """
        signals.append(('BUY', message))
    
    # Check for SELL signal (EMA 50 crosses below EMA 200)
    if latest['sell_signal']:
        message = f"""
🔴 <b>SELL SIGNAL - {SYMBOL}</b> 🔴

📊 <b>Signal:</b> EMA 50 crossed below EMA 200
⏰ <b>Time:</b> {latest['timestamp']}
💰 <b>Price:</b> ${latest['close']:.2f}

<b>EMAs:</b>
  • EMA 21: ${latest['ema21']:.2f}
  • EMA 34: ${latest['ema34']:.2f}
  • EMA 40: ${latest['ema40']:.2f}
  • EMA 50: ${latest['ema50']:.2f}
  • EMA 200: ${latest['ema200']:.2f}

<b>Fibonacci Zone:</b>
  • Fib 0.54: ${latest['fib_54']:.2f}
  • Fib 0.67: ${latest['fib_67']:.2f}
  • EMAs in Zone: {int(latest['emas_in_zone'])}

🎯 <b>Trend:</b> BEARISH
        """
        signals.append(('SELL', message))
    
    # Check for maximum EMA confluence
    if latest['max_confluence'] and not previous['max_confluence']:
        message = f"""
⚡ <b>EMA CONFLUENCE ALERT - {SYMBOL}</b> ⚡

📊 <b>Signal:</b> Maximum EMA confluence in Fibonacci zone
⏰ <b>Time:</b> {latest['timestamp']}
💰 <b>Price:</b> ${latest['close']:.2f}

<b>EMAs in Fibonacci Zone:</b> {int(latest['emas_in_zone'])} / {MIN_EMAS_IN_ZONE} (minimum)

<b>Fibonacci Zone:</b>
  • Fib 0.54: ${latest['fib_54']:.2f}
  • Fib 0.67: ${latest['fib_67']:.2f}

🎯 This indicates potential support/resistance area!
        """
        signals.append(('CONFLUENCE', message))
    
    # Check if PRICE enters Fibonacci zone
    price_in_zone_now = check_price_in_fib_zone(latest['close'], latest['fib_67'], latest['fib_54'])
    price_in_zone_prev = check_price_in_fib_zone(previous['close'], previous['fib_67'], previous['fib_54'])
    
    if price_in_zone_now and not price_in_zone_prev:
        # Determine if price entered from above or below
        entry_direction = "from ABOVE" if previous['close'] > previous['fib_54'] else "from BELOW"
        entry_emoji = "⬇️" if previous['close'] > previous['fib_54'] else "⬆️"
        
        # Get trend information
        current_trend = "BULLISH 🟢" if latest['ema50'] > latest['ema200'] else "BEARISH 🔴"
        
        message = f"""
🎯 <b>PRICE ENTERED FIBONACCI ZONE - {SYMBOL}</b> 🎯

📊 <b>Signal:</b> Market price entered Fibonacci zone {entry_emoji}
⏰ <b>Time:</b> {latest['timestamp']}
💰 <b>Current Price:</b> ${latest['close']:.2f}
📍 <b>Entry Direction:</b> {entry_direction}

<b>Fibonacci Zone Levels:</b>
  • Upper (Fib 0.54): ${latest['fib_54']:.2f}
  • Lower (Fib 0.67): ${latest['fib_67']:.2f}
  • Zone Range: ${abs(latest['fib_54'] - latest['fib_67']):.2f}

<b>Previous Candle:</b>
  • Price: ${previous['close']:.2f}
  • Position: {'Above zone' if previous['close'] > previous['fib_54'] else 'Below zone' if previous['close'] < previous['fib_67'] else 'In zone'}

<b>Current Market Trend:</b> {current_trend}

<b>EMAs Status:</b>
  • EMA 21: ${latest['ema21']:.2f}
  • EMA 34: ${latest['ema34']:.2f}
  • EMA 40: ${latest['ema40']:.2f}
  • EMA 50: ${latest['ema50']:.2f}
  • EMA 200: ${latest['ema200']:.2f}
  • EMAs in Zone: {int(latest['emas_in_zone'])}

💡 <b>Note:</b> This zone often acts as support/resistance!
        """
        signals.append(('PRICE_ZONE_ENTRY', message))
    
    return signals

# ============= MAIN LOOP =============
def main():
    """Main bot loop"""
    print(f"Starting Trading Bot for {SYMBOL} on {TIMEFRAME} timeframe...")
    print(f"Bot started at: {datetime.now()}")
    
    # Send startup message
    send_telegram_message(f"🤖 <b>Trading Bot Started</b>\n\nMonitoring {SYMBOL} on {TIMEFRAME} timeframe\n⏰ Time: {datetime.now()}\n📰 News on candle close: {'Enabled' if SEND_NEWS_ON_CANDLE_CLOSE else 'Disabled'}")
    
    # Initialize exchange
    exchange = initialize_exchange()
    if not exchange:
        print("Failed to initialize exchange. Exiting...")
        return
    
    # Track last signal and candle to avoid duplicates
    last_signal_time = None
    last_candle_time = None
    candle_counter = 0
    
    # Send initial news summary
    print("\nFetching initial news summary...")
    send_news_summary()
    
    while True:
        try:
            print(f"\n[{datetime.now()}] Fetching data...")
            
            # Fetch OHLCV data
            df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, LIMIT)
            
            if df is None or df.empty:
                print("Failed to fetch data. Retrying in 60 seconds...")
                time.sleep(60)
                continue
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Get current price and trend
            current_price = df['close'].iloc[-1]
            current_candle_time = df['timestamp'].iloc[-1]
            trend = "BULLISH 🟢" if df['ema50'].iloc[-1] > df['ema200'].iloc[-1] else "BEARISH 🔴"
            
            print(f"Current Price: ${current_price:.2f}")
            print(f"Current Trend: {trend}")
            print(f"EMAs in Fib Zone: {int(df['emas_in_zone'].iloc[-1])}")
            print(f"Candle Time: {current_candle_time}")
            
            # Detect new candle completion
            new_candle_completed = False
            if last_candle_time is None or current_candle_time != last_candle_time:
                new_candle_completed = True
                candle_counter += 1
                print(f"✅ NEW CANDLE COMPLETED! (Count: {candle_counter})")
                last_candle_time = current_candle_time
            
            # Check for signals
            signals = check_for_signals(df)
            
            # Send signals to Telegram
            if signals and current_candle_time != last_signal_time:
                for signal_type, message in signals:
                    print(f"\n🚨 {signal_type} SIGNAL DETECTED!")
                    send_telegram_message(message)
                    
                    # If news sending is enabled with signals, send news too
                    if SEND_NEWS_WITH_SIGNALS and signal_type in ['BUY', 'SELL', 'PRICE_ZONE_ENTRY']:
                        print("📰 Fetching news context for signal...")
                        time.sleep(2)  # Small delay between messages
                        send_news_summary(current_price, trend)
                    
                last_signal_time = current_candle_time
            else:
                print("No new signals detected.")
            
            # Send news on candle completion (every N candles)
            if SEND_NEWS_ON_CANDLE_CLOSE and new_candle_completed:
                if candle_counter % NEWS_EVERY_N_CANDLES == 0:
                    print(f"\n📰 Candle #{candle_counter} completed - Fetching news summary...")
                    time.sleep(2)  # Small delay if signals were sent
                    send_news_summary(current_price, trend)
            
            # Wait for next check (5 minutes for 1h timeframe)
            print(f"\n⏳ Waiting 5 minutes before next check...")
            time.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user.")
            send_telegram_message("🛑 <b>Trading Bot Stopped</b>\n\nBot has been manually stopped.")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main()

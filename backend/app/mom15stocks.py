import pandas as pd
from kiteconnect import KiteConnect, KiteTicker
from datetime import datetime, timedelta
import time
import sys
import numpy as np # Added for trend detection
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
api_key_tapan = os.getenv("API_KEY_TAPAN")
api_secret_tapan = os.getenv("API_SECRET_TAPAN")
access_token_tapan = os.getenv("ACCESS_TOKEN_TAPAN")




# --- LOAD MOM'S 15 STOCKS FROM CSV ---
mom_symbols = [
    "SBIN",
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    # "HDFC",
    "INFY",
    "TCS",
    "ITC",
]

df = pd.read_csv("instrument_tokens.csv")
mom_stocks = df[df["tradingsymbol"].isin(mom_symbols)]
mom_stocks_list = mom_stocks[["tradingsymbol", "instrument_token", "name"]].to_dict("records") #///list of dicts

# print(mom_stocks_list)
# print(type(mom_stocks_list[0]))

# --- INDICATOR FUNCTIONS ---
def calculate_rsi(prices, period=14):
    """Optimized RSI calculation using vectorized operations"""
    # Calculate price changes
    delta = prices.diff()
    
    # Vectorized operations for gains and losses (more efficient)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Use ewm for exponential moving averages (Wilder's smoothing)
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()
    
    # Vectorized RS and RSI calculation
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_sma(prices, period=44):
    return prices.rolling(window=period).mean()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

# --- HISTORICAL DATA & INDICATOR STORAGE ---
historical_data = {}
indicators = {}

kite = KiteConnect(api_key=api_key_tapan)
kite.set_access_token(access_token_tapan)


#what is this for?
#it is used to fetch historical data for the stocks
#it is used to store the historical data in a dictionary
#it is used to store the indicators in a dictionary

for stock in mom_stocks_list:
    token = stock["instrument_token"]
    symbol = stock["tradingsymbol"]
    print(f"Fetching historical data for {symbol}...")
    try:
        data = kite.historical_data(
            instrument_token=token,
            from_date=(datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d"),
            to_date=datetime.now().strftime("%Y-%m-%d"),
            interval="day"
        )
        df_hist = pd.DataFrame(data) #['date', 'open', 'high', 'low', 'close', 'volume'] for specific token for specific time period
        if df_hist.empty:
            print(f"No data for {symbol}")  
            continue
        
        df_hist["rsi"] = calculate_rsi(df_hist["close"])
        df_hist["ma_44"] = calculate_sma(df_hist["close"], 44)
        df_hist["bb_upper"], df_hist["bb_middle"], df_hist["bb_lower"] = calculate_bollinger_bands(df_hist["close"])
   
        historical_data[token] = df_hist
    
        # print(historical_data[token])#['date', 'open', 'high', 'low', 'close', 'volume','rsi','ma_44','bb_upper','bb_middle','bb_lower']  for specific token for specific time period


        indicators[token] = df_hist.iloc[-1].copy()  # Store latest indicators (copy to avoid SettingWithCopyWarning)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# --- INDICATOR DIRECTION DETECTION ---
def detect_trend_direction(values, lookback=5, threshold=0.001):
    """
    Detect if indicator values are ascending, descending, or constant
    Args:
        values: pandas Series of indicator values
        lookback: number of periods to analyze (default: 5)
        threshold: minimum change to consider as trend (default: 0.001)
    Returns: "ASCENDING", "DESCENDING", or "CONSTANT"
    """
    if len(values) < lookback:
        return "INSUFFICIENT_DATA"
    
    # Get recent values
    recent_values = values.tail(lookback)
    
    # Calculate linear regression slope
    x = np.arange(len(recent_values))
    y = recent_values.values
    
    # Use numpy polyfit for efficient slope calculation
    slope = np.polyfit(x, y, 1)[0]
    
    # Normalize slope by the mean value to get relative change
    mean_value = np.mean(y)
    relative_slope = slope / mean_value if mean_value != 0 else 0
    
    # Determine direction based on threshold
    if abs(relative_slope) < threshold:
        return "CONSTANT"
    elif relative_slope > threshold:
        return "ASCENDING"
    else:
        return "DESCENDING"

def get_indicator_directions(df, lookback=5):
    """
    Get direction trends for all indicators
    Returns: dict with directions for each indicator
    """
    directions = {}
    
    # Calculate directions for each indicator
    indicators = ['rsi', 'ma_44', 'bb_upper', 'bb_middle', 'bb_lower']
    
    for indicator in indicators:
        if indicator in df.columns:
            directions[f"{indicator}_direction"] = detect_trend_direction(
                df[indicator], lookback=lookback
            )
    
    return directions

# --- ENHANCED TRADING LOGIC WITH DIRECTION ---
def apply_trading_logic_with_direction(latest, df):
    """
    Enhanced trading logic that includes indicator direction trends
    """
    rsi = latest["rsi"]
    close = latest["close"]
    ma_44 = latest["ma_44"]
    bb_upper = latest["bb_upper"]
    bb_lower = latest["bb_lower"]
    
    # Get indicator directions
    directions = get_indicator_directions(df)
    
    # Calculate price position relative to Bollinger Bands
    bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    # Enhanced BUY CONDITIONS with direction confirmation
    buy_conditions = [
        # Strong oversold with RSI ascending (recovery signal)
        rsi < 30 and close > ma_44 and directions.get("rsi_direction") == "ASCENDING",
        
        # RSI oversold with price ascending and MA stable/ascending
        rsi < 35 and close <= bb_lower * 1.02 and close > ma_44 * 0.95 and 
        directions.get("rsi_direction") == "ASCENDING" and 
        directions.get("ma_44_direction") in ["ASCENDING", "CONSTANT"],
        
        # Price above MA with RSI recovering and MA ascending
        rsi > 30 and rsi < 45 and close > ma_44 and bb_position < 0.4 and
        directions.get("rsi_direction") == "ASCENDING" and
        directions.get("ma_44_direction") == "ASCENDING",
        
        # Strong momentum with price breaking above upper BB and RSI ascending
        rsi > 50 and rsi < 70 and close > bb_upper and close > ma_44 * 1.02 and
        directions.get("rsi_direction") == "ASCENDING"
    ]
    
    # Enhanced SELL CONDITIONS with direction confirmation
    sell_conditions = [
        # Strong overbought with RSI descending (reversal signal)
        rsi > 70 and close < ma_44 and directions.get("rsi_direction") == "DESCENDING",
        
        # RSI overbought with price descending and MA stable/descending
        rsi > 65 and close >= bb_upper * 0.98 and close < ma_44 * 1.05 and
        directions.get("rsi_direction") == "DESCENDING" and
        directions.get("ma_44_direction") in ["DESCENDING", "CONSTANT"],
        
        # Price below MA with RSI declining and MA descending
        rsi < 70 and rsi > 55 and close < ma_44 and bb_position > 0.6 and
        directions.get("rsi_direction") == "DESCENDING" and
        directions.get("ma_44_direction") == "DESCENDING",
        
        # Weak momentum with price breaking below lower BB and RSI descending
        rsi < 50 and rsi > 30 and close < bb_lower and close < ma_44 * 0.98 and
        directions.get("rsi_direction") == "DESCENDING"
    ]
    
    # Check conditions
    if any(buy_conditions):
        return "BUY"
    elif any(sell_conditions):
        return "SELL"
    else:
        return "HOLD"

# --- LIVE TICK HANDLING ---
def on_ticks(ws, ticks):
    print(f"Received {len(ticks)} ticks")  # Debug print
    for tick in ticks:
        token = tick["instrument_token"]
        if token not in indicators:
            continue
        # Use the latest price from the tick, or fallback to the last known close
        latest_price = tick.get("last_price", indicators[token]["close"])
        # Recalculate indicators with new price
        df = historical_data[token].copy()
        df = pd.concat([df, pd.DataFrame([{**df.iloc[-1].to_dict(), "close": latest_price}])], ignore_index=True)
        df["rsi"] = calculate_rsi(df["close"])
        df["ma_44"] = calculate_sma(df["close"], 44)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(df["close"])
        latest = df.iloc[-1]
        indicators[token] = latest.copy()  # Update the indicators dict with the new latest row
        signal = apply_trading_logic_with_direction(latest, df)
        symbol = next((s["tradingsymbol"] for s in mom_stocks_list if s["instrument_token"] == token), str(token))
        
        # Print detailed indicator values
        print(f"{symbol}: {signal} at ₹{latest['close']:.2f} | "
              f"RSI: {latest['rsi']:.1f} ({get_indicator_directions(df).get('rsi_direction', 'N/A')}) | "
              f"MA44: ₹{latest['ma_44']:.2f} ({get_indicator_directions(df).get('ma_44_direction', 'N/A')}) | "
              f"BB Upper: ₹{latest['bb_upper']:.2f} ({get_indicator_directions(df).get('bb_upper_direction', 'N/A')}) | "
              f"BB Lower: ₹{latest['bb_lower']:.2f} ({get_indicator_directions(df).get('bb_lower_direction', 'N/A')})")

# --- BACKTEST MODE (HISTORICAL) ---
def simulate_ticks_for_backtest():
    print("\nRunning in BACKTEST mode (historical data)...\n")
    for stock in mom_stocks_list:   # [{'tradingsymbol': 'APARINDS', 'instrument_token': 2941697, 'name': 'APAR INDUSTRIES'},{'tradingsymbol': 'HAPPSTMNDS', 'instrument_token': 2941698, 'name': 'HAPPINESS SHOPPE'}] list of dicts
        token = stock["instrument_token"]
        symbol = stock["tradingsymbol"]
        df = historical_data[token]
        for _, row in df.iterrows():
            fake_tick = {"instrument_token": token, "last_price": row["close"]}
            on_ticks(None, [fake_tick])

# --- LIVE MODE ---
def start_live_mode():
    print("Starting live tick monitoring for mom's stocks...")
    
    def connect_websocket():
        kws = KiteTicker(api_key_tapan, access_token_tapan)
        kws.on_ticks = on_ticks_with_direction # Changed to new handler
        def on_connect(ws, response):
            print("WebSocket connected, subscribing to tokens...")
            tokens = [stock["instrument_token"] for stock in mom_stocks_list]
            print(f"Subscribing to tokens: {tokens}")
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            print("Subscription completed, waiting for ticks...")
        kws.on_connect = on_connect
        def on_error(ws, code, reason):
            print(f"WebSocket error: {code} {reason}")

        def on_close(ws, code, reason):
            print(f"WebSocket closed: {code} {reason}")
            print("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            try:
                connect_websocket()
            except Exception as e:
                print(f"Reconnection failed: {e}")

        def on_noreconnect(ws, code, reason):
            print(f"WebSocket noreconnect: {code} {reason}")
            print("Attempting to reconnect in 10 seconds...")
            time.sleep(10)
            try:
                connect_websocket()
            except Exception as e:
                print(f"Reconnection failed: {e}")

        kws.on_error = on_error
        kws.on_close = on_close
        kws.on_noreconnect = on_noreconnect
        print("Attempting to connect to WebSocket...")
        try:
            kws.connect(threaded=True)
            print("WebSocket connection initiated...")
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            print("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            connect_websocket()
    
    connect_websocket()

# --- ENHANCED TICK HANDLER WITH DIRECTION ---
def on_ticks_with_direction(ws, ticks):
    print(f"Received {len(ticks)} ticks")
    for tick in ticks:
        token = tick["instrument_token"]
        if token not in indicators:
            continue
        
        latest_price = tick.get("last_price", indicators[token]["close"])
        df = historical_data[token].copy()
        df = pd.concat([df, pd.DataFrame([{**df.iloc[-1].to_dict(), "close": latest_price}])], ignore_index=True)
        
        # Calculate indicators
        df["rsi"] = calculate_rsi(df["close"])
        df["ma_44"] = calculate_sma(df["close"], 44)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(df["close"])
        
        latest = df.iloc[-1]
        indicators[token] = latest.copy()
        
        # Get directions
        directions = get_indicator_directions(df)
        
        # Apply enhanced trading logic
        signal = apply_trading_logic_with_direction(latest, df)
        symbol = next((s["tradingsymbol"] for s in mom_stocks_list if s["instrument_token"] == token), str(token))
        
        # Print enhanced output with directions
        print(f"{symbol}: {signal} at ₹{latest['close']:.2f} | "
              f"RSI: {latest['rsi']:.1f} ({directions.get('rsi_direction', 'N/A')}) | "
              f"MA44: ₹{latest['ma_44']:.2f} ({directions.get('ma_44_direction', 'N/A')}) | "
              f"BB Upper: ₹{latest['bb_upper']:.2f} ({directions.get('bb_upper_direction', 'N/A')}) | "
              f"BB Lower: ₹{latest['bb_lower']:.2f} ({directions.get('bb_lower_direction', 'N/A')})")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Test access token validity
    try:
        print("Testing access token validity...")
        holdings = kite.holdings()
        print("✅ Access token is valid!")
    except Exception as e:
        print(f"❌ Access token error: {e}")
        print("Please generate a fresh access token from Zerodha Kite")
        exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        simulate_ticks_for_backtest()
    else:
        # Run both: backtest first, then live
        simulate_ticks_for_backtest()
        start_live_mode() 
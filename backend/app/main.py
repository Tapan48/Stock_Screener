from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker
from typing import List, Dict, Optional
import asyncio
from pydantic import BaseModel
import json
import time
import sys
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
api_key_tapan = os.getenv("API_KEY_TAPAN")
api_secret_tapan = os.getenv("API_SECRET_TAPAN")
access_token_tapan = os.getenv("ACCESS_TOKEN_TAPAN")

# --- STOCK SELECTION ---
# Well-known NSE stocks for BUY signals
buy_stocks_symbols = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ITC",
    "SBIN",
    "BHARTIARTL",
    "AXISBANK"        ####  add important stocks here  nse and bse both
]

# Load instrument tokens
df = pd.read_csv("instrument_tokens.csv")    ###  nse stocks only need to get bse tokens as well
buy_stocks = df[df["tradingsymbol"].isin(buy_stocks_symbols)]
buy_stocks_list = buy_stocks[["tradingsymbol", "instrument_token", "name"]].to_dict("records")

# Sell stocks will be loaded from kite.holdings() dynamically
sell_stocks_list = []

# --- INDICATOR FUNCTIONS ---
def calculate_rsi(prices, period=14):
    """Optimized RSI calculation using vectorized operations"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()
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

# Initialize Kite connection
kite = None
if api_key_tapan and access_token_tapan:
    kite = KiteConnect(api_key=api_key_tapan)
    kite.set_access_token(access_token_tapan)

def load_sell_stocks_from_holdings():
    """Load sell stocks from kite.holdings()"""
    global sell_stocks_list
    try:
        if kite:
            holdings = kite.holdings()
            holdings_stocks = []
            
            # Handle different possible structures of holdings data
            if isinstance(holdings, dict):
                # If holdings is a dictionary with 'net' key
                holdings_list = holdings.get('net', [])
            elif isinstance(holdings, list):
                # If holdings is directly a list
                holdings_list = holdings
            else:
                print(f"Unexpected holdings structure: {type(holdings)}")
                holdings_list = []
            
            print(f"Total holdings found: {len(holdings_list)}")
            
            # Get all instruments from both NSE and BSE
            try:
                nse_instruments = kite.instruments("NSE")
                bse_instruments = kite.instruments("BSE")
                all_instruments = nse_instruments + bse_instruments
                print(f"Loaded {len(all_instruments)} instruments from Kite API")
            except Exception as api_error:
                print(f"Error loading instruments from Kite API: {api_error}")
                all_instruments = []
            
            for holding in holdings_list:
                if isinstance(holding, dict):
                    symbol = holding.get('tradingsymbol')
                    name = holding.get('name', symbol)
                    exchange = holding.get('exchange', 'NSE')  # Default to NSE
                    
                    if symbol:
                        # First try to find the stock in instrument tokens CSV
                        stock_info = df[df["tradingsymbol"] == symbol]
                        
                        if not stock_info.empty:
                            # Stock found in CSV
                            stock_data = stock_info.iloc[0]
                            holdings_stocks.append({
                                "tradingsymbol": symbol,
                                "instrument_token": stock_data["instrument_token"],
                                "name": stock_data["name"]
                            })
                            print(f"✅ Found {symbol} in instrument tokens CSV")
                        else:
                            # Stock not in CSV, find it in Kite instruments
                            found_instrument = None
                            
                            # Search in all instruments
                            for instrument in all_instruments:
                                if (instrument.get('tradingsymbol') == symbol and 
                                    instrument.get('exchange') == exchange):
                                    found_instrument = instrument
                                    break
                            
                            if found_instrument:
                                holdings_stocks.append({
                                    "tradingsymbol": symbol,
                                    "instrument_token": found_instrument.get('instrument_token'),
                                    "name": name
                                })
                                print(f"✅ Found {symbol} via Kite API ({exchange})")
                            else:
                                print(f"❌ Could not find instrument for {symbol} ({exchange})")
            
            sell_stocks_list = holdings_stocks
            print(f"Loaded {len(sell_stocks_list)} stocks from holdings for SELL signals")
        else:
            print("Kite connection not available, using mock data for sell stocks")
            # Mock data for testing
            sell_stocks_list = [
                {"tradingsymbol": "SBIN", "instrument_token": 3045, "name": "SBIN"},
                {"tradingsymbol": "RELIANCE", "instrument_token": 2885, "name": "RELIANCE"},
                {"tradingsymbol": "HDFCBANK", "instrument_token": 341, "name": "HDFCBANK"}
            ]
    except Exception as e:
        print(f"Error loading holdings: {e}")
        # Fallback to mock data
        sell_stocks_list = [
            {"tradingsymbol": "SBIN", "instrument_token": 3045, "name": "SBIN"},
            {"tradingsymbol": "RELIANCE", "instrument_token": 2885, "name": "RELIANCE"},
            {"tradingsymbol": "HDFCBANK", "instrument_token": 341, "name": "HDFCBANK"}
        ]

def fetch_historical_data_for_stocks(stocks_list, signal_type):
    """Fetch historical data for given stocks"""
    print(f"Fetching historical data for {signal_type} stocks...")
    
    for i, stock in enumerate(stocks_list):
        token = stock["instrument_token"]
        symbol = stock["tradingsymbol"]
        print(f"Fetching data for {symbol} (token: {token})...")
        
        # Add small delay between API calls to avoid rate limiting
        if i > 0 and kite:
            import time
            time.sleep(0.5)  # 500ms delay between calls
        
        try:
            if kite:
                # Test if we can access the API
                try:
                    print(f"  Making API call for {symbol}...")
                    data = kite.historical_data(
                        instrument_token=token,
                        from_date=(datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d"),
                        to_date=datetime.now().strftime("%Y-%m-%d"),
                        interval="day"
                    )
                    print(f"  API call successful for {symbol}, got {len(data)} records")
                except Exception as api_error:
                    print(f"  API error for {symbol}: {api_error}")
                    # Use mock data if API fails
                    data = generate_mock_historical_data(symbol)
            else:
                # Mock data for testing
                data = generate_mock_historical_data(symbol)
            
            df_hist = pd.DataFrame(data)
            if df_hist.empty:
                print(f"  No data for {symbol}, using mock data")
                data = generate_mock_historical_data(symbol)
                df_hist = pd.DataFrame(data)
            
            print(f"  Calculating indicators for {symbol}...")
            try:
                df_hist["rsi"] = calculate_rsi(df_hist["close"])
                df_hist["ma_44"] = calculate_sma(df_hist["close"], 44)
                df_hist["bb_upper"], df_hist["bb_middle"], df_hist["bb_lower"] = calculate_bollinger_bands(df_hist["close"])
                
                # Check if indicators are valid
                latest = df_hist.iloc[-1]
                if (pd.isna(latest["rsi"]) or pd.isna(latest["ma_44"]) or 
                    pd.isna(latest["bb_upper"]) or pd.isna(latest["bb_lower"])):
                    print(f"  ⚠️  Invalid indicators for {symbol}, using mock data")
                    data = generate_mock_historical_data(symbol)
                    df_hist = pd.DataFrame(data)
                    df_hist["rsi"] = calculate_rsi(df_hist["close"])
                    df_hist["ma_44"] = calculate_sma(df_hist["close"], 44)
                    df_hist["bb_upper"], df_hist["bb_middle"], df_hist["bb_lower"] = calculate_bollinger_bands(df_hist["close"])
                
                historical_data[token] = df_hist
                indicators[token] = df_hist.iloc[-1].copy()
                print(f"✅ Successfully loaded data for {symbol}")
                
            except Exception as calc_error:
                print(f"  Error calculating indicators for {symbol}: {calc_error}")
                # Generate mock data as fallback
                data = generate_mock_historical_data(symbol)
                df_hist = pd.DataFrame(data)
                df_hist["rsi"] = calculate_rsi(df_hist["close"])
                df_hist["ma_44"] = calculate_sma(df_hist["close"], 44)
                df_hist["bb_upper"], df_hist["bb_middle"], df_hist["bb_lower"] = calculate_bollinger_bands(df_hist["close"])
                
                historical_data[token] = df_hist
                indicators[token] = df_hist.iloc[-1].copy()
                print(f"✅ Generated mock data for {symbol}")
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            # Generate mock data as fallback
            try:
                data = generate_mock_historical_data(symbol)
                df_hist = pd.DataFrame(data)
                df_hist["rsi"] = calculate_rsi(df_hist["close"])
                df_hist["ma_44"] = calculate_sma(df_hist["close"], 44)
                df_hist["bb_upper"], df_hist["bb_middle"], df_hist["bb_lower"] = calculate_bollinger_bands(df_hist["close"])
                
                historical_data[token] = df_hist
                indicators[token] = df_hist.iloc[-1].copy()
                print(f"✅ Generated mock data for {symbol} (fallback)")
            except Exception as mock_error:
                print(f"❌ Failed to generate mock data for {symbol}: {mock_error}")

def generate_mock_historical_data(symbol):
    """Generate mock historical data for testing"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
    base_price = 1000 if symbol in ["RELIANCE", "TCS"] else 500
    
    data = []
    for i, date in enumerate(dates):
        # Simulate realistic price movements
        price_change = np.random.normal(0, 0.02)  # 2% daily volatility
        base_price *= (1 + price_change)
        
        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(base_price, 2),
            "volume": int(np.random.uniform(1000000, 5000000))
        })
    
    return data

# --- INDICATOR DIRECTION DETECTION ---
def detect_trend_direction(values, lookback=5, threshold=0.001):
    if len(values) < lookback:
        return "INSUFFICIENT_DATA"
    
    recent_values = values.tail(lookback)
    x = np.arange(len(recent_values))
    y = recent_values.values
    slope = np.polyfit(x, y, 1)[0]
    mean_value = np.mean(y)
    relative_slope = slope / mean_value if mean_value != 0 else 0
    
    if abs(relative_slope) < threshold:
        return "CONSTANT"
    elif relative_slope > threshold:
        return "ASCENDING"
    else:
        return "DESCENDING"

def get_indicator_directions(df, lookback=5):
    directions = {}
    indicators = ['rsi', 'ma_44', 'bb_upper', 'bb_middle', 'bb_lower']
    
    for indicator in indicators:
        if indicator in df.columns:
            directions[f"{indicator}_direction"] = detect_trend_direction(
                df[indicator], lookback=lookback
            )
    
    return directions

# --- ENHANCED TRADING LOGIC WITH DIRECTION ---
def apply_trading_logic_with_direction(latest, df):
    rsi = latest["rsi"]
    close = latest["close"]
    ma_44 = latest["ma_44"]
    bb_upper = latest["bb_upper"]
    bb_lower = latest["bb_lower"]
    
    directions = get_indicator_directions(df)
    bb_position = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    # Enhanced BUY CONDITIONS
    buy_conditions = [
        rsi < 30 and close > ma_44 and directions.get("rsi_direction") == "ASCENDING",
        rsi < 35 and close <= bb_lower * 1.02 and close > ma_44 * 0.95 and 
        directions.get("rsi_direction") == "ASCENDING" and 
        directions.get("ma_44_direction") in ["ASCENDING", "CONSTANT"],
        rsi > 30 and rsi < 45 and close > ma_44 and bb_position < 0.4 and
        directions.get("rsi_direction") == "ASCENDING" and
        directions.get("ma_44_direction") == "ASCENDING",
        rsi > 50 and rsi < 70 and close > bb_upper and close > ma_44 * 1.02 and
        directions.get("rsi_direction") == "ASCENDING"
    ]
    
    # Enhanced SELL CONDITIONS
    sell_conditions = [
        rsi > 70 and close < ma_44 and directions.get("rsi_direction") == "DESCENDING",
        rsi > 65 and close >= bb_upper * 0.98 and close < ma_44 * 1.05 and
        directions.get("rsi_direction") == "DESCENDING" and
        directions.get("ma_44_direction") in ["DESCENDING", "CONSTANT"],
        rsi < 70 and rsi > 55 and close < ma_44 and bb_position > 0.6 and
        directions.get("rsi_direction") == "DESCENDING" and
        directions.get("ma_44_direction") == "DESCENDING",
        rsi < 50 and rsi > 30 and close < bb_lower and close < ma_44 * 0.98 and
        directions.get("rsi_direction") == "DESCENDING"
    ]
    
    if any(buy_conditions):
        return "BUY"
    elif any(sell_conditions):
        return "SELL"
    else:
        return "HOLD"

# --- PYDANTIC MODELS ---
class StockIndicator(BaseModel):
    rsi: float
    rsi_direction: str
    ma_44: float
    ma_44_direction: str
    bb_upper: float
    bb_upper_direction: str
    bb_lower: float
    bb_lower_direction: str

class StockSignal(BaseModel):
    symbol: str
    name: str
    close: float
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    signal: str
    indicators: StockIndicator
    timestamp: Optional[str] = None

class StockSignalsResponse(BaseModel):
    buy_signals: List[StockSignal]
    sell_signals: List[StockSignal]
    hold_signals: List[StockSignal]

class SystemStatus(BaseModel):
    zerodha_connected: bool
    websocket_active: bool
    stocks_monitored: int

class LiveStockUpdate(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    signal: str
    timestamp: str

# --- WEBSOCKET CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected websockets
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

# --- FASTAPI APP ---
app = FastAPI(title="Stock Screener API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INITIALIZATION ---
def initialize_stock_data():
    """Initialize stock data on startup"""
    global historical_data, indicators, buy_stocks_list
    
    print("Initializing stock data...")
    
    # Load sell stocks from holdings first
    load_sell_stocks_from_holdings()
    
    # Filter buy stocks to exclude any that are already in holdings
    holdings_symbols = [stock["tradingsymbol"] for stock in sell_stocks_list]
    original_buy_stocks = buy_stocks_list.copy()
    
    buy_stocks_list = [stock for stock in original_buy_stocks if stock["tradingsymbol"] not in holdings_symbols]
    
    filtered_out = [stock["tradingsymbol"] for stock in original_buy_stocks if stock["tradingsymbol"] in holdings_symbols]
    if filtered_out:
        print(f"⏭️  Filtered out from buy stocks (already in holdings): {filtered_out}")
    
    print(f"Buy stocks after filtering: {len(buy_stocks_list)} stocks")
    print(f"Sell stocks from holdings: {len(sell_stocks_list)} stocks")
    
    # Fetch historical data for buy stocks
    fetch_historical_data_for_stocks(buy_stocks_list, "BUY")
    
    # Fetch historical data for sell stocks
    fetch_historical_data_for_stocks(sell_stocks_list, "SELL")
    
    print(f"Initialized data for {len(historical_data)} stocks")

def get_stock_signals():
    """Get current stock signals"""
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    # Process buy stocks
    for stock in buy_stocks_list:
        token = stock["instrument_token"]
        if token in indicators:
            latest = indicators[token]
            df = historical_data[token]
            signal = apply_trading_logic_with_direction(latest, df)
            directions = get_indicator_directions(df)
            
            stock_signal = StockSignal(
                symbol=stock["tradingsymbol"],
                name=stock["name"],
                close=latest["close"],
                change=latest.get("change", 0),
                change_percent=latest.get("change_percent", 0),
                volume=latest.get("volume", 0),
                signal=signal,
                indicators=StockIndicator(
                    rsi=latest["rsi"],
                    rsi_direction=directions.get("rsi_direction", "CONSTANT"),
                    ma_44=latest["ma_44"],
                    ma_44_direction=directions.get("ma_44_direction", "CONSTANT"),
                    bb_upper=latest["bb_upper"],
                    bb_upper_direction=directions.get("bb_upper_direction", "CONSTANT"),
                    bb_lower=latest["bb_lower"],
                    bb_lower_direction=directions.get("bb_lower_direction", "CONSTANT")
                ),
                timestamp=datetime.now().isoformat()
            )
            
            if signal == "BUY":
                buy_signals.append(stock_signal)
            elif signal == "SELL":
                sell_signals.append(stock_signal)
            else:
                hold_signals.append(stock_signal)
    
    # Process sell stocks (from holdings)
    for stock in sell_stocks_list:
        token = stock["instrument_token"]
        if token in indicators:
            latest = indicators[token]
            df = historical_data[token]
            signal = apply_trading_logic_with_direction(latest, df)
            directions = get_indicator_directions(df)
            
            stock_signal = StockSignal(
                symbol=stock["tradingsymbol"],
                name=stock["name"],
                close=latest["close"],
                change=latest.get("change", 0),
                change_percent=latest.get("change_percent", 0),
                volume=latest.get("volume", 0),
                signal=signal,
                indicators=StockIndicator(
                    rsi=latest["rsi"],
                    rsi_direction=directions.get("rsi_direction", "CONSTANT"),
                    ma_44=latest["ma_44"],
                    ma_44_direction=directions.get("ma_44_direction", "CONSTANT"),
                    bb_upper=latest["bb_upper"],
                    bb_upper_direction=directions.get("bb_upper_direction", "CONSTANT"),
                    bb_lower=latest["bb_lower"],
                    bb_lower_direction=directions.get("bb_lower_direction", "CONSTANT")
                ),
                timestamp=datetime.now().isoformat()
            )
            
            if signal == "SELL":
                sell_signals.append(stock_signal)
            elif signal == "BUY":
                buy_signals.append(stock_signal)
            else:
                hold_signals.append(stock_signal)
    
    return StockSignalsResponse(
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        hold_signals=hold_signals
    )

# --- API ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "Stock Screener API is running"}

@app.get("/health")
async def health_check():
    return SystemStatus(
        zerodha_connected=kite is not None,
        websocket_active=len(manager.active_connections) > 0,
        stocks_monitored=len(historical_data)
    )

@app.get("/api/stocks/signals")
async def get_stock_signals_endpoint():
    return get_stock_signals()

@app.get("/api/stocks/holdings")
async def get_holdings_stocks():
    """Get holdings stocks with their details"""
    holdings_signals = []
    
    print(f"Total sell_stocks_list: {len(sell_stocks_list)}")
    print(f"Available indicators: {len(indicators)}")
    print(f"sell_stocks_list symbols: {[s['tradingsymbol'] for s in sell_stocks_list]}")
    print(f"Available indicator tokens: {list(indicators.keys())}")
    
    for stock in sell_stocks_list:
        token = stock["instrument_token"]
        symbol = stock["tradingsymbol"]
        name = stock["name"]
        
        print(f"Processing {symbol} (token: {token})")
        
        if token in indicators:
            print(f"✅ {symbol} has indicators")
            indicator_data = indicators[token]
            
            # Get directions
            directions = get_indicator_directions(historical_data[token])
            
            # Create stock indicator
            stock_indicator = StockIndicator(
                rsi=indicator_data["rsi"],
                rsi_direction=directions.get("rsi_direction", "CONSTANT"),
                ma_44=indicator_data["ma_44"],
                ma_44_direction=directions.get("ma_44_direction", "CONSTANT"),
                bb_upper=indicator_data["bb_upper"],
                bb_upper_direction=directions.get("bb_upper_direction", "CONSTANT"),
                bb_lower=indicator_data["bb_lower"],
                bb_lower_direction=directions.get("bb_lower_direction", "CONSTANT")
            )
            
            # Apply trading logic
            signal_result = apply_trading_logic_with_direction(indicator_data, historical_data[token])
            
            # Create stock signal
            stock_signal = StockSignal(
                symbol=symbol,
                name=name,
                close=indicator_data["close"],
                change=indicator_data.get("change", 0),
                change_percent=indicator_data.get("change_percent", 0),
                volume=indicator_data.get("volume", 0),
                signal=signal_result,  # signal_result is already a string
                indicators=stock_indicator,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            holdings_signals.append(stock_signal)
            print(f"✅ Added {symbol} to holdings signals")
        else:
            print(f"❌ {symbol} missing indicators (token: {token})")
    
    print(f"Returning {len(holdings_signals)} holdings signals")
    return {"holdings": holdings_signals}

@app.get("/api/stocks/{symbol}")
async def get_stock_details(symbol: str):
    # Find stock by symbol
    all_stocks = buy_stocks_list + sell_stocks_list
    stock = next((s for s in all_stocks if s["tradingsymbol"] == symbol), None)
    
    if not stock:
        return {"error": "Stock not found"}
    
    token = stock["instrument_token"]
    if token not in indicators:
        return {"error": "Stock data not available"}
    
    latest = indicators[token]
    return {
        "symbol": symbol,
        "name": stock["name"],
        "close": latest["close"],
        "indicators": {
            "rsi": latest["rsi"],
            "ma_44": latest["ma_44"],
            "bb_upper": latest["bb_upper"],
            "bb_lower": latest["bb_lower"]
        }
    }

@app.get("/api/stocks/{symbol}/history")
async def get_stock_history(symbol: str):
    # Find stock by symbol
    all_stocks = buy_stocks_list + sell_stocks_list
    stock = next((s for s in all_stocks if s["tradingsymbol"] == symbol), None)
    
    if not stock:
        return {"error": "Stock not found"}
    
    token = stock["instrument_token"]
    if token not in historical_data:
        return {"error": "Historical data not available"}
    
    df = historical_data[token]
    
    # Convert DataFrame to a JSON-compatible format
    history_records = []
    
    for index, row in df.iterrows():
        record = {}
        for column in df.columns:
            value = row[column]
            
            # Handle different data types
            if pd.isna(value) or value is None:
                record[column] = None
            elif isinstance(value, (np.integer, np.int64)):
                record[column] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                record[column] = float(value)
            elif isinstance(value, (np.datetime64, pd.Timestamp)):
                record[column] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
            elif isinstance(value, str):
                record[column] = value
            else:
                # Convert any other types to string
                record[column] = str(value)
        
        history_records.append(record)
    
    return {
        "symbol": symbol,
        "history": history_records
    }

@app.get("/api/stocks/popular")
async def get_popular_stocks():
    # Return the buy stocks list as popular stocks
    return {"stocks": buy_stocks_list}

# --- WEBSOCKET ENDPOINTS ---
@app.websocket("/ws/live")
async def websocket_live_data(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- BACKGROUND TASKS ---
async def broadcast_live_updates():
    """Background task to broadcast live updates"""
    while True:
        try:
            if manager.active_connections:
                # Get current signals
                signals = get_stock_signals()  # This is now a regular function
                
                # Create live updates from current data
                live_updates = []
                all_stocks = buy_stocks_list + sell_stocks_list
                
                for stock in all_stocks[:10]:  # Limit to 10 stocks for live updates
                    token = stock["instrument_token"]
                    if token in indicators:
                        latest = indicators[token]
                        df = historical_data[token]
                        signal = apply_trading_logic_with_direction(latest, df)
                        
                        live_update = LiveStockUpdate(
                            symbol=stock["tradingsymbol"],
                            price=latest["close"],
                            change=latest.get("change", 0),
                            change_percent=latest.get("change_percent", 0),
                            volume=latest.get("volume", 0),
                            signal=signal,
                            timestamp=datetime.now().isoformat()
                        )
                        live_updates.append(live_update)
                
                # Broadcast to all connected clients
                await manager.broadcast(json.dumps({
                    "type": "live_updates",
                    "data": [update.model_dump() for update in live_updates]  # Use model_dump instead of dict
                }))
                
                print(f"Broadcasted {len(live_updates)} live updates to {len(manager.active_connections)} clients")
            else:
                print("No active WebSocket connections")
                
        except Exception as e:
            print(f"Error in broadcast_live_updates: {e}")
        
        await asyncio.sleep(30)  # Update every 30 seconds

# --- STARTUP EVENT ---
@app.on_event("startup")
async def startup_event():
    initialize_stock_data()
    asyncio.create_task(broadcast_live_updates())

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
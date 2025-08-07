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
from kiteconnect import KiteConnect
from typing import List, Dict, Optional
import asyncio
from pydantic import BaseModel
import json
import time

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Stock Screener API",
    description="API for stock screening with technical indicators and real-time signals",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic models for API responses
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
    signal: str  # BUY, SELL, HOLD
    close: float
    indicators: StockIndicator
    last_updated: str

class StockSignalsResponse(BaseModel):
    buy_signals: List[StockSignal]
    sell_signals: List[StockSignal]
    hold_signals: List[StockSignal]
    total_stocks: int

class SystemStatus(BaseModel):
    status: str
    zerodha_connected: bool
    websocket_active: bool
    stocks_monitored: int
    last_update: str

class LiveStockUpdate(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str
    signal: str
    indicators: StockIndicator

# Global variables for storing data
historical_data = {}
indicators = {}
mom_stocks_list = []
kite = None
live_data_task = None

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

def detect_trend_direction(values, lookback=5, threshold=0.001):
    """Detect if indicator values are ascending, descending, or constant"""
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
    """Get direction trends for all indicators"""
    directions = {}
    indicators = ['rsi', 'ma_44', 'bb_upper', 'bb_middle', 'bb_lower']
    
    for indicator in indicators:
        if indicator in df.columns:
            directions[f"{indicator}_direction"] = detect_trend_direction(
                df[indicator], lookback=lookback
            )
    
    return directions

def apply_trading_logic_with_direction(latest, df):
    """Enhanced trading logic that includes indicator direction trends"""
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

def initialize_stock_data():
    """Initialize stock data and indicators"""
    global historical_data, indicators, mom_stocks_list, kite
    
    # Initialize Zerodha connection
    api_key = os.getenv("API_KEY_TAPAN")
    api_secret = os.getenv("API_SECRET_TAPAN")
    access_token = os.getenv("ACCESS_TOKEN_TAPAN")
    
    if not all([api_key, api_secret, access_token]):
        print("Warning: Zerodha credentials not found. Using mock data.")
        return False
    
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Load mom's stocks
        mom_symbols = ["SBIN", "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC"]
        
        # For now, use mock data structure
        for symbol in mom_symbols:
            # Create mock data structure
            mock_data = pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
                'close': np.random.normal(1000, 100, 100),
                'open': np.random.normal(1000, 100, 100),
                'high': np.random.normal(1020, 100, 100),
                'low': np.random.normal(980, 100, 100),
                'volume': np.random.randint(1000000, 5000000, 100)
            })
            
            # Calculate indicators
            mock_data["rsi"] = calculate_rsi(mock_data["close"])
            mock_data["ma_44"] = calculate_sma(mock_data["close"], 44)
            mock_data["bb_upper"], mock_data["bb_middle"], mock_data["bb_lower"] = calculate_bollinger_bands(mock_data["close"])
            
            historical_data[symbol] = mock_data
            indicators[symbol] = mock_data.iloc[-1].copy()
            mom_stocks_list.append({"tradingsymbol": symbol, "name": symbol})
        
        return True
    except Exception as e:
        print(f"Error initializing Zerodha connection: {e}")
        return False

# Initialize data on startup
@app.on_event("startup")
async def startup_event():
    """Initialize stock data and start live data broadcasting"""
    initialize_stock_data()
    
    # Start live data broadcasting
    global live_data_task
    if live_data_task is None:
        live_data_task = asyncio.create_task(broadcast_live_updates())

@app.get("/")
async def root():
    return {"message": "Stock Screener API is running!", "version": "1.0.0"}

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Get system health status"""
    return SystemStatus(
        status="healthy",
        zerodha_connected=kite is not None,
        websocket_active=len(manager.active_connections) > 0,
        stocks_monitored=len(mom_stocks_list),
        last_update=datetime.now().isoformat()
    )

@app.get("/api/stocks/signals", response_model=StockSignalsResponse)
async def get_stock_signals():
    """Get all stock signals (BUY/SELL/HOLD)"""
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    for stock in mom_stocks_list:
        symbol = stock["tradingsymbol"]
        if symbol not in indicators:
            continue
            
        latest = indicators[symbol]
        df = historical_data[symbol]
        signal = apply_trading_logic_with_direction(latest, df)
        directions = get_indicator_directions(df)
        
        stock_signal = StockSignal(
            symbol=symbol,
            signal=signal,
            close=float(latest["close"]),
            indicators=StockIndicator(
                rsi=float(latest["rsi"]),
                rsi_direction=directions.get("rsi_direction", "CONSTANT"),
                ma_44=float(latest["ma_44"]),
                ma_44_direction=directions.get("ma_44_direction", "CONSTANT"),
                bb_upper=float(latest["bb_upper"]),
                bb_upper_direction=directions.get("bb_upper_direction", "CONSTANT"),
                bb_lower=float(latest["bb_lower"]),
                bb_lower_direction=directions.get("bb_lower_direction", "CONSTANT")
            ),
            last_updated=datetime.now().isoformat()
        )
        
        if signal == "BUY":
            buy_signals.append(stock_signal)
        elif signal == "SELL":
            sell_signals.append(stock_signal)
        else:
            hold_signals.append(stock_signal)
    
    return StockSignalsResponse(
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        hold_signals=hold_signals,
        total_stocks=len(mom_stocks_list)
    )

@app.get("/api/stocks/{symbol}")
async def get_stock_details(symbol: str):
    """Get detailed data for a specific stock"""
    if symbol not in indicators:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    latest = indicators[symbol]
    df = historical_data[symbol]
    signal = apply_trading_logic_with_direction(latest, df)
    directions = get_indicator_directions(df)
    
    return {
        "symbol": symbol,
        "signal": signal,
        "close": float(latest["close"]),
        "indicators": {
            "rsi": float(latest["rsi"]),
            "rsi_direction": directions.get("rsi_direction", "CONSTANT"),
            "ma_44": float(latest["ma_44"]),
            "ma_44_direction": directions.get("ma_44_direction", "CONSTANT"),
            "bb_upper": float(latest["bb_upper"]),
            "bb_upper_direction": directions.get("bb_upper_direction", "CONSTANT"),
            "bb_lower": float(latest["bb_lower"]),
            "bb_lower_direction": directions.get("bb_lower_direction", "CONSTANT")
        },
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/stocks/{symbol}/history")
async def get_stock_history(symbol: str, days: int = 30):
    """Get historical data for a specific stock"""
    if symbol not in historical_data:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    df = historical_data[symbol].tail(days)
    
    return {
        "symbol": symbol,
        "data": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "close": float(row["close"]),
                "rsi": float(row["rsi"]),
                "ma_44": float(row["ma_44"]),
                "bb_upper": float(row["bb_upper"]),
                "bb_lower": float(row["bb_lower"])
            }
            for _, row in df.iterrows()
        ]
    }

@app.get("/api/stocks/popular")
async def get_popular_stocks():
    """Get data for popular Indian stocks using yfinance"""
    popular_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    results = []
    
    for symbol in popular_stocks:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            
            if data.empty:
                continue
            
            # Calculate basic indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            latest = data.iloc[-1]
            
            results.append({
                "symbol": symbol.replace('.NS', ''),
                "current_price": round(float(latest['Close']), 2),
                "volume": int(latest['Volume']),
                "sma_20": round(float(latest['SMA_20']), 2) if not pd.isna(latest['SMA_20']) else None,
                "sma_50": round(float(latest['SMA_50']), 2) if not pd.isna(latest['SMA_50']) else None,
                "last_updated": latest.name.strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    
    return {"stocks": results, "count": len(results)}

# --- WEBSOCKET ENDPOINTS FOR LIVE DATA ---

@app.websocket("/ws/live")
async def websocket_live_data(websocket: WebSocket):
    """WebSocket endpoint for live stock data updates"""
    await manager.connect(websocket)
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "message": "Connected to live stock data feed",
                "timestamp": datetime.now().isoformat()
            }), 
            websocket
        )
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message from client (ping/pong)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }), 
                        websocket
                    )
                    
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time trading signals"""
    await manager.connect(websocket)
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "message": "Connected to trading signals feed",
                "timestamp": datetime.now().isoformat()
            }), 
            websocket
        )
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }), 
                        websocket
                    )
                    
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- LIVE DATA FUNCTIONS ---

async def fetch_live_stock_data(symbol: str):
    """Fetch live stock data using yfinance"""
    try:
        # Handle both string and dictionary inputs
        if isinstance(symbol, dict):
            symbol_str = symbol.get('tradingsymbol', symbol.get('name', str(symbol)))
        else:
            symbol_str = str(symbol)
        
        # Add .NS suffix for Indian stocks if not present
        if not symbol_str.endswith('.NS'):
            symbol_str = f"{symbol_str}.NS"
        
        stock = yf.Ticker(symbol_str)
        # Get real-time data
        info = stock.info
        hist = stock.history(period="5d")
        
        if hist.empty:
            return None
            
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        # Calculate basic indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        
        current_price = float(latest['Close'])
        prev_price = float(prev['Close'])
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100 if prev_price > 0 else 0
        
        # Simple signal logic
        sma_20 = hist['SMA_20'].iloc[-1] if not pd.isna(hist['SMA_20'].iloc[-1]) else current_price
        rsi = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else 50
        
        if current_price > sma_20 and rsi < 70:
            signal = "BUY"
        elif current_price < sma_20 and rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Extract clean symbol name for display
        display_symbol = symbol_str.replace('.NS', '')
        
        return {
            "symbol": display_symbol,
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": int(latest['Volume']),
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "indicators": {
                "rsi": round(rsi, 2),
                "rsi_direction": "ASCENDING" if rsi > 50 else "DESCENDING",
                "ma_44": round(sma_20, 2),
                "ma_44_direction": "ASCENDING" if current_price > sma_20 else "DESCENDING",
                "bb_upper": round(current_price * 1.02, 2),
                "bb_upper_direction": "ASCENDING",
                "bb_lower": round(current_price * 0.98, 2),
                "bb_lower_direction": "DESCENDING"
            }
        }
    except Exception as e:
        print(f"Error fetching live data for {symbol}: {e}")
        return None

async def broadcast_live_updates():
    """Background task to broadcast live stock updates"""
    global live_data_task
    
    while True:
        try:
            if manager.active_connections:
                # Fetch live data for monitored stocks
                live_updates = []
                
                for stock_item in mom_stocks_list[:10]:  # Limit to first 10 stocks for performance
                    try:
                        live_data = await fetch_live_stock_data(stock_item)
                        if live_data:
                            live_updates.append(live_data)
                    except Exception as e:
                        print(f"Error processing stock {stock_item}: {e}")
                        continue
                
                if live_updates:
                    # Broadcast to all connected clients
                    try:
                        await manager.broadcast(json.dumps({
                            "type": "live_update",
                            "data": live_updates,
                            "timestamp": datetime.now().isoformat()
                        }))
                        print(f"Broadcasted {len(live_updates)} live updates to {len(manager.active_connections)} clients")
                    except Exception as e:
                        print(f"Error broadcasting live updates: {e}")
                
                # Update system status
                try:
                    await manager.broadcast(json.dumps({
                        "type": "system_status",
                        "data": {
                            "status": "active",
                            "zerodha_connected": kite is not None,
                            "websocket_active": len(manager.active_connections) > 0,
                            "stocks_monitored": len(mom_stocks_list),
                            "last_update": datetime.now().isoformat()
                        }
                    }))
                except Exception as e:
                    print(f"Error broadcasting system status: {e}")
            else:
                print("No active WebSocket connections")
            
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            print(f"Error in live update broadcast: {e}")
            await asyncio.sleep(60)  # Wait longer on error

# Start live data broadcasting on startup
# @app.on_event("startup")
# async def start_live_data():
#     global live_data_task
#     if live_data_task is None:
#         live_data_task = asyncio.create_task(broadcast_live_updates())

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
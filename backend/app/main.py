from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yfinance as yf
import pandas as pd
from datetime import datetime

app = FastAPI(
    title="Stock Screener API",
    description="API for stock screening with technical indicators",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Stock Screener API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "stock-screener-api"}

@app.get("/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get basic stock data for testing"""
    try:
        # Add .NS for Indian stocks if not present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        stock = yf.Ticker(symbol)
        data = stock.history(period="3mo")
        
        if data.empty:
            return {"error": f"No data found for {symbol}"}
        
        # Calculate basic indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Get latest data
        latest = data.iloc[-1]
        
        return {
            "symbol": symbol,
            "current_price": round(float(latest['Close']), 2),
            "volume": int(latest['Volume']),
            "sma_20": round(float(latest['SMA_20']), 2) if not pd.isna(latest['SMA_20']) else None,
            "sma_50": round(float(latest['SMA_50']), 2) if not pd.isna(latest['SMA_50']) else None,
            "last_updated": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
            "data_points": len(data)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/stocks/popular")
async def get_popular_stocks():
    """Get data for popular Indian stocks"""
    popular_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    results = []
    
    for stock in popular_stocks:
        try:
            stock_data = await get_stock_data(stock)
            if 'error' not in stock_data:
                results.append(stock_data)
        except:
            continue
    
    return {"stocks": results, "count": len(results)}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
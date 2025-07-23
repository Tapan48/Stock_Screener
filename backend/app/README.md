# Stock Screener for Mom's Portfolio

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the `backend/app` directory with your Zerodha API credentials:

```env
# Zerodha Kite Connect API Credentials
API_KEY_TAPAN=your_api_key_here
API_SECRET_TAPAN=your_api_secret_here
ACCESS_TOKEN_TAPAN=your_access_token_here
```

### 3. Get API Credentials
1. Go to [Zerodha Kite Connect](https://kite.trade/docs/connect/v3/)
2. Create an application to get your API key and secret
3. Generate an access token using the API
4. Add them to the `.env` file

### 4. Run the Application
```bash
python mom15stocks.py
```

## Features
- Real-time stock monitoring
- RSI, SMA, and Bollinger Bands indicators
- Direction trend detection
- Live WebSocket connection
- Backtesting mode

## Files
- `mom15stocks.py` - Main application
- `live_ticks.py` - Live tick monitoring
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create this)
- `.gitignore` - Git ignore rules 
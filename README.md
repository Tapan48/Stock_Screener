# Stock Screener

A comprehensive stock screening application built with Python and React that uses Zerodha Kite Connect API to fetch historical and live stock data, calculate technical indicators, and generate trading signals.

## Features

- **Real-time Stock Monitoring**: Live WebSocket connection for real-time tick data
- **Technical Indicators**: RSI, SMA44, Bollinger Bands with optimized calculations
- **Trend Analysis**: Direction detection (ascending/descending/constant) using linear regression
- **Trading Signals**: Buy/Sell/Hold recommendations based on indicator values and trends
- **Backtesting**: Historical data analysis and strategy testing
- **Modern Web Interface**: Built with React, TypeScript, and Material-UI

## Project Structure

```
stock-screener/
├── backend/
│   └── app/
│       ├── mom15stocks.py      # Main stock screening application
│       ├── live_ticks.py       # Live tick monitoring
│       ├── requirements.txt     # Python dependencies
│       └── README.md           # Backend setup instructions
├── frontend/
│   └── src/
│       └── components/         # React components
└── README.md                   # This file
```

## Quick Start

### Backend Setup
1. Navigate to `backend/app/`
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with your Zerodha API credentials
4. Run: `python mom15stocks.py`

### Frontend Setup
1. Navigate to `frontend/`
2. Install dependencies: `npm install`
3. Run: `npm start`

## Technologies Used

- **Backend**: Python, pandas, kiteconnect, python-dotenv
- **Frontend**: React, TypeScript, Material-UI, Tailwind CSS
- **API**: Zerodha Kite Connect
- **Data Analysis**: Technical indicators, trend analysis

## Development Phases

1. **Phase 1**: Basic API integration and data fetching
2. **Phase 2**: Technical indicator calculations (RSI, SMA, Bollinger Bands)
3. **Phase 3**: Trading logic and signal generation
4. **Phase 4**: Live WebSocket integration
5. **Phase 5**: Environment variable management and security
6. **Phase 6**: Frontend development with modern UI
7. **Phase 7**: Integration and testing

## License

MIT License 
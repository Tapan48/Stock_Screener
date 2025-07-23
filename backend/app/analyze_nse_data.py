import pandas as pd
import re

def analyze_nse_data():
    """
    Analyze the cleaned NSE data to understand what types of instruments are included
    """
    print("="*60)
    print("NSE DATA ANALYSIS")
    print("="*60)
    
    # Load the cleaned data
    df = pd.read_csv('nse_cleaned_stocks.csv')
    print(f"Total instruments: {len(df)}")
    
    # Categorize instruments
    categories = {
        'Government Securities': [],
        'Treasury Bills': [],
        'ETFs': [],
        'Equity Stocks': [],
        'Others': []
    }
    
    for _, row in df.iterrows():
        symbol = row['tradingsymbol']
        name = row['name']
        
        # Government Securities
        if 'GS' in symbol or 'GOI LOAN' in name or 'GOVERNMENT' in name.upper():
            categories['Government Securities'].append(row)
        # Treasury Bills
        elif 'TB' in symbol or 'TBILL' in name.upper():
            categories['Treasury Bills'].append(row)
        # ETFs
        elif 'IETF' in symbol or 'ETF' in name.upper():
            categories['ETFs'].append(row)
        # Equity Stocks (major companies)
        elif any(major in symbol for major in ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI', 'ITC', 'SBIN', 'WIPRO', 'BHARTI', 'AXIS']):
            categories['Equity Stocks'].append(row)
        # Other equity stocks (not ETFs, not government securities)
        elif not any(pattern in symbol for pattern in ['IETF', 'GS', 'TB']) and not any(pattern in name.upper() for pattern in ['ETF', 'GOVERNMENT', 'TBILL']):
            categories['Equity Stocks'].append(row)
        else:
            categories['Others'].append(row)
    
    # Print summary
    print("\n" + "="*60)
    print("BREAKDOWN BY CATEGORY")
    print("="*60)
    
    total_equity = 0
    for category, instruments in categories.items():
        count = len(instruments)
        print(f"{category}: {count}")
        if category == 'Equity Stocks':
            total_equity = count
    
    print(f"\n📊 Total Equity Stocks: {total_equity}")
    print(f"📊 Total Instruments: {len(df)}")
    
    # Show sample of equity stocks
    print("\n" + "="*60)
    print("SAMPLE EQUITY STOCKS")
    print("="*60)
    
    equity_stocks = categories['Equity Stocks']
    if equity_stocks:
        # Show first 20 equity stocks
        for i, stock in enumerate(equity_stocks[:20]):
            print(f"{i+1:2d}. {stock['tradingsymbol']:15s} - {stock['name']}")
        
        if len(equity_stocks) > 20:
            print(f"... and {len(equity_stocks) - 20} more equity stocks")
    
    # Check for major stocks
    print("\n" + "="*60)
    print("MAJOR STOCKS CHECK")
    print("="*60)
    
    major_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'ITC', 'SBIN', 
                   'WIPRO', 'BHARTIARTL', 'AXISBANK', 'KOTAKBANK', 'ASIANPAINT', 'MARUTI']
    
    found_stocks = []
    missing_stocks = []
    
    for stock in major_stocks:
        stock_data = df[df['tradingsymbol'] == stock]
        if not stock_data.empty:
            found_stocks.append(stock)
            print(f"✅ {stock}: {stock_data.iloc[0]['name']}")
        else:
            missing_stocks.append(stock)
            print(f"❌ {stock}: Not found")
    
    print(f"\n✅ Found: {len(found_stocks)} major stocks")
    print(f"❌ Missing: {len(missing_stocks)} major stocks")
    
    return df, categories

if __name__ == "__main__":
    df, categories = analyze_nse_data() 
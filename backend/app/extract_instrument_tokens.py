import pandas as pd

def extract_instrument_tokens():
    """
    Extract instrument tokens and trading symbols from cleaned NSE data
    """
    print("="*60)
    print("EXTRACTING INSTRUMENT TOKENS")
    print("="*60)
    
    # Load the cleaned data
    try:
        df = pd.read_csv('nse_cleaned_stocks.csv')
        print(f"Loaded {len(df)} records from nse_cleaned_stocks.csv")
    except FileNotFoundError:
        print("Error: nse_cleaned_stocks.csv not found!")
        return None
    
    # Filter for equity stocks only (exclude ETFs, government securities, treasury bills)
    print("\nFiltering for equity stocks only...")
    
    equity_stocks = df[
        ~df['tradingsymbol'].str.contains('IETF|GS|TB', na=False) &
        ~df['name'].str.contains('ETF|GOVERNMENT|TBILL', case=False, na=False)
    ]
    
    print(f"Found {len(equity_stocks)} equity stocks")
    
    # Select only relevant columns
    instrument_tokens = equity_stocks[['instrument_token', 'tradingsymbol', 'name']].copy()
    
    # Sort by trading symbol
    instrument_tokens = instrument_tokens.sort_values('tradingsymbol').reset_index(drop=True)
    
    # Save to CSV
    output_file = 'instrument_tokens.csv'
    instrument_tokens.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved {len(instrument_tokens)} instrument tokens to '{output_file}'")
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE INSTRUMENT TOKENS")
    print("="*60)
    print(instrument_tokens.head(10))
    
    # Show statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Total equity stocks: {len(instrument_tokens)}")
    print(f"Unique instrument tokens: {instrument_tokens['instrument_token'].nunique()}")
    print(f"Unique trading symbols: {instrument_tokens['tradingsymbol'].nunique()}")
    
    # Check for major stocks
    print("\n" + "="*60)
    print("MAJOR STOCKS VERIFICATION")
    print("="*60)
    
    major_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'ITC', 'SBIN']
    
    for stock in major_stocks:
        stock_data = instrument_tokens[instrument_tokens['tradingsymbol'] == stock]
        if not stock_data.empty:
            token = stock_data.iloc[0]['instrument_token']
            name = stock_data.iloc[0]['name']
            print(f"✅ {stock}: Token={token}, Name={name}")
        else:
            print(f"❌ {stock}: Not found")
    
    print(f"\n📁 File saved as: {output_file}")
    print(f"📊 Total equity stocks: {len(instrument_tokens)}")
    
    return instrument_tokens

if __name__ == "__main__":
    tokens_df = extract_instrument_tokens()
    
    if tokens_df is not None:
        print(f"\n✅ Successfully extracted instrument tokens!")
        print(f"📊 Total equity stocks: {len(tokens_df)}")
        print(f"💾 Saved to: instrument_tokens.csv")
    else:
        print("❌ Extraction failed!") 
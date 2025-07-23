import pandas as pd
import numpy as np

def clean_nse_instruments():
    """
    Clean NSE instruments data to get only tradable equity stocks
    """
    print("="*60)
    print("NSE INSTRUMENTS DATA CLEANING")
    print("="*60)
    
    # Load the unclean data
    try:
        df = pd.read_csv('nse_instruments_unclean.csv')
        print(f"Loaded {len(df)} records from nse_instruments_unclean.csv")
    except FileNotFoundError:
        print("Error: nse_instruments_unclean.csv not found!")
        return None
    
    # Step 1: Basic filtering for equity stocks
    print("\nStep 1: Filtering for equity stocks...")
    equity_stocks = df[
        (df['segment'] == 'NSE') & 
        (df['instrument_type'] == 'EQ') &
        (df['expiry'].isna() | (df['expiry'] == ''))  # No expiry = cash market
    ]
    print(f"After equity filter: {len(equity_stocks)} stocks")
    
    # Step 2: Remove duplicates
    print("\nStep 2: Removing duplicates...")
    equity_stocks = equity_stocks.drop_duplicates(subset=['tradingsymbol'])
    print(f"After removing duplicates: {len(equity_stocks)} stocks")
    
    # Step 3: Remove stocks with invalid data
    print("\nStep 3: Removing invalid data...")
    equity_stocks = equity_stocks[
        (equity_stocks['tradingsymbol'].notna()) &
        (equity_stocks['tradingsymbol'] != '') &
        (equity_stocks['name'].notna()) &
        (equity_stocks['name'] != '') &
        (equity_stocks['instrument_token'].notna())
    ]
    print(f"After removing invalid data: {len(equity_stocks)} stocks")
    
    # Step 4: Remove stocks with zero lot size (non-tradable)
    print("\nStep 4: Removing non-tradable stocks...")
    equity_stocks = equity_stocks[equity_stocks['lot_size'] > 0]
    print(f"After removing zero lot size: {len(equity_stocks)} stocks")
    
    # Step 5: Remove stocks with invalid tick sizes
    print("\nStep 5: Removing invalid tick sizes...")
    equity_stocks = equity_stocks[equity_stocks['tick_size'] > 0]
    print(f"After removing invalid tick size: {len(equity_stocks)} stocks")
    
    # Step 6: Remove bonds and government securities (SDL, SGB, etc.)
    print("\nStep 6: Removing bonds and government securities...")
    # Remove SDL (State Development Loans)
    equity_stocks = equity_stocks[~equity_stocks['tradingsymbol'].str.contains('SDL', na=False)]
    # Remove SGB (Sovereign Gold Bonds)
    equity_stocks = equity_stocks[~equity_stocks['tradingsymbol'].str.contains('SGB', na=False)]
    # Remove NHAI bonds
    equity_stocks = equity_stocks[~equity_stocks['tradingsymbol'].str.contains('NHAI', na=False)]
    # Remove IRFC bonds
    equity_stocks = equity_stocks[~equity_stocks['tradingsymbol'].str.contains('IRFC', na=False)]
    # Remove other bond patterns
    equity_stocks = equity_stocks[~equity_stocks['tradingsymbol'].str.contains('-SG$', na=False)]
    equity_stocks = equity_stocks[~equity_stocks['tradingsymbol'].str.contains('-GB$', na=False)]
    equity_stocks = equity_stocks[~equity_stocks['tradingsymbol'].str.contains('-N\d+$', na=False)]
    print(f"After removing bonds: {len(equity_stocks)} stocks")
    
    # Step 7: Remove stocks with suspicious names (likely bonds/securities)
    print("\nStep 7: Removing suspicious entries...")
    suspicious_patterns = [
        'SDL', 'SGB', 'NHAI', 'IRFC', 'GOLDBOND', 'GOLD BOND',
        'SOVEREIGN', 'GOVERNMENT', 'BOND', 'SECURITY'
    ]
    
    for pattern in suspicious_patterns:
        equity_stocks = equity_stocks[
            ~equity_stocks['name'].str.contains(pattern, case=False, na=False)
        ]
    print(f"After removing suspicious entries: {len(equity_stocks)} stocks")
    
    # Step 8: Sort and reset index
    print("\nStep 8: Sorting and organizing...")
    equity_stocks = equity_stocks.sort_values('tradingsymbol').reset_index(drop=True)
    
    # Step 9: Select relevant columns
    print("\nStep 9: Selecting relevant columns...")
    cleaned_stocks = equity_stocks[[
        'instrument_token',
        'tradingsymbol', 
        'name',
        'lot_size',
        'tick_size'
    ]].copy()
    
    # Step 10: Final summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Original data: {len(df)} records")
    print(f"Final cleaned stocks: {len(cleaned_stocks)} stocks")
    print(f"Removed: {len(df) - len(cleaned_stocks)} records")
    print(f"Removal rate: {((len(df) - len(cleaned_stocks)) / len(df) * 100):.1f}%")
    
    # Step 11: Save cleaned data
    print("\nStep 11: Saving cleaned data...")
    cleaned_stocks.to_csv('nse_cleaned_stocks.csv', index=False)
    print("Cleaned data saved to 'nse_cleaned_stocks.csv'")
    
    # Step 12: Display sample
    print("\n" + "="*60)
    print("SAMPLE OF CLEANED STOCKS")
    print("="*60)
    print(cleaned_stocks.head(10))
    
    # Step 13: Basic statistics
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(f"Total tradable stocks: {len(cleaned_stocks)}")
    print(f"Unique lot sizes: {cleaned_stocks['lot_size'].nunique()}")
    if len(cleaned_stocks) > 0:
        print(f"Most common lot size: {cleaned_stocks['lot_size'].mode().iloc[0]}")
    print(f"Unique tick sizes: {cleaned_stocks['tick_size'].nunique()}")
    
    # Step 14: Show lot size distribution
    print("\n" + "="*60)
    print("LOT SIZE DISTRIBUTION")
    print("="*60)
    lot_size_counts = cleaned_stocks['lot_size'].value_counts().head(10)
    for lot_size, count in lot_size_counts.items():
        print(f"Lot size {lot_size}: {count} stocks")
    
    # Step 15: Show tick size distribution
    print("\n" + "="*60)
    print("TICK SIZE DISTRIBUTION")
    print("="*60)
    tick_size_counts = cleaned_stocks['tick_size'].value_counts().head(10)
    for tick_size, count in tick_size_counts.items():
        print(f"Tick size {tick_size}: {count} stocks")
    
    print("\n" + "="*60)
    print("CLEANING COMPLETE!")
    print("="*60)
    
    return cleaned_stocks

if __name__ == "__main__":
    cleaned_data = clean_nse_instruments()
    
    if cleaned_data is not None:
        print(f"\n✅ Successfully cleaned NSE data!")
        print(f"📊 Total tradable stocks: {len(cleaned_data)}")
        print(f"💾 Saved to: nse_cleaned_stocks.csv")
        
        # Show some popular stocks
        popular_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'ITC', 'SBIN']
        print(f"\n🔍 Checking popular stocks:")
        for stock in popular_stocks:
            stock_info = cleaned_data[cleaned_data['tradingsymbol'] == stock]
            if not stock_info.empty:
                print(f"✅ {stock}: {stock_info.iloc[0]['name']}")
            else:
                print(f"❌ {stock}: Not found in cleaned data")
    else:
        print("❌ Cleaning failed!") 
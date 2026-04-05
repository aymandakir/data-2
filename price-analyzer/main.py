import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration
TICKERS = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA']
MONTHS = 6
OUTPUT_DIR = 'output'

def fetch_data(tickers, months=6):
    """Fetch 6 months of historical OHLCV data for given tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    data = {}
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                data[ticker] = df
                print(f"  ✓ Downloaded {len(df)} records for {ticker}")
            else:
                print(f"  ✗ No data found for {ticker}")
        except Exception as e:
            print(f"  ✗ Error fetching {ticker}: {e}")
    
    return data

def calculate_metrics(df):
    """Calculate all required metrics for a single ticker."""
    close = df['Close']
    volume = df['Volume']
    
    # Calculate SMAs
    sma_7 = close.rolling(window=7).mean()
    sma_30 = close.rolling(window=30).mean()
    
    # Daily percentage change
    daily_pct_change = close.pct_change() * 100
    
    # Total return %
    start_price = close.iloc[0]
    end_price = close.iloc[-1]
    total_return = ((end_price - start_price) / start_price) * 100
    
    # Max and Min price
    max_price = close.max()
    min_price = close.min()
    
    # Volatility (std dev of daily % change)
    volatility = daily_pct_change.std()
    
    return {
        'sma_7': sma_7,
        'sma_30': sma_30,
        'daily_pct_change': daily_pct_change,
        'total_return': total_return,
        'start_price': start_price,
        'end_price': end_price,
        'max_price': max_price,
        'min_price': min_price,
        'volatility': volatility,
        'volume': volume
    }

def create_charts(data, metrics_dict, output_path):
    """Create 2x2 matplotlib dashboard with price charts and volume."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (ticker, df) in enumerate(data.items()):
        ax = axes[idx]
        metrics = metrics_dict[ticker]
        
        # Close price line (solid blue)
        ax.plot(df.index, df['Close'], 'b-', label='Close', linewidth=1.5)
        
        # 7-day MA (dashed orange)
        ax.plot(df.index, metrics['sma_7'], '--', color='orange', label='7-day MA', linewidth=1.2)
        
        # 30-day MA (dotted green)
        ax.plot(df.index, metrics['sma_30'], ':', color='green', label='30-day MA', linewidth=1.2)
        
        # Volume on secondary y-axis - use stem plot instead of bar for cleaner display
        ax2 = ax.twinx()
        volume_list = [float(v) for v in metrics['volume'].values]
        ax2.fill_between(range(len(df.index)), volume_list, alpha=0.3, color='gray', label='Volume')
        ax2.set_ylabel('Volume', color='gray', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Set x-axis to use dates but hide for cleaner look
        ax2.set_xticks([])
        
        # Formatting
        ax.set_title(f'{ticker} - Price Analysis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Price ($)', fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Chart saved to: {output_path}")

def print_summary_table(metrics_dict):
    """Print a clean summary table in the terminal."""
    print("\n" + "=" * 80)
    print("📈 CRYPTO & STOCK PRICE ANALYZER")
    print("=" * 80)
    print(f"{'Ticker':<10} | {'Start Price':>12} | {'End Price':>12} | {'Return %':>10} | {'Max':>10} | {'Min':>10} | {'Volatility':>10}")
    print("-" * 80)
    
    for ticker, metrics in metrics_dict.items():
        # Extract scalar values from pandas Series
        start_price = float(metrics['start_price'])
        end_price = float(metrics['end_price'])
        total_return = float(metrics['total_return'])
        max_price = float(metrics['max_price'])
        min_price = float(metrics['min_price'])
        volatility = float(metrics['volatility'])
        
        # Format numbers nicely
        start_str = f"${start_price:,.2f}"
        end_str = f"${end_price:,.2f}"
        return_str = f"{total_return:+.2f}%"
        max_str = f"${max_price:,.2f}"
        min_str = f"${min_price:,.2f}"
        vol_str = f"{volatility:.2f}%"
        
        print(f"{ticker:<10} | {start_str:>12} | {end_str:>12} | {return_str:>10} | {max_str:>10} | {min_str:>10} | {vol_str:>10}")
    
    print("=" * 80)

def export_summary_csv(metrics_dict, output_path):
    """Export summary metrics to CSV."""
    rows = []
    for ticker, metrics in metrics_dict.items():
        rows.append({
            'Ticker': ticker,
            'Start Price': float(metrics['start_price']),
            'End Price': float(metrics['end_price']),
            'Return %': float(metrics['total_return']),
            'Max Price': float(metrics['max_price']),
            'Min Price': float(metrics['min_price']),
            'Volatility': float(metrics['volatility'])
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n📁 Summary exported to: {output_path}")

def main():
    """Main function to run the price analyzer."""
    print("Price Analyzer starting...\n")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Fetch data
    print("=" * 50)
    print("STEP 1: Fetching Data")
    print("=" * 50)
    data = fetch_data(TICKERS, MONTHS)
    
    if not data:
        print("No data fetched. Exiting.")
        return
    
    # Step 2: Calculate metrics
    print("\n" + "=" * 50)
    print("STEP 2: Calculating Metrics")
    print("=" * 50)
    metrics_dict = {}
    for ticker, df in data.items():
        print(f"Calculating metrics for {ticker}...")
        metrics_dict[ticker] = calculate_metrics(df)
    
    # Step 3: Create charts
    print("\n" + "=" * 50)
    print("STEP 3: Creating Charts")
    print("=" * 50)
    chart_path = os.path.join(OUTPUT_DIR, 'price_analysis.png')
    create_charts(data, metrics_dict, chart_path)
    
    # Step 4: Print summary table
    print_summary_table(metrics_dict)
    
    # Step 5: Export to CSV
    print("\n" + "=" * 50)
    print("STEP 5: Exporting Data")
    print("=" * 50)
    csv_path = os.path.join(OUTPUT_DIR, 'summary.csv')
    export_summary_csv(metrics_dict, csv_path)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
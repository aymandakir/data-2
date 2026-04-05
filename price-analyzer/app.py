import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Crypto & Stock Price Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styled cards
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-title { font-size: 2.5rem; font-weight: 700; color: #ffffff; text-align: center; padding: 1rem 0; margin-bottom: 2rem; }
    .kpi-card { background: linear-gradient(135deg, #1e1e2f 0%, #262638 100%); border-radius: 12px; padding: 1.25rem; border: 1px solid #3d3d5c; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }
    .kpi-ticker { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; }
    .kpi-price { font-size: 1.75rem; font-weight: 600; color: #e0e0e0; margin-bottom: 0.25rem; }
    .kpi-delta { font-size: 1rem; font-weight: 500; }
    .delta-positive { color: #4caf50; }
    .delta-negative { color: #f44336; }
    .section-header { font-size: 1.25rem; font-weight: 600; color: #ffffff; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #3d3d5c; }
    .insight-box { padding: 1rem; border-radius: 8px; margin: 0.75rem 0; font-weight: 500; }
    .insight-best { background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%); border-left: 4px solid #4caf50; }
    .insight-worst { background: linear-gradient(135deg, #5c1a1a 0%, #7f2d2d 100%); border-left: 4px solid #f44336; }
    .insight-volatile { background: linear-gradient(135deg, #5c4018 0%, #7f5a18 100%); border-left: 4px solid #ff9800; }
    .insight-signal { background: linear-gradient(135deg, #1a1a5c 0%, #2d2d7f 100%); border-left: 4px solid #5c95ff; }
    .spacer-20 { height: 20px; }
    </style>
""", unsafe_allow_html=True)

DEFAULT_TICKERS = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA']
MONTHS = 6

@st.cache_data(ttl=3600)
def fetch_data(tickers, months=6):
    """Fetch historical data for given tickers with error handling."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    data = {}
    errors = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is None or df.empty:
                errors.append(ticker)
            else:
                data[ticker] = df
        except Exception as e:
            errors.append(ticker)
    
    if errors:
        # Show errors after data is returned (not during)
        pass
    
    return data, errors

def calculate_metrics(df):
    """Calculate all required metrics for a ticker."""
    if df is None or df.empty:
        return None
    
    try:
        close = df['Close']
        volume = df['Volume']
        
        sma_7 = close.rolling(window=7).mean()
        sma_30 = close.rolling(window=30).mean()
        daily_pct_change = close.pct_change() * 100
        
        start_price = float(close.iloc[0])
        end_price = float(close.iloc[-1])
        total_return = ((end_price - start_price) / start_price) * 100
        max_price = float(close.max())
        min_price = float(close.min())
        volatility = float(daily_pct_change.std())
        
        return {
            'sma_7': sma_7, 'sma_30': sma_30,
            'start_price': start_price, 'end_price': end_price,
            'total_return': total_return, 'max_price': max_price,
            'min_price': min_price, 'volatility': volatility,
            'volume': volume, 'close': close
        }
    except Exception:
        return None

def create_chart(data, ticker, metrics):
    """Create matplotlib chart for a single ticker."""
    df = data[ticker]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    
    ax.plot(df.index, df['Close'], 'b-', label='Close', linewidth=1.5)
    ax.plot(df.index, metrics['sma_7'], '--', color='orange', label='7-day MA', linewidth=1.2)
    ax.plot(df.index, metrics['sma_30'], ':', color='green', label='30-day MA', linewidth=1.2)
    
    ax2 = ax.twinx()
    volume_list = [float(v) for v in metrics['volume'].values]
    ax2.fill_between(range(len(df.index)), volume_list, alpha=0.3, color='gray', label='Volume')
    ax2.set_ylabel('Volume', color='gray', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_xticks([])
    
    ax.set_title(f'{ticker} - Price Analysis', fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Date', fontsize=9, color='white')
    ax.set_ylabel('Price ($)', fontsize=9, color='white')
    ax.legend(loc='upper left', fontsize=8, facecolor='#1e1e2f', labelcolor='white')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('gray')
    
    return fig

def create_comparison_chart(data, tickers):
    """Create normalized comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, ticker in enumerate(tickers):
        if ticker in data:
            df = data[ticker]
            close = df['Close']
            normalized = (close / close.iloc[0]) * 100
            ax.plot(df.index, normalized, label=ticker, linewidth=2, color=colors[idx % len(colors)])
    
    ax.set_title('Normalized Price Comparison (Base = 100)', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Date', fontsize=10, color='white')
    ax.set_ylabel('Normalized Price', fontsize=10, color='white')
    ax.legend(loc='best', facecolor='#1e1e2f', labelcolor='white')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('gray')
    
    return fig

def render_kpi_card(ticker, current_price, delta, delta_pct):
    """Render a styled KPI card."""
    delta_class = "delta-positive" if delta >= 0 else "delta-negative"
    delta_icon = "▲" if delta >= 0 else "▼"
    
    html = f"""
    <div class="kpi-card">
        <div class="kpi-ticker">{ticker}</div>
        <div class="kpi-price">${current_price:,.2f}</div>
        <div class="kpi-delta {delta_class}">{delta_icon} {abs(delta_pct):.2f}%</div>
    </div>
    """
    return html

def main():
    """Main Streamlit app."""
    
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Custom Compare", "Auto Insights"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("This app analyzes cryptocurrency and stock prices using 6 months of historical data.")
    
    # Fetch default data
    data, errors = fetch_data(DEFAULT_TICKERS, MONTHS)
    
    # Show errors for failed tickers
    if errors:
        for ticker in errors:
            st.error(f"Failed to fetch data for {ticker}")
    
    if not data:
        st.error("Failed to fetch data for all tickers. Please check your internet connection.")
        return
    
    # Calculate metrics for all tickers (skip failed ones)
    metrics_dict = {}
    for ticker, df in data.items():
        metrics = calculate_metrics(df)
        if metrics is not None:
            metrics_dict[ticker] = metrics
    
    # Check if we have valid metrics
    if not metrics_dict:
        st.error("Failed to calculate metrics for any ticker.")
        return
    
    # Show which tickers were loaded
    st.success(f"Loaded data for: {', '.join(metrics_dict.keys())}")
    
    # ===== DASHBOARD PAGE =====
    if page == "Dashboard":
        st.markdown('<div class="main-title">📈 Crypto & Stock Price Analyzer</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📊 Market Overview (6 Month Performance)</div>', unsafe_allow_html=True)
        
        cols = st.columns(min(4, len(metrics_dict)))
        for idx, (ticker, metrics) in enumerate(metrics_dict.items()):
            with cols[idx]:
                html = render_kpi_card(
                    ticker, metrics['end_price'],
                    metrics['end_price'] - metrics['start_price'],
                    metrics['total_return']
                )
                st.markdown(html, unsafe_allow_html=True)
        
        st.markdown('<div class="spacer-20"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📈 Price Charts with Moving Averages</div>', unsafe_allow_html=True)
        
        chart_cols = st.columns(2)
        for idx, (ticker, metrics) in enumerate(metrics_dict.items()):
            with chart_cols[idx % 2]:
                fig = create_chart(data, ticker, metrics)
                st.pyplot(fig)
                st.markdown('<div class="spacer-20"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📋 Data Summary</div>', unsafe_allow_html=True)
        
        rows = []
        for ticker, metrics in metrics_dict.items():
            rows.append({
                'Ticker': ticker,
                'Start Price': f"${metrics['start_price']:,.2f}",
                'End Price': f"${metrics['end_price']:,.2f}",
                'Return %': f"{metrics['total_return']:+.2f}%",
                'Max': f"${metrics['max_price']:,.2f}",
                'Min': f"${metrics['min_price']:,.2f}",
                'Volatility': f"{metrics['volatility']:.2f}%"
            })
        
        df_display = pd.DataFrame(rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # ===== CUSTOM COMPARE PAGE =====
    elif page == "Custom Compare":
        st.markdown('<div class="main-title">🔍 Custom Compare</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown("### ⚙️ Compare Settings")
        
        available_tickers = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'AMD']
        selected_tickers = st.sidebar.multiselect(
            "Select tickers to compare",
            available_tickers,
            default=['MSFT', 'NVDA']
        )
        
        months_range = st.sidebar.slider("Timeframe (months)", 1, 24, 6)
        
        if selected_tickers:
            compare_data, compare_errors = fetch_data(selected_tickers, months_range)
            
            if compare_errors:
                for ticker in compare_errors:
                    st.error(f"Failed to fetch data for {ticker}")
            
            if compare_data:
                # Calculate metrics only for successful tickers
                compare_metrics = {}
                for ticker, df in compare_data.items():
                    m = calculate_metrics(df)
                    if m is not None:
                        compare_metrics[ticker] = m
                
                if compare_metrics:
                    st.markdown('<div class="section-header">📈 Normalized Price Comparison</div>', unsafe_allow_html=True)
                    st.markdown("*All prices normalized to start at 100 for direct comparison*")
                    
                    fig = create_comparison_chart(compare_data, list(compare_metrics.keys()))
                    st.pyplot(fig)
                    
                    st.markdown('<div class="section-header">📋 Comparison Metrics</div>', unsafe_allow_html=True)
                    
                    compare_rows = []
                    for ticker, m in compare_metrics.items():
                        compare_rows.append({
                            'Ticker': ticker,
                            'Return %': f"{m['total_return']:+.2f}%",
                            'Volatility': f"{m['volatility']:.2f}%",
                            'Max': f"${m['max_price']:,.2f}",
                            'Min': f"${m['min_price']:,.2f}"
                        })
                    
                    compare_df = pd.DataFrame(compare_rows)
                    st.dataframe(compare_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No data available for selected tickers.")
            else:
                st.warning("No data available for selected tickers.")
        else:
            st.info("Please select tickers from the sidebar to compare.")
    
    # ===== AUTO INSIGHTS PAGE =====
    elif page == "Auto Insights":
        st.markdown('<div class="main-title">💡 Auto Insights</div>', unsafe_allow_html=True)
        st.markdown("Automated analysis based on the available tickers")
        
        if not metrics_dict:
            st.error("No data available for insights.")
            return
        
        best_ticker = max(metrics_dict.items(), key=lambda x: x[1]['total_return'])
        worst_ticker = min(metrics_dict.items(), key=lambda x: x[1]['total_return'])
        most_volatile = max(metrics_dict.items(), key=lambda x: x[1]['volatility'])
        
        signals = []
        for ticker, metrics in metrics_dict.items():
            if len(metrics['sma_7'].dropna()) > 0 and len(metrics['sma_30'].dropna()) > 0:
                sma7_current = float(metrics['sma_7'].iloc[-1])
                sma30_current = float(metrics['sma_30'].iloc[-1])
                
                if sma7_current > sma30_current:
                    signals.append(f"🐂 Bullish: 7-day MA above 30-day MA for {ticker}")
                else:
                    signals.append(f"🐻 Bearish: 7-day MA below 30-day MA for {ticker}")
        
        st.markdown("---")
        
        st.markdown(f"""
        <div class="insight-box insight-best">
            <strong>🏆 Best Performer:</strong> {best_ticker[0]}<br>
            Return: {best_ticker[1]['total_return']:+.2f}% | End Price: ${best_ticker[1]['end_price']:,.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box insight-worst">
            <strong>📉 Worst Performer:</strong> {worst_ticker[0]}<br>
            Return: {worst_ticker[1]['total_return']:+.2f}% | End Price: ${worst_ticker[1]['end_price']:,.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box insight-volatile">
            <strong>⚡ Most Volatile:</strong> {most_volatile[0]}<br>
            Volatility: {most_volatile[1]['volatility']:.2f}% | Price Range: ${most_volatile[1]['min_price']:,.2f} - ${most_volatile[1]['max_price']:,.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📊 Moving Average Signals</div>', unsafe_allow_html=True)
        for signal in signals:
            st.markdown(f"<div class='insight-box insight-signal'>{signal}</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📋 Complete Metrics</div>', unsafe_allow_html=True)
        
        rows = []
        for ticker, metrics in metrics_dict.items():
            rows.append({
                'Ticker': ticker,
                'Return %': f"{metrics['total_return']:+.2f}%",
                'Volatility': f"{metrics['volatility']:.2f}%",
                'Max': f"${metrics['max_price']:,.2f}",
                'Min': f"{metrics['min_price']:,.2f}",
                'Current': f"${metrics['end_price']:,.2f}"
            })
        
        df_insights = pd.DataFrame(rows)
        st.dataframe(df_insights, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

# ========== Data Fetching ==========
def fetch_data(symbols, period="6mo", interval="1d"):
    all_data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)  # Use adjusted close
            if not df.empty:
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                df['Symbol'] = symbol
                all_data[symbol] = df
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
    return all_data

# ========== Fetch Nifty for RS calculation ==========
def fetch_nifty(period="6mo", interval="1d"):
    nifty = yf.download("^NSEI", period=period, interval=interval, auto_adjust=True)
    nifty['Return'] = nifty['Close'].pct_change()
    return nifty

nifty = fetch_nifty()

# ========== Relative Strength Calculation ==========
def relative_strength(stock_df, nifty_df):
    stock_df = stock_df.copy()
    stock_df['Return'] = stock_df['Close'].pct_change()
    combined = stock_df[['Return']].join(nifty_df['Return'], lsuffix='_stock', rsuffix='_nifty').dropna()
    rs_series = (1 + combined['Return_stock']).cumprod() / (1 + combined['Return_nifty']).cumprod()
    return rs_series.iloc[-1]

# ========== Metrics Computation ==========
def compute_behavior_metrics(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, got {type(df)} instead.")

    if df.empty or df.shape[0] < 30:
        raise ValueError(f"DataFrame is empty or too short. Shape: {df.shape}")

    df = df.copy()
    required_cols = ['High', 'Low', 'Close', 'Volume']

    for col in required_cols:
        if col not in df.columns.tolist():
            raise ValueError(f"Missing required column: '{col}'. Available: {df.columns.tolist()}")
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=required_cols, inplace=True)

    if len(df) < 30:
        raise ValueError("Insufficient data after cleaning for metric computation.")

    # ATR %
    df['ATR'] = df['High'] - df['Low']
    df['ATR%'] = df['ATR'] / df['Close']

    # Bollinger Band Width %
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    df['BB_Width%'] = (bb_upper - bb_lower) / df['Close']

    # Inside Bars %
    df['Inside'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    inside_bar_pct = df['Inside'].sum() / len(df)

    # RSI Slope
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    rsi_slope = df['RSI'].diff().mean()

    # Price Slope (last 30 days)
    y = df['Close'].dropna().values
    x = np.arange(len(y))
    price_slope = 0
    if len(x) >= 30:
        coeffs = np.polyfit(x[-30:], y[-30:], 1)
        price_slope = coeffs[0] / y[-1] if y[-1] != 0 else 0

    # Volume Confirmation - volume > 1.5x 20-day avg volume (new condition)
    volume_mean = df['Volume'].rolling(window=20).mean()
    volume_confirm = df['Volume'].iloc[-1] > 1.5 * volume_mean.iloc[-1]

    # Reversal Signal: RSI Divergence
    recent_rsi = df['RSI'].iloc[-1]
    prev_rsi = df['RSI'].iloc[-15:-5].mean() if len(df['RSI']) >= 15 else recent_rsi
    rsi_delta = recent_rsi - prev_rsi
    rsi_reversal = abs(rsi_delta) > 10  # Only strong RSI moves qualify

    # Moving Averages for trend filter
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    return {
        'ATR%': df['ATR%'].mean(),
        'BB_Width%': df['BB_Width%'].mean(),
        'Inside_Bar_%': inside_bar_pct,
        'RSI_Slope': rsi_slope,
        'Price_Slope': price_slope,
        'Volume_Confirm': volume_confirm,
        'RSI_Reversal': rsi_reversal,
        'SMA50': df['SMA50'].iloc[-1],
        'SMA200': df['SMA200'].iloc[-1],
        'Close': df['Close'].iloc[-1]
    }

# ========== Dynamic Classification ==========
def classify_behavior(metrics_df):
    atr_thresh = metrics_df['ATR%'].quantile(0.7)
    bb_thresh = metrics_df['BB_Width%'].quantile(0.7)
    rsi_thresh = metrics_df['RSI_Slope'].quantile(0.7)
    slope_thresh = metrics_df['Price_Slope'].quantile(0.7)

    classified = []
    for _, row in metrics_df.iterrows():
        # Add moving average trend confirmation filter for Uptrend:
        ma_trend = (row['Close'] > row['SMA50'] > row['SMA200'])
        vol_spike = row['Volume_Confirm']
        rs_value = row.get('RS', 1)  # default 1 (neutral) if not computed

        if row['Price_Slope'] > slope_thresh and row['RSI_Slope'] > rsi_thresh and vol_spike and ma_trend and rs_value > 1:
            behavior = "Uptrend"
        elif row['Price_Slope'] < -slope_thresh and row['RSI_Slope'] < -rsi_thresh and vol_spike and ma_trend==False:
            behavior = "Downtrend"
        elif row['RSI_Reversal'] and vol_spike:
            behavior = "Reversal"
        elif row['ATR%'] > atr_thresh and row['BB_Width%'] > bb_thresh and row['Inside_Bar_%'] > 0.25:
            behavior = "Noisy"
        else:
            behavior = "Choppy"
        classified.append(behavior)
    metrics_df['Behavior'] = classified
    return metrics_df

# ========== Forward Return Computation ==========
def compute_forward_returns(df, days=[3, 5, 10]):
    df = df.copy()
    for day in days:
        df[f'Forward_{day}d'] = df['Close'].shift(-day) / df['Close'] - 1
    return df

# ========== Full Pipeline ==========
def analyze_symbols(symbols):
    raw_data = fetch_data(symbols)
    results = []

    for symbol, df in raw_data.items():
        df = compute_forward_returns(df)
        try:
            metrics = compute_behavior_metrics(df)
        except Exception as e:
            print(f"Skipping {symbol} due to metric error: {e}")
            continue
        
        # Calculate RS vs Nifty and add to metrics
        rs_val = relative_strength(df, nifty)
        metrics['RS'] = rs_val
        
        avg_returns = {f'Avg_Forward_{d}d': df[f'Forward_{d}d'].mean() for d in [3, 5, 10]}
        results.append({"Symbol": symbol, **metrics, **avg_returns})

    results_df = pd.DataFrame(results)
    results_df = classify_behavior(results_df)
    return results_df, raw_data

# ========== Backtesting ==========
def backtest_strategy(results_df, raw_data):
    portfolio = []
    win_count = 0
    loss_count = 0
    returns_list = []

    for i, row in results_df.iterrows():
        symbol = row['Symbol']
        df = raw_data[symbol].copy()
        df = compute_forward_returns(df)

        # Find entry index: when the current behavior started - simplified as last 11 days from end
        entry_idx = len(df) - 11
        exit_idx = entry_idx + 10

        if exit_idx >= len(df):
            continue

        # Entry price and exit price
        entry_price = df['Close'].iloc[entry_idx]
        exit_price = df['Close'].iloc[exit_idx]
        ret = (exit_price - entry_price) / entry_price

        # Apply stop-loss (10%) for demo risk control example (not in original, but good practice)
        # Assuming daily close; stop loss if price drops 10% from entry on any day in holding period
        holding_prices = df['Close'].iloc[entry_idx:exit_idx+1]
        if (holding_prices < entry_price * 0.9).any():
            # simulate stop loss hit at first day below 90%
            sl_day = holding_prices[holding_prices < entry_price * 0.9].index[0]
            exit_price = df.loc[sl_day, 'Close']
            ret = (exit_price - entry_price) / entry_price

        # Track win/loss
        if ret > 0:
            win_count += 1
        else:
            loss_count += 1
        returns_list.append(ret)

        portfolio.append({
            'Symbol': symbol,
            'Behavior': row['Behavior'],
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Return': ret
        })

    portfolio_df = pd.DataFrame(portfolio)
    win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
    avg_return = np.mean(returns_list) if returns_list else 0
    max_drawdown = min(returns_list) if returns_list else 0

    print(f"Backtest results:\nWin Rate: {win_rate:.2%}\nAverage Return: {avg_return:.2%}\nMax Drawdown: {max_drawdown:.2%}")

    # T-test for Uptrend vs Downtrend forward 10-day returns (if enough samples)
    uptrend_returns = portfolio_df[portfolio_df['Behavior'] == 'Uptrend']['Return']
    downtrend_returns = portfolio_df[portfolio_df['Behavior'] == 'Downtrend']['Return']

    if len(uptrend_returns) > 5 and len(downtrend_returns) > 5:
        t_stat, p_val = ttest_ind(uptrend_returns, downtrend_returns, equal_var=False)
        print(f"T-test Uptrend vs Downtrend returns: t-stat={t_stat:.3f}, p-value={p_val:.3f}")

    return portfolio_df

# ========== Sample Universe ==========
universe = [
                
                 'SJVN.NS','PREMIERENE.NS', 'ATHERENERG.NS', 'HAL.NS', 'COFORGE.NS', 'AGIIL.NS', 'ZENSARTECH.NS', 
                'ABCAPITAL.NS', 'LTF.NS', 'INDIANHUME.NS', 'SKIPPER.NS', 'MANAPPURAM.NS', 'SUNDARMHLD.NS', 'VHL.NS', 'INDIANB.NS'
]

# ========== Run Analysis and Backtest ==========
results_df, raw_data = analyze_symbols(universe)
print(results_df[['Symbol', 'Behavior', 'ATR%', 'BB_Width%', 'RSI_Slope', 'Price_Slope', 'Volume_Confirm', 'RS', 'SMA50', 'SMA200']])

portfolio_df = backtest_strategy(results_df, raw_data)
print(portfolio_df)

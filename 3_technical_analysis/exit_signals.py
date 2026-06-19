import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from tabulate import tabulate


def exit_signal(symbol, entry_price, tp_hit_price, qty, lookback_days=60):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
    df.reset_index(inplace=True)
    df.dropna(inplace=True)

    # Ensure correct types and shapes
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)

    # Extract 1D close prices for indicators
    close = df['Close'].squeeze()  # Ensure Series, not DataFrame
    high = df['High'].squeeze()
    low = df['Low'].squeeze()


    # Indicators
    df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
    macd = ta.trend.MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close)
    df['ATR'] = atr.average_true_range()

    # Patterns
    def detect_bearish_engulfing(df):
        return (
            (df['Open'].shift(1) < df['Close'].shift(1)) &
            (df['Open'] > df['Close']) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1))
        )

    def detect_shooting_star(df):
    # Ensure Series, not DataFrames
        high = df['High']
        low = df['Low']
        close = df['Close']
        open_ = df['Open']

        body = abs(close - open_)
        upper_shadow = high - close.where(close > open_, open_)
        lower_shadow = open_.where(close > open_, close) - low

        # All are 1D Series, safe for comparison
        return (upper_shadow > 2 * body) & (lower_shadow < body)


    df['Bearish_Engulfing'] = detect_bearish_engulfing(df)
    df['Shooting_Star'] = detect_shooting_star(df)

    # Exit logic
    latest = df.iloc[-1]
    rsi_cond = latest['RSI'].item() > 65
    macd_cond = (latest['MACD'].item() > latest['MACD_signal'].item()) and (latest['MACD_hist'].item() > 0)
    bearish_candle = latest['Bearish_Engulfing'].item() or latest['Shooting_Star'].item()

    atr_factor = 1.5
    suggested_trailing_exit = tp_hit_price - (latest['ATR'].item() * atr_factor)

    if rsi_cond and macd_cond and not bearish_candle:
        signal = "HOLD"
        reason = "Strong momentum with MACD/RSI and no bearish candle detected."
        extension_potential = tp_hit_price + latest['ATR'].item() * 2
    else:
        signal = "EXIT"
        reason = "Bearish divergence or reversal pattern found."
        extension_potential = None


    result = {
        "Symbol": symbol,
        "Entry Price": entry_price,
        "TP Hit Price": tp_hit_price,
        "Suggested Exit Signal": signal,
        "Suggested Trailing Exit": round(suggested_trailing_exit, 2),
        "Max Extension Potential": round(extension_potential, 2) if extension_potential is not None else np.nan,
        "Reason": reason
    }

    return pd.DataFrame([result])


# Example usage
if __name__ == "__main__":
    from tabulate import tabulate
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # 🔁 List of stocks to check
    stock_list = [
        
        ("KIMS.NS", 662, 660, 42),
        ("PNCINFRA.NS", 307, 320, 43),

        # Add more tuples as (symbol, entry_price, tp_hit_price, qty)
    ]

    # 📊 Collect results
    results = []
    for symbol, entry_price, tp_hit_price, qty in stock_list:
        try:
            df = exit_signal(symbol, entry_price, tp_hit_price, qty)
            results.append(df)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # 🧾 Combine all rows
    if results:
        final_df = pd.concat(results, ignore_index=True)
        print(tabulate(final_df, headers="keys", tablefmt="pretty", showindex=False))
    else:
        print("No results to display.")



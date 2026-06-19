import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import ta
from scipy.signal import argrelextrema
import numpy as np
from ta.trend import ADXIndicator
import datetime
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from backtest import backtest_signals, apply_atr_risk
import pandas as pd
import os
import seaborn as sns

# Store results here
performance_log = []


def fetch_data(symbol, period="2y", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)

        # Check if data was returned
        if df is None or df.empty:
            print(f"⚠️ No data returned for symbol: {symbol}")
            return None

        # Flatten multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure essential columns exist
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ Missing required columns for {symbol}. Found: {df.columns.tolist()}")
            return None

        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None


#
def get_weekly_trend(symbol):
    df_weekly = yf.download(symbol, period="2y", interval="1wk", auto_adjust=False)

    # Fix multi-index column names if needed
    if isinstance(df_weekly.columns, pd.MultiIndex):
        df_weekly.columns = df_weekly.columns.get_level_values(0)

    df_weekly = df_weekly[['Close']].copy()
    df_weekly['MA20'] = df_weekly['Close'].rolling(window=20).mean()

    # ✅ Ensure both are Series before comparing
    close = df_weekly['Close']
    ma20 = df_weekly['MA20']
    df_weekly['Trend_Weekly'] = np.where(close > ma20, 'Uptrend', 'Downtrend')

    df_weekly = df_weekly.reset_index()[['Date', 'Trend_Weekly']]
    return df_weekly

#
def add_indicators(df):
    # Trend
    df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['MA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['Parabolic_SAR'] = ta.trend.psar_up(df['High'], df['Low'], df['Close'])

    # Momentum
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['Stoch_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()

    # Volume
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['VolumeMA5'] = df['Volume'].rolling(5).mean()

    # Volatility
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)

    return df
    
def add_anchored_vwap(df):
    df['Anchored_VWAP'] = np.nan
    anchor_index = df['Swing_Low'].first_valid_index()
    if anchor_index is not None:
        cum_vol = df.loc[anchor_index:, 'Volume'].cumsum()
        cum_vp = (df.loc[anchor_index:, 'Close'] * df.loc[anchor_index:, 'Volume']).cumsum()
        df.loc[anchor_index:, 'Anchored_VWAP'] = cum_vp / cum_vol
        return df
    

    # Simulate a DataFrame with no valid Swing_Low (all NaN)
    df_test = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=10),
    'Close': np.random.rand(10) * 100,
    'Volume': np.random.randint(1000, 10000, 10),
    'Swing_Low': [np.nan] * 10
    })

    # Apply the broken function to simulate the bug
    result = add_anchored_vwap(df_test)
    type(result)  # To see if it's None or a DataFrame



def detect_rsi_macd_divergence(df, lookback=5):
    df['RSI_Div'] = ""
    df['MACD_Div'] = ""

    for i in range(lookback, len(df)):
        # Price higher highs / lower lows
        price_curr = df.loc[i, 'Close']
        price_prev = df.loc[i - lookback, 'Close']
        rsi_curr = df.loc[i, 'RSI']
        rsi_prev = df.loc[i - lookback, 'RSI']
        macd_curr = df.loc[i, 'MACD']
        macd_prev = df.loc[i - lookback, 'MACD']

        # RSI divergence
        if price_curr > price_prev and rsi_curr < rsi_prev:
            df.loc[i, 'RSI_Div'] = "Bearish"
        elif price_curr < price_prev and rsi_curr > rsi_prev:
            df.loc[i, 'RSI_Div'] = "Bullish"

        # MACD divergence
        if price_curr > price_prev and macd_curr < macd_prev:
            df.loc[i, 'MACD_Div'] = "Bearish"
        elif price_curr < price_prev and macd_curr > macd_prev:
            df.loc[i, 'MACD_Div'] = "Bullish"

    return df

def detect_patterns(df):
    patterns = []
    for i in range(3, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        o, h, l, c, v, rsi = row[['Open','High','Low','Close','Volume','RSI']]
        prev_o, prev_c = prev['Open'], prev['Close']
        ma50, ma200 = row['MA50'], row['MA200']
        macd_hist, prev_hist = row['MACD_hist'], df.at[i-1, 'MACD_hist']
        vol_ok = v > df.at[i, 'VolumeMA5']
        in_up = ma50 > ma200
        in_down = ma50 < ma200
        body = abs(c - o)
        rng = h - l
        momentum_flip = (macd_hist > 0 and prev_hist < 0) or (macd_hist < 0 and prev_hist > 0)
        weekly_trend = row.get('Trend_Weekly', 'Uptrend')  # fallback default

        # ✅ First define pattern flags
        bull_engulf = in_down and (c > o > prev_c > prev_o) and vol_ok and rsi < 60 and momentum_flip
        bear_engulf = in_up and (c < o < prev_c < prev_o) and vol_ok and rsi > 40 and momentum_flip
        hammer = in_down and (body < 0.3 * rng) and ((min(o, c) - l) > body * 1.5) and vol_ok and rsi < 50 and momentum_flip
        doji = body < 0.1 * rng and vol_ok and momentum_flip

        # ✅ Continuation: Flags
        recent = df.iloc[i-3:i]
        flag_height = recent['High'].max() - recent['Low'].min()
        avg_range = (recent['High'] - recent['Low']).mean()
        tight_flag = flag_height < 0.25 * avg_range
        bull_flag = in_up and tight_flag and c > recent['High'].max() and vol_ok and momentum_flip
        bear_flag = in_down and tight_flag and c < recent['Low'].min() and vol_ok and momentum_flip

        # ✅ Now check with multi-timeframe confirmation
        pattern = ""
        if bull_engulf and weekly_trend == 'Uptrend':
            pattern = "Bullish Engulfing"
        elif bear_engulf and weekly_trend == 'Downtrend':
            pattern = "Bearish Engulfing"
        elif hammer and weekly_trend == 'Uptrend':
            pattern = "Hammer"
        elif doji:
            pattern = "Doji"
        elif bull_flag and weekly_trend == 'Uptrend':
            pattern = "Bull Flag"
        elif bear_flag and weekly_trend == 'Downtrend':
            pattern = "Bear Flag"

        patterns.append(pattern)

    for _ in range(3): patterns.insert(0, "")
    
    df['Pattern'] = patterns
    
    return df

#
def detect_trendlines(df, window=20):
    # Detect swing highs and lows
    df['Swing_High'] = df['High'][(df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])]
    df['Swing_Low'] = df['Low'][(df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])]
    return df
#
def detect_mss_and_order_blocks(df):
    df['MSS'] = ""
    df['Order_Block_Price'] = np.nan
    df['Order_Block_Type'] = ""

    last_swing_high = last_swing_low = None

    for i in range(1, len(df)):
        # Detect MSS
        if not np.isnan(df.loc[i, 'Swing_High']):
            if last_swing_low is not None and df.loc[i, 'High'] > last_swing_low:
                df.loc[i, 'MSS'] = "Bullish MSS"
                # Set Bullish Order Block (Last red candle before breakout)
                for j in range(i-3, i)[::-1]:
                    if df.loc[j, 'Close'] < df.loc[j, 'Open']:
                        df.loc[i, 'Order_Block_Price'] = df.loc[j, 'Low']
                        df.loc[i, 'Order_Block_Type'] = "Bullish"
                        break
            last_swing_high = df.loc[i, 'High']

        if not np.isnan(df.loc[i, 'Swing_Low']):
            if last_swing_high is not None and df.loc[i, 'Low'] < last_swing_high:
                df.loc[i, 'MSS'] = "Bearish MSS"
                # Set Bearish Order Block (Last green candle before breakdown)
                for j in range(i-3, i)[::-1]:
                    if df.loc[j, 'Close'] > df.loc[j, 'Open']:
                        df.loc[i, 'Order_Block_Price'] = df.loc[j, 'High']
                        df.loc[i, 'Order_Block_Type'] = "Bearish"
                        break
            last_swing_low = df.loc[i, 'Low']

    return df

def detect_fvg(df):
    df['FVG'] = ""
    for i in range(2, len(df)):
        prev1 = df.iloc[i - 2]
        prev2 = df.iloc[i - 1]
        curr = df.iloc[i]

        # Bullish FVG: gap between prev1 high and curr low
        if prev1['High'] < curr['Low']:
            df.loc[i, 'FVG'] = "Bullish FVG"

        # Bearish FVG: gap between curr high and prev1 low
        elif curr['High'] < prev1['Low']:
            df.loc[i, 'FVG'] = "Bearish FVG"
    return df

def add_adx(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=window).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=window).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=window).sum() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()

    df['ADX'] = adx
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    return df

def add_atr(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    df['ATR'] = atr
    return df

def is_choppy(df, i, adx_thresh=20, bb_thresh=0.02):
    if i < 1 or i >= len(df):
        return True
    adx = df.loc[i, 'ADX']
    bb_width = df.loc[i, 'BB_High'] - df.loc[i, 'BB_Low']
    price = df.loc[i, 'Close']

    if pd.isna(adx) or pd.isna(bb_width) or pd.isna(price):
        return True

    return (adx < adx_thresh) and ((bb_width / price) < bb_thresh)

def apply_risk_management(df, capital=50000, max_risk_per_trade=0.01):
    # Default values
    df['Stop_Loss'] = np.nan
    df['Target'] = np.nan
    df['Position_Size'] = 0.0
    df['Risk'] = 0.0
    df['Reward'] = 0.0
    df['R_R'] = 0.0
    df['Valid_Trade'] = False

    risk_amount = capital * max_risk_per_trade
    atr = df['ATR']

    # Buy Logic
    buy_mask = df['Signal'] == 'Buy'
    df.loc[buy_mask, 'Stop_Loss'] = df['Close'] - 1.5 * atr
    df.loc[buy_mask, 'Target'] = df['Close'] + 2.5 * atr
    df.loc[buy_mask, 'Risk'] = 1.5 * atr
    df.loc[buy_mask, 'Reward'] = 2.5 * atr

    # Sell Logic
    sell_mask = df['Signal'] == 'Sell'
    df.loc[sell_mask, 'Stop_Loss'] = df['Close'] + 1.5 * atr
    df.loc[sell_mask, 'Target'] = df['Close'] - 2.5 * atr
    df.loc[sell_mask, 'Risk'] = 1.5 * atr
    df.loc[sell_mask, 'Reward'] = 2.5 * atr

    # Position Sizing (based on capital and stop loss)
    df['Position_Size'] = np.where(df['Risk'] > 0, risk_amount / df['Risk'], 0)

    # Risk/Reward Ratio
    df['R_R'] = np.where(df['Risk'] > 0, df['Reward'] / df['Risk'], 0)

    # Flag valid trades with RR > 1.5
    df['Valid_Trade'] = df['R_R'] >= 1.5

    return df

def add_consolidation_flag(df):
    range_10 = df['High'].rolling(window=10).max() - df['Low'].rolling(window=10).min()
    range_pct = range_10 / df['Close']
    df['Consolidation'] = range_pct < 0.05  # Example threshold for consolidation
    return df

def generate_signals(df):
    df['Signal'] = ""
    df['Almost_Signal'] = ""

    for i in range(3, len(df)):
        if is_choppy(df, i):
            continue

        pattern = df.loc[i, 'Pattern']
        mss = df.loc[i, 'MSS']
        ob_type = df.loc[i, 'Order_Block_Type']
        ob_price = df.loc[i, 'Order_Block_Price']
        price = df.loc[i, 'Close']
        ma50, ma200 = df.loc[i, 'MA50'], df.loc[i, 'MA200']
        rsi = df.loc[i, 'RSI']
        macd_hist = df.loc[i, 'MACD_hist']
        prev_hist = df.loc[i-1, 'MACD_hist']
        trend = df.loc[i, 'Trend_Weekly']
        adx = df.loc[i, 'ADX']
        cci = df.loc[i, 'CCI']
        rsi_div = df.loc[i, 'RSI_Div']
        macd_div = df.loc[i, 'MACD_Div']
        vol = df.loc[i, 'Volume']
        vol_ma5 = df.loc[i, 'VolumeMA5']
        obv_trend_up = df.loc[i, 'OBV'] > df.loc[i - 1, 'OBV']
        obv_trend_down = df.loc[i, 'OBV'] < df.loc[i - 1, 'OBV']
        anchored_vwap = df.loc[i, 'Anchored_VWAP']

        recent_high = df['High'][i - 5:i].max()
        recent_low = df['Low'][i - 5:i].min()
        volume_spike = vol > 1.5 * vol_ma5
        price_breakout = price > recent_high or price < recent_low

        ma_convergence = abs(ma50 - ma200) / price < 0.01
        if ma_convergence:
            continue

        bb_width = df.loc[i, 'BB_High'] - df.loc[i, 'BB_Low']
        if bb_width / price < 0.03:
            continue

        strong_uptrend = adx > 25 and df.loc[i, '+DI'] > df.loc[i, '-DI']
        strong_downtrend = adx > 25 and df.loc[i, '-DI'] > df.loc[i, '+DI']

        # Signal clustering logic
        bull_count = 0
        if pattern in ["Bullish Engulfing", "Hammer", "Doji", "Bull Flag"]:
            bull_count += 1
        if macd_div == 'Bullish':
            bull_count += 1
        if rsi_div == 'Bullish':
            bull_count += 1
        if obv_trend_up:
            bull_count += 1
        if mss == 'Bullish MSS':
            bull_count += 1
        if df.loc[i, 'FVG'] == 'Bullish FVG':
            bull_count += 1
        if price > anchored_vwap:
            bull_count += 1

        bear_count = 0
        if pattern in ["Bearish Engulfing", "Bear Flag"]:
            bear_count += 1
        if macd_div == 'Bearish':
            bear_count += 1
        if rsi_div == 'Bearish':
            bear_count += 1
        if obv_trend_down:
            bear_count += 1
        if mss == 'Bearish MSS':
            bear_count += 1
        if df.loc[i, 'FVG'] == 'Bearish FVG':
            bear_count += 1
        if price < anchored_vwap:
            bear_count += 1

        if bull_count >= 4 and trend == 'Uptrend' and strong_uptrend and volume_spike and price_breakout:
            df.loc[i, 'Signal'] = "Buy"
        elif bear_count >= 4 and trend == 'Downtrend' and strong_downtrend and volume_spike and price_breakout:
            df.loc[i, 'Signal'] = "Sell"
        elif mss == "Bullish MSS" and ob_type == "Bullish" and price > ob_price:
            df.loc[i, 'Signal'] = "Buy"
        elif mss == "Bearish MSS" and ob_type == "Bearish" and price < ob_price:
            df.loc[i, 'Signal'] = "Sell"
        else:
            if pattern in ["Bullish Engulfing", "Hammer", "Bull Flag"] and ma50 >= ma200:
                df.loc[i, 'Almost_Signal'] = "Buy"
            elif pattern in ["Bearish Engulfing", "Bear Flag"] and ma50 <= ma200:
                df.loc[i, 'Almost_Signal'] = "Sell"
        
        # Holy Grail Buy
        if adx > 30 and price > ma50 and price > ma200:
            if df.loc[i-1, 'Close'] > df.loc[i-1, 'EMA20'] and price < df.loc[i, 'EMA20'] and price > df.loc[i-1, 'Low']:
                df.loc[i, 'Signal'] = "Buy"


    return df



def plot_signals(df, symbol):
    required_cols = ['Signal', 'Valid_Trade', 'R_R', 'Target', 'Stop_Loss', 'Close', 'Date', 'Score']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' is missing. Skipping signal plot.")
            return

    df['Date'] = pd.to_datetime(df['Date'])
    df['R_R'] = pd.to_numeric(df['R_R'], errors='coerce')
    df_signals = df[(df['Signal'].isin(['Buy', 'Sell'])) & (df['Valid_Trade']) & (df['R_R'] >= 1.49)].copy()

    if df_signals.empty:
        print("⚠️ No valid high-quality signals found.")
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df['Date'], df['Close'], label='Close Price', alpha=0.6, color='gray')
    shown_labels = set()

    for i, row in df_signals.iterrows():
        try:
            date = row['Date']
            x = mdates.date2num(date)
            price = row['Close']
            width = 0.8
            signal = row['Signal']
            score = row.get('Score', 0)

            color = 'green' if signal == 'Buy' else 'red'
            marker = '^' if signal == 'Buy' else 'v'
            ax.scatter(date, price, color=color, marker=marker, s=score * 10,
                       label=signal if signal not in shown_labels else "")
            ax.text(date, price + 0.5, f"{score:.1f}", fontsize=8, ha='center', color='black')
            shown_labels.add(signal)

            target = row['Target']
            stop = row['Stop_Loss']

            if pd.notna(target) and pd.notna(stop):
                if signal == 'Buy':
                    ax.add_patch(Rectangle((x - width/2, price), width, max(0, target - price),
                                           color='green', alpha=0.2, label='Target' if 'Target' not in shown_labels else ""))
                    ax.add_patch(Rectangle((x - width/2, stop), width, max(0, price - stop),
                                           color='red', alpha=0.2, label='Stop Loss' if 'Stop Loss' not in shown_labels else ""))
                else:
                    ax.add_patch(Rectangle((x - width/2, target), width, max(0, price - target),
                                           color='green', alpha=0.2, label='Target' if 'Target' not in shown_labels else ""))
                    ax.add_patch(Rectangle((x - width/2, price), width, max(0, stop - price),
                                           color='red', alpha=0.2, label='Stop Loss' if 'Stop Loss' not in shown_labels else ""))
                shown_labels.update(['Target', 'Stop Loss'])

        except Exception as e:
            print(f"⚠️ Skipping row {i} due to error: {e}")

    ax.set_title(f"{symbol} - High Quality Trade Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.grid(True)
    ax.legend(loc='upper left')
    import os

    # ➤ Final formatting (keep all previous lines)
    plt.tight_layout()

    # ➤ Save to folder instead of showing
    output_dir = "buy_sell_charts"
    os.makedirs(output_dir, exist_ok=True)

    # Clean symbol name (optional, avoids file name errors)
    clean_symbol = symbol.replace(" ", "_").replace("&", "and")

    # Save chart
    plt.savefig(f"{output_dir}/{clean_symbol}.png", bbox_inches="tight")

    # Clear figure to avoid overlap in the next loop
    plt.clf()
    plt.close()


    # Summary table
    print("📊 High quality signal trades:")
    print(df_signals[['Date', 'Signal', 'R_R', 'Score', 'Target', 'Stop_Loss']].tail(10))



def plot_patterns(df, symbol):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle


    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df['Date'], df['Close'], label='Close', alpha=0.7)

    shown = set()
    for idx, row in df.iterrows():
        if row['Pattern']:
            if 'Flag' in row['Pattern']:
                color, marker = 'blue', '>'
            elif 'Bull' in row['Pattern']:
                color, marker = 'green', '^'
            else:
                color, marker = 'red', 'v'
            label = row['Pattern'] if row['Pattern'] not in shown else ""
            ax.scatter(row['Date'], row['Close'], color=color, marker=marker, label=label)
            shown.add(row['Pattern'])

    # ➤ Plot Divergences
    bull_rsi = df[df['RSI_Div'] == 'Bullish']
    bear_rsi = df[df['RSI_Div'] == 'Bearish']
    ax.scatter(bull_rsi['Date'], bull_rsi['Close'], marker='^', color='springgreen', label='RSI Bullish Div', s=100)
    ax.scatter(bear_rsi['Date'], bear_rsi['Close'], marker='v', color='indianred', label='RSI Bearish Div', s=100)

    bull_macd = df[df['MACD_Div'] == 'Bullish']
    bear_macd = df[df['MACD_Div'] == 'Bearish']
    ax.scatter(bull_macd['Date'], bull_macd['Close'], marker='^', color='forestgreen', label='MACD Bullish Div', s=100)
    ax.scatter(bear_macd['Date'], bear_macd['Close'], marker='v', color='darkred', label='MACD Bearish Div', s=100)

    # ➤ Plot Swing Highs and Lows
    ax.scatter(df['Date'], df['Swing_High'], color='orange', label='Swing Highs', marker='x')
    ax.scatter(df['Date'], df['Swing_Low'], color='purple', label='Swing Lows', marker='x')

    shown.add('Target')
    shown.add('Stop Loss')

    # ➤ Final formatting

    ax.set_title(f"{symbol} - Pattern, Divergence & Risk Zones")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()

    # ➤ Save the plot instead of showing it
    import os
    output_dir = "trend_charts"
    os.makedirs(output_dir, exist_ok=True)

    # Clean the symbol name (optional but helpful)
    clean_symbol = symbol.replace(" ", "_").replace("&", "and")

    # Save the figure
    plt.savefig(f"{output_dir}/{clean_symbol}.png", bbox_inches="tight")

    # Clear figure for the next iteration
    plt.clf()
    plt.close()



def score_signals(df):
    df['Score'] = 0.0
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        score = 0
        if row['Close'] > row['MA50']:
            score += 1
        if row['Close'] > row['MA200']:
            score += 1
        if row['Trend_Weekly'] == 'Uptrend' and row['Signal'] == 'Buy':
            score += 2
        if row['Trend_Weekly'] == 'Downtrend' and row['Signal'] == 'Sell':
            score += 2
        if row['MACD_hist'] > prev_row['MACD_hist']:
            score += 1
        if row['MACD'] > row['MACD_signal']:
            score += 1
        if row['RSI'] < 70 and row['Signal'] == 'Buy':
            score += 1
        if row['RSI'] > 30 and row['Signal'] == 'Sell':
            score += 1
        if row['Volume'] > row['VolumeMA5']:
            score += 1

        # Pattern
        if row['Pattern'] in ["Bull Flag", "Bear Flag"]:
            score += 2
        elif row['Pattern'] in ["Bullish Engulfing", "Bearish Engulfing"]:
            score += 1
        elif row['Pattern'] == "Hammer":
            score += 1
        elif row['Pattern'] == "Doji":
            score += 0.5

        if row['MSS'] and row['Order_Block_Type'] in ['Bullish', 'Bearish']:
            score += 2

        if row['RSI_Div'] == "Bullish" and row['Signal'] == 'Buy':
            score += 1
        elif row['RSI_Div'] == "Bearish" and row['Signal'] == 'Sell':
            score += 1

        if row['MACD_Div'] == "Bullish" and row['Signal'] == 'Buy':
            score += 1
        elif row['MACD_Div'] == "Bearish" and row['Signal'] == 'Sell':
            score += 1

        if row['OBV'] > prev_row['OBV'] and row['Signal'] == 'Buy':
            score += 1
        elif row['OBV'] < prev_row['OBV'] and row['Signal'] == 'Sell':
            score += 1

        if pd.notna(row['Anchored_VWAP']) and row['Close'] > row['Anchored_VWAP'] and row['Signal'] == 'Buy':
            score += 1
        elif pd.notna(row['Anchored_VWAP']) and row['Close'] < row['Anchored_VWAP'] and row['Signal'] == 'Sell':
            score += 1

        if row['FVG'] == "Bullish FVG" and row['Signal'] == "Buy":
            score += 1
        elif row['FVG'] == "Bearish FVG" and row['Signal'] == "Sell":
            score += 1

        # Previous results bonus/penalty
        if 'Result' in df.columns and i >= 2:
            prev1 = df.iloc[i - 1]['Result']
            prev2 = df.iloc[i - 2]['Result']
            if prev1 == 'SL' and prev2 == 'SL':
                score -= 2
        
        if row['Close'] > row.get('EMA20', row['MA50']):
            score += 1

        df.at[df.index[i], 'Score'] = score

    print(df[['Date', 'Signal', 'Score']].tail(20))
    return df


pd.set_option('future.no_silent_downcasting', True)  # Future-proofing fillna behavior

def confirm_entry(df):
    df['Confirmed_Signal'] = ""
    for i in range(len(df) - 1):
        signal = df.loc[i, 'Signal']
        next_open = df.loc[i + 1, 'Open']
        next_close = df.loc[i + 1, 'Close']
        if signal == 'Buy' and next_open > df.loc[i, 'Close'] and next_close > next_open:
            df.loc[i, 'Confirmed_Signal'] = 'Buy'
        elif signal == 'Sell' and next_open < df.loc[i, 'Close'] and next_close < next_open:
            df.loc[i, 'Confirmed_Signal'] = 'Sell'
    return df

# Patch: Fix deprecated fillna and type casting safely

def fix_valid_trade_column(df):
    df = df.copy()
    df['Valid_Trade'] = df.get('Valid_Trade', False).fillna(False)
    df = df.infer_objects(copy=False)
    df['Valid_Trade'] = df['Valid_Trade'].astype(bool)
    return df

# === PATCH: Safer rolling accuracy + warning-free ===
def plot_backtest_analysis(result_df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 4))
    sns.countplot(data=result_df, x='Result', palette='pastel', hue='Result', legend=False)
    plt.title('Trade Result Distribution')
    plt.grid(True)
    plt.show()

    result_df = result_df.copy()
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    result_df['Result_Binary'] = result_df['Result'].map({'TP': 1, 'SL': 0})
    result_df = result_df.dropna(subset=['Result_Binary'])
    result_df['Rolling_Accuracy'] = result_df['Result_Binary'].rolling(window=10).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(result_df['Date'], result_df['Rolling_Accuracy'], label='10-Trade Rolling Accuracy', color='blue')
    plt.axhline(0.5, color='red', linestyle='--', label='50% Baseline')
    plt.title('Rolling Trade Accuracy Over Time')
    plt.ylabel('Accuracy')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === PATCH: Filter confirmed signals before scoring/backtest ===
def filter_confirmed_valid(df):
    df['R_R'] = pd.to_numeric(df.get('R_R', np.nan), errors='coerce')
    df = fix_valid_trade_column(df)
    df_signals = df[(df['Confirmed_Signal'].isin(['Buy', 'Sell'])) & df['Valid_Trade'] & (df['R_R'] >= 1.5)]
    return df_signals

def backtest_signals(df, max_trade_duration=10):
    results = []

    if df.empty:
        print("⚠️ Backtest skipped: input DataFrame is empty.")
        return pd.DataFrame()

    required_cols = ['Confirmed_Signal', 'Close', 'Target', 'Stop_Loss']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Backtest error: missing required column '{col}'")
            return pd.DataFrame()

    signal_rows = df[df['Confirmed_Signal'].isin(['Buy', 'Sell'])].copy()

    for idx in signal_rows.index:
        row = df.loc[idx]

        if idx + max_trade_duration >= len(df):
            continue

        entry = row['Close']
        stop = row['Stop_Loss']
        target = row['Target']
        atr = row.get('ATR', None)
        result = 'Open'
        final_price = None

        for j in range(1, max_trade_duration + 1):
            if idx + j >= len(df):
                break

            fut = df.iloc[idx + j]
            high = fut['High']
            low = fut['Low']
            open_ = fut.get('Open', None)
            close_ = fut.get('Close', None)

            if row['Confirmed_Signal'] == 'Buy':
                if low <= stop and high >= target:
                    # Ambiguous: both SL and TP hit
                    if open_ is not None and abs(open_ - stop) < abs(open_ - target):
                        result = 'SL'
                        final_price = stop
                    else:
                        result = 'TP'
                        final_price = target
                    break
                elif low <= stop:
                    result = 'SL'
                    final_price = stop
                    break
                elif high >= target:
                    result = 'TP'
                    final_price = target
                    break

            elif row['Confirmed_Signal'] == 'Sell':
                if high >= stop and low <= target:
                    if open_ is not None and abs(open_ - stop) < abs(open_ - target):
                        result = 'SL'
                        final_price = stop
                    else:
                        result = 'TP'
                        final_price = target
                    break
                elif high >= stop:
                    result = 'SL'
                    final_price = stop
                    break
                elif low <= target:
                    result = 'TP'
                    final_price = target
                    break

        if result == 'Open':
            final_price = df.iloc[idx + max_trade_duration]['Close']
            if atr is not None and abs(final_price - entry) < 0.5 * atr:
                result = 'Open'
            else:
                if row['Confirmed_Signal'] == 'Buy':
                    result = 'TP' if final_price > entry else 'SL'
                else:
                    result = 'TP' if final_price < entry else 'SL'

        results.append({
            'Date': row['Date'],
            'Signal': row['Confirmed_Signal'],
            'Result': result,
            'Score': row.get('Score', 0),
            'R_R': row['R_R']
        })

    result_df = pd.DataFrame(results)

    if result_df.empty or 'Result' not in result_df.columns:
        print("❌ Critical: No trades evaluated or 'Result' column missing.")
        return pd.DataFrame()

    # Summary
    total = len(result_df)
    tp = (result_df['Result'] == 'TP').sum()
    sl = (result_df['Result'] == 'SL').sum()
    open_ = (result_df['Result'] == 'Open').sum()
    acc = round(tp / (tp + sl) * 100, 2) if (tp + sl) else 0

    print("\n📊 Backtest Performance:")
    print(f"Total Trades       : {total}")
    print(f"✅ TP Hit (Wins)    : {tp}")
    print(f"❌ SL Hit (Losses)  : {sl}")
    print(f"🕐 Still Open       : {open_}")
    print(f"🎯 Accuracy         : {acc}%")

    return result_df


# === PATCH: Final integration fix ===
def run_full_backtest_pipeline(df_signals):
    if df_signals.empty:
        print("❌ No valid trades found. Skipping backtest.")
        return pd.DataFrame()

    try:
        result_df = backtest_signals(df, max_trade_duration=10)
    except Exception as e:
        print("❌ Backtest failed:", e)
        return pd.DataFrame()

    try:
        print("\n📊 Previewing result_df before plotting:")
        print(result_df.tail(5))
        plot_backtest_analysis(result_df)
    except Exception as e:
        print("⚠️ Plotting failed:", e)

    return result_df


if __name__ == "__main__":
    symbols = [ 
                 'BERGEPAINT.NS', 'SANOFI.NS', 'COMSYN.NS', 
                'RKDL.NS', 'INSECTICID.NS', 'MAKEINDIA.NS', 'EGOLD.NS', 'DHANUKA.NS', 'UNOMINDA.NS', 'ACLGATI.NS', 'SOUTHWEST.NS', 
                'LIBERTSHOE.NS', 'PFS.NS', 'UJJIVANSFB.NS', 'MENONBE.NS', 'CLEDUCATE.NS', 'NORTHARC.NS', 'MODISONLTD.NS', 'GOPAL.NS', 
                'MIDHANI.NS', 'GOCLCORP.NS', 'JNKINDIA.NS', 'AARVEEDEN.NS', 'RBZJEWEL.NS', 'VALIANTLAB.NS', 'VALIANTORG.NS', 
                'MEGASOFT.NS', 'VIPIND.NS', 'OSWALGREEN.NS', 'CAMLINFINE.NS', 'ADVENZYMES.NS', 'NIITLTD.NS', 'NAVKARCORP.NS', 
                'ENGINERSIN.NS', 'TALBROAUTO.NS', 'SKMEGGPPROD.NS', 'FCL.NS', 'ELECTCAST.NS', 'ELIN.NS', 'DDEVPLSTIK.NS', 'TIRUMALCHM.NS', 
                'GODAVARIB.NS', 'GENCON.NS', 'OSWALAGRO.NS', 'SPMLINFRA.NS'

]
    performance_summary = []  # ✅ Collector for all stock summaries

    for symbol in symbols:
        print(f"\n📊 Processing {symbol}...\n")
        try:
            df = fetch_data(symbol)
            if df is None:
                print(f"❌ Skipping {symbol} due to data fetch issue.")
                continue

            df = add_indicators(df)
            print("Date range in df:", df['Date'].min(), "to", df['Date'].max())

            recent_data = df[(df['Date'] >= '2025-04-15') & (df['Date'] <= '2025-05-28')]
            print(f"Number of rows between 2025-04-15 and 2025-05-28: {recent_data.shape[0]}")

            important_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
            missing_values = recent_data[important_cols].isnull().sum()
            print("Missing values in recent data:\n", missing_values)

            print("Rows with missing values in important columns:")
            print(recent_data[recent_data[important_cols].isnull().any(axis=1)])

            df = add_adx(df)
            df = add_atr(df)
            df = detect_rsi_macd_divergence(df)
            weekly_trend = get_weekly_trend(symbol)
            df = pd.merge_asof(df.sort_values('Date'), weekly_trend.sort_values('Date'), on='Date')

            df = detect_trendlines(df)
            df = detect_patterns(df)
            df = detect_mss_and_order_blocks(df)
            df = add_anchored_vwap(df)
            df = detect_fvg(df)
            df = generate_signals(df)
            df = confirm_entry(df)

            print("\n📌 Confirmed entries preview:")
            print(df[df['Confirmed_Signal'].isin(['Buy', 'Sell'])][['Date', 'Signal', 'Confirmed_Signal', 'Close']].tail(10))

            df = apply_atr_risk(df, rr_ratio=2.0)
            df.dropna(subset=['ATR', 'Stop_Loss', 'Target', 'R_R'], inplace=True)
            df = fix_valid_trade_column(df)

            df['Conflict'] = ""
            for i in range(1, len(df)):
                row = df.iloc[i]
                signal = row['Confirmed_Signal']
                swing_high = not pd.isna(row['Swing_High'])
                swing_low = not pd.isna(row['Swing_Low'])
                trend = row['Trend_Weekly']
                if signal == "Buy" and swing_high:
                    df.at[df.index[i], 'Conflict'] = "Buy near swing high"
                elif signal == "Sell" and swing_low:
                    df.at[df.index[i], 'Conflict'] = "Sell near swing low"
                elif signal == "Buy" and trend != "Uptrend":
                    df.at[df.index[i], 'Conflict'] = "Buy in Downtrend"
                elif signal == "Sell" and trend != "Downtrend":
                    df.at[df.index[i], 'Conflict'] = "Sell in Uptrend"

            df = score_signals(df)

            def classify_confidence(score):
                if score >= 8:
                    return "High"
                elif score >= 5:
                    return "Medium"
                else:
                    return "Low"

            df['Confidence'] = df['Score'].apply(classify_confidence)

            df_signals = filter_confirmed_valid(df)
            df_signals = df_signals.reset_index(drop=True)

            # ✅ Show most recent actionable signals
            from datetime import datetime, timedelta

            lookback_days = 7
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_signals = df_signals[df_signals['Date'] >= cutoff_date]

            if recent_signals.empty:
                print(f"⚠️ No valid signals in the last {lookback_days} days.")
            else:
                print(f"\n📢 Recent high-quality signals (last {lookback_days} days):")
                print(recent_signals[['Date', 'Confirmed_Signal', 'Close', 'R_R', 'Score']])

                # Optional: Save to Excel
                recent_signals.to_excel(f"{symbol}_recent_signals.xlsx", index=False)

            print("✅ Final signals being sent to backtest:")
            print(df_signals[['Date', 'Confirmed_Signal', 'R_R', 'Valid_Trade']].tail())

            plot_patterns(df, symbol)
            plot_signals(df_signals, symbol)

            print("Signals before backtest:")
            print(df_signals[['Date', 'Confirmed_Signal', 'Close', 'R_R', 'Score', 'Valid_Trade']].sort_values(by='Date', ascending=False).head(10))

            result_df = backtest_signals(df, max_trade_duration=10)

            if result_df is None or not isinstance(result_df, pd.DataFrame):
                raise ValueError("Backtest returned invalid result: not a DataFrame")

            # ✅ Store stock-level performance into summary
            tp = (result_df['Result'] == 'TP').sum()
            sl = (result_df['Result'] == 'SL').sum()
            open_trades = (result_df['Result'] == 'Open').sum()
            total = len(result_df)
            accuracy = round(tp / total * 100, 2) if total > 0 else 0

            performance_summary.append({
                'Symbol': symbol,
                'Total Signals': len(df_signals),
                'Trades Taken': total,
                'TP Hit': tp,
                'SL Hit': sl,
                'Still Open': open_trades,
                'Accuracy (%)': accuracy,
                'Avg Score': round(df_signals['Score'].mean(), 2) if not df_signals.empty else 0,
                'Avg R/R': round(df_signals['R_R'].mean(), 2) if not df_signals.empty else 0
            })

            import os
            output_folder = os.path.join("detailed_analysis")
            os.makedirs(output_folder, exist_ok=True)
            df.to_excel(os.path.join(output_folder, f"{symbol}_detailed_signals.xlsx"), index=False)

        except Exception as e:
            print(f"❌ Critical error with {symbol}: {repr(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ✅ Write final performance summary across all stocks
    if performance_summary:
        summary_df = pd.DataFrame(performance_summary)
        summary_df = summary_df.sort_values(by='Accuracy (%)', ascending=False)
        summary_df.to_excel("backtest_performance_summary.xlsx", index=False)
        print("\n✅ Batch performance summary saved to backtest_performance_summary.xlsx")
        print(summary_df)
    else:
        print("\n⚠️ No valid performance data to summarize.")




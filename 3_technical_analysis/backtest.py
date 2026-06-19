

import pandas as pd
import numpy as np
from datetime import timedelta

def backtest_signals(df, max_trade_duration=10, min_score=7, min_rr=1.5):
    results = []

    for i in range(len(df)):
        row = df.iloc[i]

        if row.get('Confirmed_Signal') not in ['Buy', 'Sell']:
            continue


        entry_price = row['Close']
        signal_date = row['Date']
        stop = row['Stop_Loss']
        target = row['Target']

        result = 'Open'
        final_price = None  # for fallback exit

        for j in range(1, max_trade_duration + 1):
            if i + j >= len(df):
                break

            future_row = df.iloc[i + j]
            high = future_row['High']
            low = future_row['Low']
            final_price = future_row['Close']

            if row['Confirmed_Signal'] == 'Buy':
                if low <= stop:
                    result = 'SL'
                    break
                elif high >= target:
                    result = 'TP'
                    break

            elif row['Confirmed_Signal'] == 'Sell':
                if high >= stop:
                    result = 'SL'
                    break
                elif low <= target:
                    result = 'TP'
                    break

        # ✅ Force exit if still Open after max_trade_duration
        if result == 'Open' and final_price is not None:
            if row['Confirmed_Signal'] == 'Buy':
                result = 'TP' if final_price > entry_price else 'SL'
            elif row['Confirmed_Signal'] == 'Sell':
                result = 'TP' if final_price < entry_price else 'SL'

        results.append({
            'Date': signal_date,
            'Signal': row['Confirmed_Signal'],
            'Result': result,
            'Score': row['Score'],
            'R_R': row['R_R']
        })

    result_df = pd.DataFrame(results)

    # === Summary ===
    total = len(result_df)
    tp = len(result_df[result_df['Result'] == 'TP'])
    sl = len(result_df[result_df['Result'] == 'SL'])
    open_ = len(result_df[result_df['Result'] == 'Open'])
    accuracy = round(tp / (tp + sl) * 100, 2) if (tp + sl) > 0 else 0

    print("\n📊 Backtest Performance:")
    print(f"Total Trades       : {total}")
    print(f"✅ TP Hit (Wins)    : {tp}")
    print(f"❌ SL Hit (Losses)  : {sl}")
    print(f"⏰ Still Open       : {open_}")
    print(f"🎯 Accuracy         : {accuracy}%")

    print("\nHigh quality signal trades:")
    print(df[df['Score'] >= min_score][['Date', 'Signal', 'R_R', 'Score', 'Target', 'Stop_Loss']].tail(10))

    return result_df


def apply_atr_risk(df, atr_multiplier=1.0, rr_ratio=2.0):
    df['Stop_Loss'] = np.nan
    df['Target'] = np.nan
    df['R_R'] = np.nan

    for i in range(len(df)):
        row = df.iloc[i]

        if row['Signal'] == 'Buy':
            stop = row['Close'] - row['ATR'] * atr_multiplier
            target = row['Close'] + (row['ATR'] * atr_multiplier * rr_ratio)
            rr = (target - row['Close']) / (row['Close'] - stop) if (row['Close'] - stop) > 0 else 0.0

        elif row['Signal'] == 'Sell':
            stop = row['Close'] + row['ATR'] * atr_multiplier
            target = row['Close'] - (row['ATR'] * atr_multiplier * rr_ratio)
            rr = (row['Close'] - target) / (stop - row['Close']) if (stop - row['Close']) > 0 else 0.0
        else:
            continue

        df.at[i, 'Stop_Loss'] = stop
        df.at[i, 'Target'] = target
        df.at[i, 'R_R'] = rr
        df.at[i, 'Valid_Trade'] = rr >= 1.5

    return df

import yfinance as yf
import pandas as pd
from ta.volume import MFIIndicator
import numpy as np

# ETF proxies for indexes where available
etf_proxies = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
}

# NSE Index tickers from Yahoo (more accurate and direct than top 5 averaging)
sector_index_symbols = {
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY CONSUMPTION": "^CNXCONSUM",
    "INFRASTRUCTURE": "^CNXINFRA",
}

# Fallback: top 5 stocks per index
index_constituents = {
    "NIFTY AUTO": ["TATAMOTORS.NS", "M&M.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "MARUTI.NS"],
    "NIFTY FIN SERVICE": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "AXISBANK.NS"],
    "NIFTY FMCG": ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS", "DABUR.NS", "NESTLEIND.NS"],
    "NIFTY IT": ["INFY.NS", "TCS.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS"],
    "NIFTY PHARMA": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "GLAXO.NS"],
    "NIFTY ENERGY": ["RELIANCE.NS", "IOC.NS", "ONGC.NS", "BPCL.NS", "NTPC.NS"],
    "NIFTY METAL": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS", "NMDC.NS"],
    "NIFTY CONSUMPTION": ["ITC.NS", "MARICO.NS", "EICHERMOT.NS", "TITAN.NS", "HEROMOTOCO.NS"],
    "INFRASTRUCTURE": ["L&T.NS", "IRCON.NS", "IRB.NS", "GMRINFRA.NS", "NBCC.NS"],
    "DEFENSE": ["HAL.NS", "BEL.NS", "BEML.NS", "MAZDOCK.NS", "BDL.NS"]
}

def fetch_metrics(ticker):
    df = yf.download(ticker, period="7d", interval="1d", auto_adjust=True, progress=False)
    if df.empty or df.shape[0] < 5:
        raise ValueError(f"Not enough data for {ticker}")
    df.dropna(inplace=True)

    # Flatten columns
    high = pd.Series(df["High"].values.ravel(), index=df.index)
    low = pd.Series(df["Low"].values.ravel(), index=df.index)
    close = pd.Series(df["Close"].values.ravel(), index=df.index)
    volume = pd.Series(df["Volume"].values.ravel(), index=df.index)

    mfi = MFIIndicator(high=high, low=low, close=close, volume=volume, window=7).money_flow_index()
    last_mean = volume[-7:-1].replace(0, pd.NA).dropna().mean()
    vol_cluster = volume.iloc[-1] / last_mean if pd.notna(last_mean) and last_mean != 0 else 0

    price_change_pct = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

    return {
        "Avg Volume": volume.mean(),
        "MFI": mfi.iloc[-1],
        "7D Price Change (%)": price_change_pct,
        "Vol Cluster": vol_cluster
    }

results = []

for index_name in sorted(set(list(etf_proxies.keys()) + list(index_constituents.keys()))):
    # Try ETF proxy first
    etf_ticker = etf_proxies.get(index_name, None)
    proxy_used = ""
    
    try:
        if etf_ticker:
            metrics = fetch_metrics(etf_ticker)
            proxy_used = etf_ticker
        elif index_name in sector_index_symbols:
            metrics = fetch_metrics(sector_index_symbols[index_name])
            proxy_used = sector_index_symbols[index_name]
        else:
            raise ValueError("No ETF or NSE index proxy available")

    except Exception as e:
        # Fallback to averaging top stocks
        proxy_used = "Averaged Top 5 Stocks"
        try:
            metrics_list = []
            for stock_ticker in index_constituents.get(index_name, []):
                try:
                    m = fetch_metrics(stock_ticker)
                    metrics_list.append(m)
                except Exception as se:
                    print(f"⚠️ Skipping {stock_ticker} for {index_name}: {se}")

            if not metrics_list:
                raise ValueError("No stock data available to average")

            avg_volume = sum(m["Avg Volume"] for m in metrics_list) / len(metrics_list)
            avg_mfi = sum(m["MFI"] for m in metrics_list) / len(metrics_list)
            avg_price_change = sum(m["7D Price Change (%)"] for m in metrics_list) / len(metrics_list)
            avg_vol_cluster = sum(m["Vol Cluster"] for m in metrics_list) / len(metrics_list)

            metrics = {
                "Avg Volume": avg_volume,
                "MFI": avg_mfi,
                "7D Price Change (%)": avg_price_change,
                "Vol Cluster": avg_vol_cluster
            }
        except Exception as e2:
            print(f"❌ Error processing {index_name} via stocks fallback: {e2}")
            continue

    # Composite score
    score = (
        metrics["Avg Volume"] * 0.3 +
        metrics["MFI"] * 0.3 +
        metrics["7D Price Change (%)"] * 0.2 +
        metrics["Vol Cluster"] * 0.2
    )

    results.append({
        "Index": index_name,
        "Proxy Used": proxy_used,
        "Avg Volume": int(metrics["Avg Volume"]),
        "MFI": round(metrics["MFI"], 2),
        "7D Price Change (%)": round(metrics["7D Price Change (%)"], 2),
        "Vol Cluster": round(metrics["Vol Cluster"], 2),
        "Score": round(score, 2)
    })

if not results:
    print("\n❌ No data could be processed.")
else:
    df_results = pd.DataFrame(results)
    

    def min_max_normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    df_results = pd.DataFrame(results)

    # Normalize the columns you want to combine in score
    df_results["Norm_Avg_Volume"] = min_max_normalize(df_results["Avg Volume"]) #AVG VOLUME BETWEEN 10-50 LAKH- GOOD, ABOVE 50 LKH- EXCELLENT
    df_results["Norm_MFI"] = min_max_normalize(df_results["MFI"]) #50-80 MEANS ACCUMILLATION
    df_results["Norm_Price_Change"] = min_max_normalize(df_results["7D Price Change (%)"])
    df_results["Norm_Vol_Cluster"] = min_max_normalize(df_results["Vol Cluster"]) # 1 MEANS STRONG BASE

    # Calculate score from normalized columns now
    df_results["Score"] = (
        df_results["Norm_Avg_Volume"] * 0.3 +
        df_results["Norm_MFI"] * 0.3 +
        df_results["Norm_Price_Change"] * 0.2 +
        df_results["Norm_Vol_Cluster"] * 0.2
    )

    # Columns to display in the final printed output
    display_cols = ["Index", "Proxy Used", "Avg Volume", "MFI", "7D Price Change (%)", "Vol Cluster", "Score"]

    print("\n📊 Top 5 Favorable NSE Indexes for Swing Trading:\n")
    print(df_results[display_cols].head(5).to_string(index=False))


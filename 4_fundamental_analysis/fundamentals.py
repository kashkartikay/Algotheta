import pandas as pd
import numpy as np

# Load your Excel file
file_path = r"C:\Users\PC\algot\eris_lifescience.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)

# Extract sheets
pl = sheets["Profit & Loss"]
bs = sheets["Balance Sheet"]
cf = sheets["Cash Flow"]

# Cleaning helper
def clean_value(val):
    if isinstance(val, str):
        val = val.replace(",", "").strip()
        try:
            return float(val)
        except:
            return np.nan
    return val

# Get metric row by name
def get_metric(sheet, name):
    metric = sheet[sheet.iloc[:, 0].str.strip().str.lower() == name.lower()]
    return metric.iloc[0, 1:].apply(clean_value) if not metric.empty else pd.Series(dtype='float64')

# Extract key metrics
revenue = get_metric(pl, "Sales")
net_profit = get_metric(pl, "Profit after tax")
equity = get_metric(bs, "Share Capital") + get_metric(bs, "Reserves")
debt = get_metric(bs, "Borrowings")
current_assets = get_metric(bs, "Current Assets")
current_liabilities = get_metric(bs, "Current Liabilities")
free_cash_flow = get_metric(cf, "Free Cash Flow")

# Calculate ratios
roe = (net_profit / equity) * 100
de_ratio = debt / equity
current_ratio = current_assets / current_liabilities

# Get latest available year
latest = revenue.last_valid_index()
summary = pd.Series({
    "Revenue": revenue.get(latest),
    "Net Income": net_profit.get(latest),
    "ROE": roe.get(latest),
    "Debt to Equity": de_ratio.get(latest),
    "Current Ratio": current_ratio.get(latest),
    "Free Cash Flow": free_cash_flow.get(latest)
})

print("\n🔍 Fundamental Analysis Summary:")
print(summary)

# Basic assessment
print("\n🧾 Assessment:")
if summary["ROE"] and summary["ROE"] < 10:
    print("⚠️ Weak Return on Equity")
if summary["Debt to Equity"] and summary["Debt to Equity"] > 1:
    print("⚠️ High Leverage")
if summary["Current Ratio"] and summary["Current Ratio"] < 1:
    print("⚠️ Weak Liquidity")
if summary["Free Cash Flow"] and summary["Free Cash Flow"] < 0:
    print("⚠️ Negative Free Cash Flow")

# Save analysis
summary.to_excel(file_path.replace(".xlsx", "_fundamental_analysis.xlsx"))

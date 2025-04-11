import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
options_data = pd.read_csv("spy_options_march.csv")
historical_prices = pd.read_csv("spy_historical_prices_mar2025.csv")

# Convert dates
options_data["Date"] = pd.to_datetime(options_data["Date"])
options_data["Expiration"] = pd.to_datetime(options_data["Expiration"])
historical_prices["Date"] = pd.to_datetime(historical_prices["Date"])
historical_prices.set_index("Date", inplace=True)

# Add spot price to options data
options_data = options_data.merge(
    historical_prices[["Close"]].rename(columns={"Close": "Spot_Price"}),
    left_on="Date", right_index=True, how="left"
)

# Backtesting with numpy and vectorization

def backtest_vectorized(offset=5, expiration_days=1):
    df = options_data.copy()
    df["Target_Strike_Put"] = df["Spot_Price"] - offset
    df["Strike_Diff"] = np.abs(df["Strike"] - df["Target_Strike_Put"])
    df["Target_Exp"] = df["Date"] + pd.to_timedelta(expiration_days, unit='D')

    # Filter for expiration at least expiration_days out
    df = df[df["Expiration"] >= df["Target_Exp"]]

    # Sort to prepare for groupby
    df.sort_values(["Date", "Strike_Diff"], inplace=True)
    best_puts = df.groupby("Date").first().reset_index()

    # Merge in expiration prices
    best_puts = best_puts.merge(
        historical_prices[["Close"]].rename(columns={"Close": "Price_At_Exp"}),
        left_on="Expiration", right_index=True, how="left"
    )

    # Determine assignment and profit
    best_puts["Assigned"] = best_puts["Price_At_Exp"] < best_puts["Strike"]
    best_puts["Put_Profit"] = best_puts["Close"]
    assigned_idx = best_puts["Assigned"]
    best_puts.loc[assigned_idx, "Put_Profit"] = (
        best_puts.loc[assigned_idx, "Close"] - 
        (best_puts.loc[assigned_idx, "Strike"] - best_puts.loc[assigned_idx, "Price_At_Exp"])
    )

    # Calculate cumulative profit
    best_puts["Cumulative_Profit"] = best_puts["Put_Profit"].cumsum()
    return best_puts

if __name__ == "__main__":
    # Run backtest
    results = backtest_vectorized(offset=5, expiration_days=1)

    # Print summary
    print("Total Trades:", len(results))
    print("Assigned Puts:", results["Assigned"].sum())
    print("Win Rate:", (results["Put_Profit"] > 0).mean())
    print("Total Profit:", results["Put_Profit"].sum())
    print("Max Drawdown:", results["Cumulative_Profit"].min())

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(results["Date"], results["Cumulative_Profit"], marker='o')
    plt.title("Cumulative Profit - Vectorized Put Backtest ($5 Offset)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

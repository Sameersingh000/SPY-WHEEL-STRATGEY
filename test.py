import pandas as pd
import matplotlib.pyplot as plt

# Load data
options_data = pd.read_csv("spy_options_march.csv")
historical_prices = pd.read_csv("spy_historical_prices_mar2025.csv")

# Convert dates
options_data["Date"] = pd.to_datetime(options_data["Date"])
options_data["Expiration"] = pd.to_datetime(options_data["Expiration"])
historical_prices["Date"] = pd.to_datetime(historical_prices["Date"])
underlying_prices = historical_prices.set_index("Date")["Close"]

# Function: backtest for a specific offset and expiration
def backtest_wheel(offset, expiration_days):
    results = []

    for current_date in historical_prices["Date"]:
        try:
            spot_price = underlying_prices.loc[current_date]
            target_strike_put = spot_price - offset
            target_expiration = current_date + pd.Timedelta(days=expiration_days)

            options_today = options_data[options_data["Date"] == current_date]
            if options_today.empty:
                continue

            valid_puts = options_today[options_today["Expiration"] >= target_expiration].copy()
            if valid_puts.empty:
                continue

            valid_puts["Strike_Diff"] = abs(valid_puts["Strike"] - target_strike_put)
            best_put = valid_puts.loc[valid_puts["Strike_Diff"].idxmin()]

            put_strike = best_put["Strike"]
            put_premium = best_put["Close"]
            put_expiration = best_put["Expiration"]

            if put_expiration not in underlying_prices:
                continue
            price_at_put_exp = underlying_prices.loc[put_expiration]

            assigned = price_at_put_exp < put_strike
            put_profit = put_premium if not assigned else -1 * (put_strike - price_at_put_exp)

            call_profit = 0
            if assigned:
                call_offset = 5
                target_strike_call = price_at_put_exp + call_offset
                call_date = put_expiration
                call_exp_target = call_date + pd.Timedelta(days=expiration_days)

                call_options = options_data[options_data["Date"] == call_date]
                valid_calls = call_options[call_options["Expiration"] >= call_exp_target].copy()

                if not valid_calls.empty:
                    valid_calls["Strike_Diff"] = abs(valid_calls["Strike"] - target_strike_call)
                    best_call = valid_calls.loc[valid_calls["Strike_Diff"].idxmin()]

                    call_strike = best_call["Strike"]
                    call_premium = best_call["Close"]
                    call_expiration = best_call["Expiration"]

                    if call_expiration in underlying_prices:
                        price_at_call_exp = underlying_prices.loc[call_expiration]
                        call_assigned = price_at_call_exp > call_strike
                        call_profit = call_premium if not call_assigned else -1 * (price_at_call_exp - call_strike)

            total_profit = put_profit + call_profit

            results.append({
                "Entry Date": current_date,
                "Put Strike": put_strike,
                "Put Assigned": assigned,
                "Put Profit": put_profit,
                "Call Profit": call_profit,
                "Total Profit": total_profit
            })

        except Exception:
            continue

    df = pd.DataFrame(results)
    df["Cumulative Profit"] = df["Total Profit"].cumsum()
    return df

# Run fixed offset = $5
print("\n=== Fixed Offset ($5) Backtest ===")
df_fixed = backtest_wheel(offset=5, expiration_days=1)
print("Total Trades:", len(df_fixed))
print("Assigned (Put):", df_fixed["Put Assigned"].sum())
print("Win Rate:", (df_fixed["Total Profit"] > 0).mean())
print("Total Profit:", df_fixed["Total Profit"].sum())
print("Max Drawdown:", df_fixed["Cumulative Profit"].min())

# Optimization across multiple offsets
print("\n=== Optimized Offset Scan ===")
strategy_summary = []
for offset in range(5, 21, 2):
    df = backtest_wheel(offset=offset, expiration_days=1)
    if len(df) == 0:
        continue
    summary = {
        "Offset": offset,
        "Profit": df["Total Profit"].sum(),
        "Assignments": df["Put Assigned"].sum(),
        "Win Rate": (df["Total Profit"] > 0).mean(),
        "Trades": len(df)
    }
    strategy_summary.append(summary)

strategy_df = pd.DataFrame(strategy_summary)
strategy_df = strategy_df.sort_values(by="Profit", ascending=False)
print(strategy_df.head(10))

# Plot fixed offset performance
plt.figure(figsize=(12, 6))
plt.plot(df_fixed["Entry Date"], df_fixed["Cumulative Profit"], marker='o')
plt.title("Fixed Offset ($5) - Cumulative Profit")
plt.xlabel("Date")
plt.ylabel("Cumulative Profit ($)")
plt.grid(True)
plt.tight_layout()
plt.show()
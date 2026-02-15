import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from random import random
from datetime import date, timedelta
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
def setup_page():
    st.set_page_config(page_title="Stock Prediction", layout="wide")
    st.title("📈 Stock Price Simulation")


def yesterday() -> date:
    return date.today() - timedelta(days=1)


# -----------------------------
# Data Fetch
# -----------------------------
@st.cache_data(ttl=60 * 30)
def fetch_history_yf(ticker: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    """
    Returns DataFrame with a single column: Close
    """
    end_plus = end_dt + timedelta(days=1)  # helps include end date
    df = yf.download(
        tickers=ticker,
        start=start_dt.isoformat(),
        end=end_plus.isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="column"
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns (sometimes returned by yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)].to_frame(name="Close")
        else:
            close_cols = [c for c in df.columns if c[0] == "Close"]
            if not close_cols:
                return pd.DataFrame()
            close = df[close_cols[0]].to_frame(name="Close")
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df[["Close"]].copy()

    close = close.dropna()
    return close


def compute_returns_stats(close_series: pd.Series):
    pct = close_series.pct_change().dropna()
    if pct.empty:
        return 0.0, 0.0
    return float(pct.mean()), float(pct.std(ddof=0))


# -----------------------------
# Monte Carlo
# -----------------------------
def run_monte_carlo(last_close: float, mean: float, std: float, sims: int, days_to_sim: int):
    """
    Returns:
      paths_for_plot: list of arrays (length days_to_sim+1)
      end_prices: np.array of length sims
      avg_close, avg_pct, prob_up
    """
    if std == 0 or np.isnan(std):
        paths_for_plot = [np.full(days_to_sim + 1, last_close) for _ in range(min(sims, 50))]
        end_prices = np.full(sims, last_close, dtype=float)
        avg_close = float(last_close)
        avg_pct = 0.0
        prob_up = 0.0
        return paths_for_plot, end_prices, avg_close, avg_pct, prob_up

    end_prices = np.empty(sims, dtype=float)
    up_flags = np.empty(sims, dtype=float)
    paths_for_plot = []

    plot_cap = min(sims, 200)  # don’t plot too many lines (slow)

    for i in range(sims):
        prices = [last_close]
        for _ in range(days_to_sim):
            r = norm.ppf(random(), loc=mean, scale=std)
            prices.append(prices[-1] * (1 + r))

        end_prices[i] = prices[-1]
        up_flags[i] = 1.0 if prices[-1] > last_close else 0.0

        if i < plot_cap:
            paths_for_plot.append(np.array(prices))

    avg_close = float(end_prices.mean())
    avg_pct = float((avg_close - last_close) / last_close)
    prob_up = float(up_flags.mean())

    return paths_for_plot, end_prices, avg_close, avg_pct, prob_up


def plot_paths(paths, avg_close: float, ticker: str, days_to_sim: int, sims: int):
    fig, ax = plt.subplots()
    for p in paths:
        ax.plot(p)  # x-axis is simulation day (0..days_to_sim)

    ax.axhline(y=avg_close, linestyle="dashed", label="Average closing price")
    ax.set_title(f"Monte Carlo simulated paths — {ticker} ({min(sims, 200)} plotted of {sims})", fontweight="bold")
    ax.set_xlabel(f"Simulation day (0 to {days_to_sim})")
    ax.set_ylabel("Price")
    ax.legend()
    return fig


# -----------------------------
# Sidebar
# -----------------------------
def sidebar_controls():
    st.sidebar.header("Controls")

    pred_type = st.sidebar.radio("Prediction type", ["Monte Carlo", "ML (coming soon)"], index=0)

    ticker = st.sidebar.text_input(
        "Main ticker",
        value=st.session_state.get("ticker", "NVDA"),
        placeholder="e.g., MSTR, NVDA, AAPL, ^GSPC, BTC-USD"
    ).upper().strip()
    st.session_state["ticker"] = ticker

    tickers_raw = st.sidebar.text_input(
        "Compare tickers (comma-separated)",
        value=st.session_state.get("tickers_raw", "NVDA, MSTR"),
        placeholder="e.g., NVDA, MSTR"
    )
    st.session_state["tickers_raw"] = tickers_raw
    compare_tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    st.sidebar.subheader("Date range")
    start_dt = st.sidebar.date_input("Start date", value=st.session_state.get("start_dt", date(2012, 1, 1)))
    end_dt = st.sidebar.date_input("End date", value=st.session_state.get("end_dt", yesterday()))
    st.session_state["start_dt"] = start_dt
    st.session_state["end_dt"] = end_dt

    st.sidebar.subheader("Simulation settings")
    sims = st.sidebar.number_input("Number of simulations", value=200, min_value=1, step=50)
    days_to_sim = st.sidebar.number_input("Number of simulation days", value=252, min_value=1, step=10)

    st.sidebar.divider()

    st.sidebar.subheader("Word Notes")
    notes = st.sidebar.text_area(
        "Edit notes shown in the report",
        value=st.session_state.get("notes", "Add your notes here..."),
        height=220
    )
    st.session_state["notes"] = notes

    st.sidebar.download_button(
        "Download notes (.txt)",
        data=notes.encode("utf-8"),
        file_name="notes.txt",
        mime="text/plain"
    )

    return pred_type, ticker, compare_tickers, start_dt, end_dt, int(sims), int(days_to_sim), notes


# -----------------------------
# Main
# -----------------------------
def main():
    setup_page()

    st.write("Stock Analysis(data pulled directly from Yahoo Finance via yfinance.)")
    st.markdown(
        "<span style='color:red; font-weight: bold'>Not financial advice.</span>",
        unsafe_allow_html=True
    )

    pred_type, ticker, compare_tickers, start_dt, end_dt, sims, days_to_sim, notes = sidebar_controls()

    if not ticker:
        st.warning("Enter a ticker to continue.")
        st.stop()

    if start_dt >= end_dt:
        st.error("Start date must be earlier than end date.")
        st.stop()

    # --- Main ticker fetch ---
    hist = fetch_history_yf(ticker, start_dt, end_dt)
    if hist.empty:
        st.error(f"No data returned for `{ticker}` in the selected date range.")
        st.stop()

    st.subheader(f"{ticker} — Close Price")
    st.line_chart(hist)

    mean, std = compute_returns_stats(hist["Close"])
    st.subheader("Daily return stats")
    st.write(f"Mean daily % change: **{mean * 100:.2f}%**")
    st.write(f"Std dev daily % change: **{std * 100:.2f}%**")

    # --- Monte Carlo for main ticker ---
    if pred_type == "Monte Carlo":
        last_close = float(hist["Close"].iloc[-1])

        paths, end_prices, avg_close, avg_pct, prob_up = run_monte_carlo(
            last_close=last_close,
            mean=mean,
            std=std,
            sims=sims,
            days_to_sim=days_to_sim
        )

        st.subheader("Monte Carlo simulation")
        fig = plot_paths(paths, avg_close, ticker, days_to_sim, sims)
        st.pyplot(fig, clear_figure=True)

        st.subheader("Predictions")
        st.write(f"Latest close price: **${last_close:.2f}**")
        st.write(f"Predicted average closing price: **${avg_close:.2f}**")
  
        st.write(f"Predicted percent change: **{avg_pct * 100:.2f}%**")
        st.write(f"Probability of increase: **{prob_up * 100:.2f}%**")

    else:
        st.info("ML mode (COMING SOON).")

    # --- Comparison Section ---
    st.divider()
    st.subheader("🔁 Stock Comparison")

    tickers_to_compare = compare_tickers if compare_tickers else [ticker]

    summary_rows = []
    all_sim_rows = []

    for tk in tickers_to_compare:
        h = fetch_history_yf(tk, start_dt, end_dt)
        if h.empty:
            summary_rows.append({"Ticker": tk, "Status": "No data"})
            continue

        last_close_tk = float(h["Close"].iloc[-1])
        mean_tk, std_tk = compute_returns_stats(h["Close"])

        if pred_type == "Monte Carlo":
            _, end_prices_tk, avg_close_tk, avg_pct_tk, prob_up_tk = run_monte_carlo(
                last_close=last_close_tk,
                mean=mean_tk,
                std=std_tk,
                sims=sims,
                days_to_sim=days_to_sim
            )

            p05 = float(np.percentile(end_prices_tk, 5))
            p50 = float(np.percentile(end_prices_tk, 50))
            p95 = float(np.percentile(end_prices_tk, 95))

            summary_rows.append({
                "Ticker": tk,
                "Status": "OK",
                "Latest Close": round(last_close_tk, 2),
                "Pred Avg Close": round(avg_close_tk, 2),
                "Pred % Change": round(avg_pct_tk * 100, 2),
                "Prob Up %": round(prob_up_tk * 100, 2),
                "P05 (Year-end)": round(p05, 2),
                "P50 (Year-end)": round(p50, 2),
                "P95 (Year-end)": round(p95, 2),
            })

            sim_df = pd.DataFrame({
                "Ticker": tk,
                "Simulation": np.arange(1, sims + 1),
                "YearEndPrice": end_prices_tk
            })
            all_sim_rows.append(sim_df)
        else:
            summary_rows.append({
                "Ticker": tk,
                "Status": "OK",
                "Latest Close": round(last_close_tk, 2),
                "Note": "Enable Monte Carlo to generate predictions"
            })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    if all_sim_rows:
        sims_df = pd.concat(all_sim_rows, ignore_index=True)
    else:
        sims_df = pd.DataFrame(columns=["Ticker", "Simulation", "YearEndPrice"])

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download summary (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="ticker_comparison_summary.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "Download simulations (CSV)",
            data=sims_df.to_csv(index=False).encode("utf-8"),
            file_name="ticker_simulated_year_end_prices.csv",
            mime="text/csv"
        )

    # Notes
    st.subheader("Notes")
    st.write(notes)


if __name__ == "__main__":
    main()

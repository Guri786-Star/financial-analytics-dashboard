
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Analytics Dashboard")
st.caption(
    "Market Analysis | Risk Management | Portfolio Analytics"
)

st.sidebar.header("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload financial CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info(
        "Upload a CSV file with Date, Close, and optionally "
        "Asset, Open, High, Low, and Volume columns."
    )

    # Generate demo data if no file is uploaded
    dates = pd.bdate_range("2024-01-01", periods=250)
    rng = np.random.default_rng(42)
    demo_frames = []

    for asset, start, volatility in [
        ("Demo_Stock_A", 100, 0.015),
        ("Demo_Stock_B", 150, 0.012),
        ("Demo_Stock_C", 80, 0.020)
    ]:
        returns = rng.normal(0.0003, volatility, len(dates))
        prices = start * np.cumprod(1 + returns)

        demo_frames.append(pd.DataFrame({
            "Date": dates,
            "Asset": asset,
            "Close": prices
        }))

    df = pd.concat(demo_frames, ignore_index=True)
    st.warning(
        "Currently displaying synthetic demonstration data, "
        "not actual market prices."
    )

# Validate and clean input
required = {"Date", "Close"}
missing = required - set(df.columns)

if missing:
    st.error(
        "Missing required columns: " + ", ".join(sorted(missing))
    )
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Date", "Close"])
df = df[df["Close"] > 0].copy()

if "Asset" not in df.columns:
    df["Asset"] = "Uploaded Asset"

df = df.sort_values(["Asset", "Date"])

if df.empty:
    st.error("No valid observations were found in the file.")
    st.stop()

assets = sorted(df["Asset"].astype(str).unique())
selected_assets = st.sidebar.multiselect(
    "Select assets",
    options=assets,
    default=assets
)

df = df[df["Asset"].astype(str).isin(selected_assets)].copy()

if df.empty:
    st.warning("Select at least one asset.")
    st.stop()

df["Daily_Return"] = (
    df.groupby("Asset")["Close"].pct_change()
)

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("Assets", df["Asset"].nunique())
c2.metric("Observations", len(df))
c3.metric("Latest Closing Price", f"{df['Close'].iloc[-1]:,.2f}")

valid_returns = df["Daily_Return"].dropna()
if len(valid_returns):
    c4.metric(
        "Annualized Volatility",
        f"{valid_returns.std() * np.sqrt(252):.2%}"
    )
else:
    c4.metric("Annualized Volatility", "Not available")

# Navigation
page = st.sidebar.radio(
    "Choose module",
    [
        "Market Overview",
        "Risk Analytics",
        "Return Correlation",
        "Data Explorer"
    ]
)

if page == "Market Overview":
    st.subheader("Closing Price Trends")
    st.plotly_chart(
        px.line(
            df, x="Date", y="Close", color="Asset",
            title="Closing Prices"
        ),
        use_container_width=True
    )

    st.subheader("Daily Returns")
    st.plotly_chart(
        px.line(
            df, x="Date", y="Daily_Return", color="Asset",
            title="Daily Percentage Returns"
        ),
        use_container_width=True
    )

    st.subheader("Asset Summary")
    summary = (
        df.groupby("Asset")
        .agg(
            Observations=("Close", "count"),
            Latest_Close=("Close", "last"),
            Mean_Daily_Return=("Daily_Return", "mean"),
            Daily_Volatility=("Daily_Return", "std")
        )
        .reset_index()
    )
    summary["Annualized_Volatility"] = (
        summary["Daily_Volatility"] * np.sqrt(252)
    )
    st.dataframe(summary, use_container_width=True)

elif page == "Risk Analytics":
    st.subheader("Historical Risk Estimates")

    confidence = st.selectbox(
        "Confidence level",
        [0.90, 0.95, 0.99],
        index=1,
        format_func=lambda x: f"{x:.0%}"
    )

    investment = st.number_input(
        "Hypothetical investment amount (₹)",
        min_value=1.0,
        value=100000.0,
        step=10000.0
    )

    rows = []
    for asset, group in df.groupby("Asset"):
        r = group["Daily_Return"].dropna()

        if len(r) < 20:
            continue

        threshold = r.quantile(1 - confidence)
        tail = r[r <= threshold]

        rows.append({
            "Asset": asset,
            "Historical VaR (₹)": max(0, -threshold * investment),
            "Expected Shortfall (₹)": max(
                0, -tail.mean() * investment
            ),
            "Annualized Volatility": r.std() * np.sqrt(252)
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "Historical one-day estimates scaled to a hypothetical "
            "investment; not guarantees of future losses."
        )
    else:
        st.warning("At least 20 valid returns per asset are needed.")

elif page == "Return Correlation":
    st.subheader("Daily Return Correlation")

    return_matrix = df.pivot(
        index="Date", columns="Asset", values="Daily_Return"
    )

    if return_matrix.shape[1] >= 2:
        corr = return_matrix.corr()
        st.plotly_chart(
            px.imshow(
                corr, text_auto=".2f",
                zmin=-1, zmax=1,
                color_continuous_scale="RdBu",
                title="Return Correlation Matrix"
            ),
            use_container_width=True
        )
        st.dataframe(corr, use_container_width=True)
    else:
        st.info("Select at least two assets for correlation analysis.")

elif page == "Data Explorer":
    st.subheader("Filtered Financial Dataset")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download filtered data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_financial_data.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Educational dashboard. Validate data and model assumptions "
    "before making financial decisions."
)

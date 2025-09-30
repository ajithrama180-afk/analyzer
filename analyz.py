import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------
# Page config
# -----------------------------------------
st.set_page_config(
    page_title="Stock Momentum Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------
# Helpers
# -----------------------------------------
@st.cache_data
def get_sample_data():
    data = {
        "Ticker": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO"],
        "Company": ["Apple", "Microsoft", "NVIDIA", "Amazon", "Alphabet", "Meta", "Tesla", "AMD", "Netflix", "Broadcom"],
        "Sector": ["Technology", "Technology", "Technology", "Consumer Discretionary", "Communication Services", "Communication Services", "Consumer Discretionary", "Technology", "Communication Services", "Technology"],
        "Performance (Week)": [1.2, 0.8, 3.4, 2.1, 1.5, 2.3, -0.9, 2.9, 1.1, 1.7],
        "Performance (Month)": [4.5, 3.1, 12.2, 5.9, 3.8, 6.7, -2.5, 10.4, 4.2, 5.5],
        "Performance (Quarter)": [12.1, 9.3, 28.4, 14.7, 8.5, 16.3, -5.1, 24.2, 10.7, 13.8],
        "Sales Growth Quarter Over Quarter": [5.0, 4.1, 7.8, 6.3, 3.7, 6.1, 2.5, 7.2, 4.8, 5.4],
        "Market Cap": [3300000, 3100000, 2500000, 2000000, 1900000, 1200000, 800000, 300000, 250000, 520000],
        "20-Day Simple Moving Average": [0.7, 0.5, 1.4, 0.9, 0.6, 1.0, -0.3, 1.1, 0.5, 0.8],
        "50-Day Simple Moving Average": [1.1, 0.9, 2.1, 1.4, 1.0, 1.6, -0.2, 1.8, 0.9, 1.3],
        "Average Volume (3 month)": [60000000, 35000000, 50000000, 70000000, 30000000, 28000000, 120000000, 45000000, 8000000, 6000000],
        "Close": [190.5, 410.2, 120.4, 180.6, 140.1, 320.3, 220.8, 160.2, 490.0, 1700.0],
    }
    return pd.DataFrame(data)

def find_column_matches(df_columns):
    column_map = {}
    column_variations = {
        "ticker": ["Ticker", "ticker", "TICKER", "Symbol", "symbol", "SYMBOL"],
        "company": ["Company", "company", "COMPANY", "Name", "name", "Company Name"],
        "sector": ["Sector", "sector", "SECTOR", "Industry", "industry"],
        "perf_week": ["Performance (Week)", "Weekly Performance", "Week Performance", "Perf Week", "1W Performance", "1W Return"],
        "perf_month": ["Performance (Month)", "Monthly Performance", "Month Performance", "Perf Month", "1M Performance", "1M Return"],
        "perf_quarter": ["Performance (Quarter)", "Quarterly Performance", "Quarter Performance", "Perf Quarter", "3M Performance", "3M Return"],
        "sales_growth": ["Sales Growth Quarter Over Quarter", "Sales Growth QoQ", "Sales Growth", "Revenue Growth QoQ", "Revenue Growth"],
        "market_cap": ["Market Cap", "Market Capitalization", "MarketCap", "Mkt Cap"],
        "sma_20": ["20-Day Simple Moving Average", "20 Day SMA", "20-Day SMA", "SMA 20", "20D SMA"],
        "sma_50": ["50-Day Simple Moving Average", "50 Day SMA", "50-Day SMA", "SMA 50", "50D SMA"],
        "volume": ["Average Volume (3 month)", "Avg Volume 3M", "Volume 3M", "Average Volume", "Volume"],
        "price": ["Price", "Close", "Adj Close", "Last", "Last Price", "Close Price", "Closeprice", "PX_LAST"],
    }
    for key, variations in column_variations.items():
        for v in variations:
            if v in df_columns:
                column_map[key] = v
                break
    return column_map

def clean_and_convert_data(df, column_map):
    df = df.copy()
    numeric_keys = [
        "perf_week", "perf_month", "perf_quarter", "sales_growth",
        "market_cap", "sma_20", "sma_50", "volume", "price"
    ]
    for key in numeric_keys:
        if key in column_map:
            col = column_map[key]
            if df[col].dtype == "object":
                df[col] = (
                    df[col].astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute Dollar Volume (3M) if possible
    if "volume" in column_map and "price" in column_map:
        vol_col = column_map["volume"]
        price_col = column_map["price"]
        df["Dollar_Volume_3M"] = df[vol_col] * df[price_col]
    else:
        df["Dollar_Volume_3M"] = np.nan

    return df

def minmax_0_100(series):
    s = series.astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        # All equal -> neutral 50
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100.0

def calculate_momentum_score(df, column_map, liquidity_mode="Dollar Volume"):
    # liquidity_mode: "Dollar Volume" or "Volume"
    # Start collecting weighted components
    score_components = []
    weights = []

    # Weekly performance
    if "perf_week" in column_map:
        w = 0.15
        score_components.append(minmax_0_100(df[column_map["perf_week"]]) * w)
        weights.append(w)

    # Monthly performance
    if "perf_month" in column_map:
        w = 0.15
        score_components.append(minmax_0_100(df[column_map["perf_month"]]) * w)
        weights.append(w)

    # Quarterly performance
    if "perf_quarter" in column_map:
        w = 0.10
        score_components.append(minmax_0_100(df[column_map["perf_quarter"]]) * w)
        weights.append(w)

    # Sales Growth QoQ
    if "sales_growth" in column_map:
        w = 0.30
        score_components.append(minmax_0_100(df[column_map["sales_growth"]]) * w)
        weights.append(w)

    # Technical indicators: average of SMA20 and SMA50 if both; if one exists, use it
    tech_cols = []
    if "sma_20" in column_map:
        tech_cols.append(df[column_map["sma_20"]].astype(float))
    if "sma_50" in column_map:
        tech_cols.append(df[column_map["sma_50"]].astype(float))
    if len(tech_cols) > 0:
        tech_avg = pd.concat(tech_cols, axis=1).mean(axis=1)
        w = 0.20
        score_components.append(minmax_0_100(tech_avg) * w)
        weights.append(w)

    # Liquidity: toggle between Dollar Volume and Volume
    liquidity_series = None
    label_used = None
    if liquidity_mode == "Dollar Volume" and "Dollar_Volume_3M" in df.columns and df["Dollar_Volume_3M"].notna().any():
        liquidity_series = df["Dollar_Volume_3M"]
        label_used = "Dollar Volume (3M)"
    elif "volume" in column_map:
        liquidity_series = df[column_map["volume"]]
        label_used = column_map["volume"]

    if liquidity_series is not None:
        w = 0.10
        score_components.append(minmax_0_100(liquidity_series) * w)
        weights.append(w)

    if len(weights) == 0:
        df["Momentum_Score"] = 0.0
        return df, {"liquidity_basis": label_used}

    total_weight = sum(weights)
    # Weighted sum normalized by total weight so final stays 0–100
    combined = sum(score_components) / total_weight
    df["Momentum_Score"] = combined.round(2)
    return df, {"liquidity_basis": label_used}

def safe_format_money(x):
    try:
        if pd.notnull(x) and float(x) > 0:
            return f"${float(x):,.0f}"
    except Exception:
        pass
    return "N/A"

def safe_format_pct(x, decimals=2):
    try:
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return "N/A"

# -----------------------------------------
# Sidebar: Data source and options
# -----------------------------------------
st.sidebar.title("Data & Settings")

data_source = st.sidebar.radio("Data source", ["Sample Data", "Upload CSV"], index=0)

uploaded_df = None
if data_source == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            uploaded_df = pd.read_csv(file)
            st.sidebar.success(f"Loaded {uploaded_df.shape[0]} rows × {uploaded_df.shape[1]} cols")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")

# Liquidity basis toggle
liquidity_mode = st.sidebar.selectbox("Liquidity basis (0.10 weight)", ["Dollar Volume", "Volume"], index=0,
                                      help="Dollar Volume = Volume × Price; falls back to Volume if Price missing.")

# -----------------------------------------
# Main header
# -----------------------------------------
st.title("📈 Stock Momentum Analyzer")
st.caption("Compute a composite momentum score with flexible liquidity basis (Volume vs Dollar Volume).")

# -----------------------------------------
# Load and prepare data
# -----------------------------------------
if data_source == "Sample Data":
    df = get_sample_data()
else:
    df = uploaded_df if uploaded_df is not None else None

if df is None or df.empty:
    st.info("Upload a CSV or select Sample Data to begin.")
    st.stop()

# Detect columns
column_map = find_column_matches(df.columns)
missing_core = []
if "ticker" not in column_map:
    missing_core.append("Ticker/Symbol")
if len(missing_core) > 0:
    st.error(f"Missing required column(s): {', '.join(missing_core)}. Please include a ticker ID.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Clean and convert
df = clean_and_convert_data(df, column_map)
df.attrs["column_map"] = column_map  # store for reference

# Sector filter (if any)
sectors = None
if "sector" in column_map:
    sectors = ["All"] + sorted(df[column_map["sector"]].astype(str).unique().tolist())
    selected_sector = st.sidebar.selectbox("Sector filter", sectors, index=0)
    if selected_sector != "All":
        df = df[df[column_map["sector"]].astype(str) == selected_sector]

# Top N
top_n = st.sidebar.slider("Top N by Momentum", min_value=5, max_value=50, value=10, step=1)

# -----------------------------------------
# Compute Momentum Score
# -----------------------------------------
df_scored, calc_meta = calculate_momentum_score(df, column_map, liquidity_mode=liquidity_mode)

# Sort by score
df_scored = df_scored.sort_values("Momentum_Score", ascending=False).reset_index(drop=True)

# -----------------------------------------
# Overview metrics
# -----------------------------------------
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Avg Momentum (Top N)", f"{df_scored.head(top_n)['Momentum_Score'].mean():.2f}")
with colB:
    if "perf_quarter" in column_map:
        best_q = df_scored.head(top_n)[column_map["perf_quarter"]].max()
        st.metric("Best Quarterly Perf (Top N)", safe_format_pct(best_q))
    else:
        st.metric("Best Quarterly Perf (Top N)", "N/A")
with colC:
    if "sales_growth" in column_map:
        avg_sales = df_scored.head(top_n)[column_map["sales_growth"]].mean()
        st.metric("Avg Sales Growth (Top N)", safe_format_pct(avg_sales))
    else:
        st.metric("Avg Sales Growth (Top N)", "N/A")
with colD:
    st.metric("Total Analyzed", f"{len(df_scored):,}")

st.write(f"Liquidity component basis: {calc_meta.get('liquidity_basis', 'N/A')} (weight 0.10)")

# -----------------------------------------
# Tabs
# -----------------------------------------
tabs = st.tabs(["Top Picks", "Performance Analysis", "Detailed View", "Sector Analysis"])

# Top Picks
with tabs[0]:
    display_cols = []
    rename_map = {}

    # Always include identifiers
    if "ticker" in column_map:
        display_cols.append(column_map["ticker"])
        rename_map[column_map["ticker"]] = "Ticker"
    if "company" in column_map:
        display_cols.append(column_map["company"])
        rename_map[column_map["company"]] = "Company"
    if "sector" in column_map:
        display_cols.append(column_map["sector"])
        rename_map[column_map["sector"]] = "Sector"

    # Key metrics
    display_cols += ["Momentum_Score"]
    rename_map["Momentum_Score"] = "Momentum Score"

    if "perf_quarter" in column_map:
        display_cols.append(column_map["perf_quarter"])
        rename_map[column_map["perf_quarter"]] = "Quarterly Perf"

    if "sales_growth" in column_map:
        display_cols.append(column_map["sales_growth"])
        rename_map[column_map["sales_growth"]] = "Sales Growth QoQ"

    if "market_cap" in column_map:
        display_cols.append(column_map["market_cap"])
        rename_map[column_map["market_cap"]] = "Market Cap"

    # Show liquidity columns as available
    if "Dollar_Volume_3M" in df_scored.columns:
        display_cols.append("Dollar_Volume_3M")
        rename_map["Dollar_Volume_3M"] = "Dollar Volume (3M)"
    if "volume" in column_map:
        display_cols.append(column_map["volume"])
        rename_map[column_map["volume"]] = "Avg Volume (3M)"
    if "price" in column_map:
        display_cols.append(column_map["price"])
        rename_map[column_map["price"]] = "Price"

    table = df_scored[display_cols].head(top_n).rename(columns=rename_map).copy()

    # Formatting
    if "Quarterly Perf" in table.columns:
        table["Quarterly Perf"] = table["Quarterly Perf"].apply(lambda x: safe_format_pct(x))
    if "Sales Growth QoQ" in table.columns:
        table["Sales Growth QoQ"] = table["Sales Growth QoQ"].apply(lambda x: safe_format_pct(x))
    if "Market Cap" in table.columns:
        table["Market Cap"] = table["Market Cap"].apply(lambda x: safe_format_money(x))
    if "Dollar Volume (3M)" in table.columns:
        table["Dollar Volume (3M)"] = table["Dollar Volume (3M)"].apply(lambda x: safe_format_money(x))
    if "Price" in table.columns:
        table["Price"] = table["Price"].apply(lambda x: safe_format_money(x))
    if "Avg Volume (3M)" in table.columns:
        # Keep as integer with commas
        table["Avg Volume (3M)"] = table["Avg Volume (3M)"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "N/A")

    st.dataframe(table, use_container_width=True)

    # Download
    csv_bytes = df_scored.head(top_n).to_csv(index=False).encode("utf-8")
    st.download_button("Download Top Picks (CSV)", data=csv_bytes, file_name="top_picks.csv", mime="text/csv")

# Performance Analysis
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.subheader("Momentum Score Distribution (Top 20)")
        hist_df = df_scored.head(20)
        fig = px.histogram(hist_df, x="Momentum_Score", nbins=10, title="Histogram of Momentum Scores")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "perf_quarter" in column_map and "ticker" in column_map:
            st.subheader("Top 10 by Quarterly Performance")
            bar_df = df_scored.nlargest(10, column_map["perf_quarter"])
            fig2 = px.bar(
                bar_df,
                x=column_map["ticker"],
                y=column_map["perf_quarter"],
                labels={column_map["ticker"]: "Ticker", column_map["perf_quarter"]: "Quarterly Perf"},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Quarterly performance or ticker not available for bar chart.")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_cols = []
    corr_rename = {}
    if "perf_week" in column_map:
        corr_cols.append(column_map["perf_week"]); corr_rename[column_map["perf_week"]] = "Perf Week"
    if "perf_month" in column_map:
        corr_cols.append(column_map["perf_month"]); corr_rename[column_map["perf_month"]] = "Perf Month"
    if "perf_quarter" in column_map:
        corr_cols.append(column_map["perf_quarter"]); corr_rename[column_map["perf_quarter"]] = "Perf Quarter"
    if "sales_growth" in column_map:
        corr_cols.append(column_map["sales_growth"]); corr_rename[column_map["sales_growth"]] = "Sales Growth"
    corr_cols.append("Momentum_Score"); corr_rename["Momentum_Score"] = "Momentum Score"

    if len(corr_cols) >= 2:
        corr_df = df_scored[corr_cols].rename(columns=corr_rename)
        corr = corr_df.corr(numeric_only=True)
        heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            colorbar=dict(title="corr")
        ))
        heat.update_layout(height=500, title="Correlation")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")

# Detailed View
with tabs[2]:
    if "ticker" in column_map:
        tickers = df_scored[column_map["ticker"]].astype(str).tolist()
        sel = st.selectbox("Select Ticker", tickers)
        row = df_scored[df_scored[column_map["ticker"]].astype(str) == sel].head(1)
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Momentum Score", f"{r['Momentum_Score']:.2f}")
            with c2:
                rank = int(df_scored['Momentum_Score'].rank(ascending=False, method="min")[row.index[0]])
                st.metric("Rank", f"{rank} / {len(df_scored)}")
            with c3:
                if "market_cap" in column_map:
                    st.metric("Market Cap", safe_format_money(r[column_map["market_cap"]]))
                else:
                    st.metric("Market Cap", "N/A")

            grid1, grid2, grid3 = st.columns(3)
            with grid1:
                if "perf_week" in column_map:
                    st.metric("Perf Week", safe_format_pct(r[column_map["perf_week"]]))
                if "sma_20" in column_map:
                    st.metric("SMA 20D", f"{r[column_map['sma_20']]:.2f}")
            with grid2:
                if "perf_month" in column_map:
                    st.metric("Perf Month", safe_format_pct(r[column_map["perf_month"]]))
                if "sma_50" in column_map:
                    st.metric("SMA 50D", f"{r[column_map['sma_50']]:.2f}")
            with grid3:
                if "perf_quarter" in column_map:
                    st.metric("Perf Quarter", safe_format_pct(r[column_map["perf_quarter"]]))
                if "sales_growth" in column_map:
                    st.metric("Sales Growth QoQ", safe_format_pct(r[column_map["sales_growth"]]))
            grid4, grid5 = st.columns(2)
            with grid4:
                if "price" in column_map:
                    st.metric("Price", safe_format_money(r[column_map["price"]]))
            with grid5:
                if "Dollar_Volume_3M" in row.columns and pd.notnull(r["Dollar_Volume_3M"]):
                    st.metric("Dollar Volume (3M)", safe_format_money(r["Dollar_Volume_3M"]))
                elif "volume" in column_map:
                    st.metric("Avg Volume (3M)", f"{int(r[column_map['volume']]):,}")
    else:
        st.info("Ticker column not detected.")

# Sector Analysis
with tabs[3]:
    if "sector" in column_map:
        grp = df_scored.groupby(column_map["sector"], dropna=False)
        agg_dict = {"Momentum_Score": "mean"}
        if "perf_quarter" in column_map:
            agg_dict[column_map["perf_quarter"]] = "mean"
        if "sales_growth" in column_map:
            agg_dict[column_map["sales_growth"]] = "mean"
        if "market_cap" in column_map:
            agg_dict[column_map["market_cap"]] = "sum"
        sector_summary = grp.agg(agg_dict)
        sector_summary["Count"] = grp.size()
        sector_summary = sector_summary.sort_values("Momentum_Score", ascending=False).reset_index()

        # Rename for display
        disp = sector_summary.rename(columns={
            column_map["sector"]: "Sector",
            "Momentum_Score": "Avg Momentum",
            column_map.get("perf_quarter", "PerfQ"): "Avg Quarterly Perf",
            column_map.get("sales_growth", "SalesG"): "Avg Sales Growth",
            column_map.get("market_cap", "MktCap"): "Total Market Cap",
        })

        # Format money/pct where applicable
        if "Avg Quarterly Perf" in disp.columns:
            disp["Avg Quarterly Perf"] = disp["Avg Quarterly Perf"].apply(lambda x: safe_format_pct(x))
        if "Avg Sales Growth" in disp.columns:
            disp["Avg Sales Growth"] = disp["Avg Sales Growth"].apply(lambda x: safe_format_pct(x))
        if "Total Market Cap" in disp.columns:
            disp["Total Market Cap"] = disp["Total Market Cap"].apply(lambda x: safe_format_money(x))

        st.subheader("Sector Summary")
        st.dataframe(disp, use_container_width=True)

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                sector_summary,
                x=column_map["sector"],
                y="Momentum_Score",
                labels={column_map["sector"]: "Sector", "Momentum_Score": "Avg Momentum"},
                title="Average Momentum by Sector",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(
                sector_summary,
                names=column_map["sector"],
                values="Count",
                title="Stock Count by Sector",
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sector column not detected.")

# Footer
st.caption("Tip: Use the sidebar to switch between Volume and Dollar Volume for the liquidity component.")

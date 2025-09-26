import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Page configuration
st.set_page_config(
    page_title="Stock Momentum Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Stock Momentum Analyzer</h1>', unsafe_allow_html=True)
st.markdown("**Identify high-momentum stocks with advanced scoring algorithms**")

# Sample data
@st.cache_data
def get_sample_data():
    data = {
        'Ticker': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'CRM',
                  'SHOP', 'SQ', 'ROKU', 'ZM', 'PLTR', 'SNOW', 'CRWD', 'DDOG', 'NET', 'OKTA'],
        'Company': ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corporation', 'Tesla Inc.', 'NVIDIA Corporation',
                   'Amazon.com Inc.', 'Meta Platforms Inc.', 'Netflix Inc.', 'Advanced Micro Devices', 'Salesforce Inc.',
                   'Shopify Inc.', 'Block Inc.', 'Roku Inc.', 'Zoom Video Communications', 'Palantir Technologies',
                   'Snowflake Inc.', 'CrowdStrike Holdings', 'Datadog Inc.', 'Cloudflare Inc.', 'Okta Inc.'],
        'Sector': ['Technology', 'Technology', 'Technology', 'Consumer Cyclical', 'Technology',
                  'Consumer Cyclical', 'Communication Services', 'Communication Services', 'Technology', 'Technology',
                  'Technology', 'Financial Services', 'Communication Services', 'Technology', 'Technology',
                  'Technology', 'Technology', 'Technology', 'Technology', 'Technology'],
        'Performance (Week)': [2.5, -1.2, 3.1, 8.2, 15.3, 1.8, 4.5, -2.1, 6.7, 3.8,
                              12.4, 8.9, 5.2, -3.5, 18.7, 9.4, 7.3, 4.6, 11.2, 2.9],
        'Performance (Month)': [8.7, 12.3, 6.9, 25.4, 35.2, 9.8, 18.2, 5.4, 22.1, 14.2,
                               28.9, 19.7, 16.8, 2.1, 42.3, 21.8, 19.4, 15.7, 24.6, 11.8],
        'Performance (Quarter)': [15.2, 22.1, 18.5, 45.6, 78.9, 19.3, 28.7, 12.8, 42.3, 21.5,
                                 52.1, 31.2, 29.4, 8.7, 68.9, 38.6, 33.7, 26.9, 41.2, 19.6],
        'Sales Growth Quarter Over Quarter': [12.5, 25.3, 18.7, 35.2, 55.4, 22.1, 19.8, 8.9, 38.7, 16.3,
                                             42.8, 28.4, 21.7, 5.2, 48.2, 31.5, 27.8, 22.4, 35.8, 15.7],
        'Market Cap': [2800000, 1650000, 2400000, 800000, 1200000, 1500000, 750000, 180000, 220000, 200000,
                      85000, 45000, 8500, 28000, 35000, 52000, 38000, 28000, 22000, 15000],
        '20-Day Simple Moving Average': [3.2, 4.1, 2.9, 12.5, 18.7, 2.3, 6.8, 1.2, 9.8, 4.2,
                                        15.6, 11.2, 7.8, -1.2, 22.1, 12.7, 9.8, 6.1, 14.3, 3.8],
        '50-Day Simple Moving Average': [2.8, 3.5, 2.1, 8.9, 15.2, 1.9, 5.2, 0.8, 7.4, 3.1,
                                        12.3, 8.7, 6.1, -2.1, 18.9, 9.8, 7.2, 4.8, 11.7, 2.9],
        'Average Volume (3 month)': [75000000, 28000000, 32000000, 85000000, 45000000, 38000000, 22000000, 8500000, 65000000, 12000000,
                                    18000000, 25000000, 15000000, 22000000, 35000000, 8500000, 4200000, 3800000, 5600000, 2900000]
    }
    return pd.DataFrame(data)

# Momentum scoring function
def calculate_momentum_score(df):
    """Calculate momentum score based on multiple factors"""
    df = df.copy()
    
    # Normalize performance metrics (0-100 scale)
    perf_cols = ['Performance (Week)', 'Performance (Month)', 'Performance (Quarter)']
    for col in perf_cols:
        df[f'{col}_norm'] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100).fillna(0)
    
    # Normalize sales growth
    df['Sales_Growth_norm'] = ((df['Sales Growth Quarter Over Quarter'] - df['Sales Growth Quarter Over Quarter'].min()) / 
                              (df['Sales Growth Quarter Over Quarter'].max() - df['Sales Growth Quarter Over Quarter'].min()) * 100).fillna(0)
    
    # Technical indicators score
    df['Tech_Score'] = ((df['20-Day Simple Moving Average'] + df['50-Day Simple Moving Average']) / 2)
    df['Tech_Score_norm'] = ((df['Tech_Score'] - df['Tech_Score'].min()) / (df['Tech_Score'].max() - df['Tech_Score'].min()) * 100).fillna(0)
    
    # Volume score (higher volume = better liquidity)
    df['Volume_Score_norm'] = ((df['Average Volume (3 month)'] - df['Average Volume (3 month)'].min()) / 
                              (df['Average Volume (3 month)'].max() - df['Average Volume (3 month)'].min()) * 100).fillna(0)
    
    # Calculate weighted momentum score
    df['Momentum_Score'] = (
        df['Performance (Week)_norm'] * 0.15 +
        df['Performance (Month)_norm'] * 0.15 +
        df['Performance (Quarter)_norm'] * 0.10 +
        df['Sales_Growth_norm'] * 0.30 +
        df['Tech_Score_norm'] * 0.20 +
        df['Volume_Score_norm'] * 0.10
    ).round(2)
    
    return df

# Sidebar
st.sidebar.header("📊 Data Source")

# Data input options
data_option = st.sidebar.radio(
    "Choose data source:",
    ["📊 Use Sample Data", "📁 Upload CSV File"]
)

df = None

if data_option == "📊 Use Sample Data":
    df = get_sample_data()
    st.sidebar.success("✅ Sample data loaded successfully!")
    
else:
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with stock data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")

if df is not None:
    # Calculate momentum scores
    df = calculate_momentum_score(df)
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Sector filter
    if 'Sector' in df.columns:
        sectors = ['All'] + sorted(df['Sector'].unique().tolist())
        selected_sector = st.sidebar.selectbox("Select Sector:", sectors)
        
        if selected_sector != 'All':
            df_filtered = df[df['Sector'] == selected_sector].copy()
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
        selected_sector = 'All'
    
    # Top N stocks
    top_n = st.sidebar.slider("Show Top N Stocks:", 5, min(50, len(df_filtered)), 10)
    
    # Sort by momentum score
    df_filtered = df_filtered.sort_values('Momentum_Score', ascending=False)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Top Picks", "📊 Performance Analysis", "🔍 Detailed View", "📈 Sector Analysis"])
    
    with tab1:
        st.header("🎯 Top Stock Picks")
        
        # Display top stocks
        top_stocks = df_filtered.head(top_n)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_momentum = top_stocks['Momentum_Score'].mean()
            st.metric("Average Momentum Score", f"{avg_momentum:.1f}")
        
        with col2:
            best_performer = top_stocks.iloc[0]['Performance (Quarter)']
            st.metric("Best Quarterly Performance", f"{best_performer:.1f}%")
        
        with col3:
            avg_sales_growth = top_stocks['Sales Growth Quarter Over Quarter'].mean()
            st.metric("Avg Sales Growth", f"{avg_sales_growth:.1f}%")
        
        with col4:
            total_stocks = len(df_filtered)
            st.metric("Total Stocks Analyzed", total_stocks)
        
        st.markdown("---")
        
        # Top picks table
        display_cols = ['Ticker', 'Company', 'Sector', 'Momentum_Score', 
                       'Performance (Quarter)', 'Sales Growth Quarter Over Quarter', 'Market Cap']
        
        if all(col in top_stocks.columns for col in display_cols):
            display_df = top_stocks[display_cols].copy()
            display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:,.0f}M")
            display_df = display_df.rename(columns={
                'Momentum_Score': 'Momentum Score',
                'Performance (Quarter)': 'Q Performance (%)',
                'Sales Growth Quarter Over Quarter': 'Sales Growth (%)'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Download button
        csv_buffer = io.StringIO()
        top_stocks.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Top Picks CSV",
            data=csv_buffer.getvalue(),
            file_name=f"top_{top_n}_momentum_stocks.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.header("📊 Performance Analysis")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Momentum score distribution
            fig_hist = px.histogram(
                df_filtered.head(20), 
                x='Momentum_Score',
                title="Momentum Score Distribution",
                nbins=10,
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Performance comparison
            top_10 = df_filtered.head(10)
            fig_bar = px.bar(
                top_10,
                x='Ticker',
                y='Performance (Quarter)',
                title="Top 10 Quarterly Performance",
                color='Performance (Quarter)',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("📈 Performance Metrics Correlation")
        numeric_cols = ['Performance (Week)', 'Performance (Month)', 'Performance (Quarter)',
                       'Sales Growth Quarter Over Quarter', 'Momentum_Score']
        
        if all(col in df_filtered.columns for col in numeric_cols):
            corr_matrix = df_filtered[numeric_cols].corr()
            
            fig_heatmap = px.imshow(
                corr_matrix,
                title="Correlation Matrix of Key Metrics",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.header("🔍 Detailed Stock Analysis")
        
        # Stock selector
        stock_options = df_filtered['Ticker'].tolist()
        selected_stock = st.selectbox("Select a stock for detailed analysis:", stock_options)
        
        if selected_stock:
            stock_data = df_filtered[df_filtered['Ticker'] == selected_stock].iloc[0]
            
            # Stock info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"{stock_data['Company']} ({selected_stock})")
                st.write(f"**Sector:** {stock_data.get('Sector', 'N/A')}")
                st.write(f"**Market Cap:** ${stock_data.get('Market Cap', 0):,.0f}M")
            
            with col2:
                st.metric(
                    "Momentum Score",
                    f"{stock_data['Momentum_Score']:.1f}",
                    delta=f"Rank #{df_filtered[df_filtered['Ticker'] == selected_stock].index[0] + 1}"
                )
            
            # Performance metrics
            st.subheader("📊 Performance Breakdown")
            
            perf_cols = st.columns(3)
            with perf_cols[0]:
                st.metric("Weekly Performance", f"{stock_data['Performance (Week)']:.1f}%")
            with perf_cols[1]:
                st.metric("Monthly Performance", f"{stock_data['Performance (Month)']:.1f}%")
            with perf_cols[2]:
                st.metric("Quarterly Performance", f"{stock_data['Performance (Quarter)']:.1f}%")
            
            # Additional metrics
            st.subheader("📈 Additional Metrics")
            
            add_cols = st.columns(2)
            with add_cols[0]:
                st.metric("Sales Growth (QoQ)", f"{stock_data['Sales Growth Quarter Over Quarter']:.1f}%")
                st.metric("20-Day SMA", f"{stock_data['20-Day Simple Moving Average']:.1f}%")
            with add_cols[1]:
                st.metric("50-Day SMA", f"{stock_data['50-Day Simple Moving Average']:.1f}%")
                st.metric("Avg Volume (3M)", f"{stock_data['Average Volume (3 month)']:,.0f}")
    
    with tab4:
        st.header("📈 Sector Analysis")
        
        if 'Sector' in df.columns:
            # Sector performance summary
            sector_summary = df.groupby('Sector').agg({
                'Momentum_Score': ['mean', 'count'],
                'Performance (Quarter)': 'mean',
                'Sales Growth Quarter Over Quarter': 'mean',
                'Market Cap': 'sum'
            }).round(2)
            
            sector_summary.columns = ['Avg Momentum Score', 'Stock Count', 'Avg Q Performance', 'Avg Sales Growth', 'Total Market Cap']
            sector_summary = sector_summary.sort_values('Avg Momentum Score', ascending=False)
            
            st.subheader("🏆 Sector Performance Rankings")
            st.dataframe(sector_summary, use_container_width=True)
            
            # Sector charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_sector_momentum = px.bar(
                    sector_summary.reset_index(),
                    x='Sector',
                    y='Avg Momentum Score',
                    title="Average Momentum Score by Sector",
                    color='Avg Momentum Score',
                    color_continuous_scale='viridis'
                )
                fig_sector_momentum.update_xaxes(tickangle=45)
                fig_sector_momentum.update_layout(height=400)
                st.plotly_chart(fig_sector_momentum, use_container_width=True)
            
            with col2:
                fig_sector_count = px.pie(
                    sector_summary.reset_index(),
                    values='Stock Count',
                    names='Sector',
                    title="Stock Distribution by Sector"
                )
                fig_sector_count.update_layout(height=400)
                st.plotly_chart(fig_sector_count, use_container_width=True)
        else:
            st.info("Sector information not available in the dataset.")

else:
    # Welcome message
    st.markdown("""
    ## 👋 Welcome to Stock Momentum Analyzer!
    
    This tool helps you identify high-momentum stocks using advanced scoring algorithms.
    
    ### 🚀 Features:
    - **Momentum Scoring**: Weighted algorithm considering performance, growth, and technical indicators
    - **Interactive Charts**: Visualize stock performance and correlations
    - **Sector Analysis**: Compare performance across different sectors
    - **Export Functionality**: Download your analysis results
    
    ### 📊 Getting Started:
    1. Choose **"Use Sample Data"** to explore the tool with demo data
    2. Or **"Upload CSV File"** to analyze your own stock data
    
    ### 📋 Required CSV Columns:
    - `Ticker`: Stock symbol
    - `Company`: Company name
    - `Sector`: Industry sector
    - `Performance (Week)`: Weekly return %
    - `Performance (Month)`: Monthly return %
    - `Performance (Quarter)`: Quarterly return %
    - `Sales Growth Quarter Over Quarter`: QoQ sales growth %
    - `Market Cap`: Market capitalization
    - `20-Day Simple Moving Average`: 20-day SMA %
    - `50-Day Simple Moving Average`: 50-day SMA %
    - `Average Volume (3 month)`: 3-month average volume
    
    **Select a data source from the sidebar to begin your analysis!**
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        📈 Stock Momentum Analyzer | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
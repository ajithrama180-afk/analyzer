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

# Column mapping function
def find_column_matches(df_columns):
    """Find matching columns in the dataframe with flexible naming"""
    column_map = {}
    
    # Define possible column name variations
    column_variations = {
        'ticker': ['Ticker', 'ticker', 'TICKER', 'Symbol', 'symbol', 'SYMBOL'],
        'company': ['Company', 'company', 'COMPANY', 'Name', 'name', 'Company Name'],
        'sector': ['Sector', 'sector', 'SECTOR', 'Industry', 'industry'],
        'perf_week': ['Performance (Week)', 'Weekly Performance', 'Week Performance', 'Perf Week', '1W Performance', '1W Return'],
        'perf_month': ['Performance (Month)', 'Monthly Performance', 'Month Performance', 'Perf Month', '1M Performance', '1M Return'],
        'perf_quarter': ['Performance (Quarter)', 'Quarterly Performance', 'Quarter Performance', 'Perf Quarter', '3M Performance', '3M Return'],
        'sales_growth': ['Sales Growth Quarter Over Quarter', 'Sales Growth QoQ', 'Sales Growth', 'Revenue Growth QoQ', 'Revenue Growth'],
        'market_cap': ['Market Cap', 'Market Capitalization', 'MarketCap', 'Mkt Cap'],
        'sma_20': ['20-Day Simple Moving Average', '20 Day SMA', '20-Day SMA', 'SMA 20', '20D SMA'],
        'sma_50': ['50-Day Simple Moving Average', '50 Day SMA', '50-Day SMA', 'SMA 50', '50D SMA'],
        'volume': ['Average Volume (3 month)', 'Avg Volume 3M', 'Volume 3M', 'Average Volume', 'Volume']
    }
    
    # Find matches
    for key, variations in column_variations.items():
        for variation in variations:
            if variation in df_columns:
                column_map[key] = variation
                break
    
    return column_map

# Data cleaning and conversion function
def clean_and_convert_data(df, column_map):
    """Clean and convert data types for analysis"""
    df = df.copy()
    
    # Define numeric column keys that should be converted
    numeric_keys = ['perf_week', 'perf_month', 'perf_quarter', 'sales_growth', 
                   'market_cap', 'sma_20', 'sma_50', 'volume']
    
    # Convert columns to numeric, handling errors
    for key in numeric_keys:
        if key in column_map:
            col = column_map[key]
            # Remove any % signs, commas, or other characters
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.replace('$', '')
            
            # Convert to numeric, invalid values become NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with 0 for calculation purposes
            df[col] = df[col].fillna(0)
    
    return df

# Momentum scoring function
def calculate_momentum_score(df):
    """Calculate momentum score based on multiple factors"""
    df = df.copy()
    
    # Find column mappings
    column_map = find_column_matches(df.columns.tolist())
    
    # Clean and convert data first
    df = clean_and_convert_data(df, column_map)
    
    # Initialize momentum score with base value
    df['Momentum_Score'] = 50.0
    
    try:
        score_components = []
        weights = []
        
        # Performance metrics (if available)
        perf_metrics = [
            ('perf_week', 0.15, 'Weekly Performance'),
            ('perf_month', 0.15, 'Monthly Performance'), 
            ('perf_quarter', 0.10, 'Quarterly Performance')
        ]
        
        for key, weight, name in perf_metrics:
            if key in column_map:
                col = column_map[key]
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max != col_min:  # Avoid division by zero
                    normalized = ((df[col] - col_min) / (col_max - col_min) * 100)
                    score_components.append(normalized * weight)
                    weights.append(weight)
        
        # Sales growth (if available)
        if 'sales_growth' in column_map:
            col = column_map['sales_growth']
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                normalized = ((df[col] - col_min) / (col_max - col_min) * 100)
                score_components.append(normalized * 0.30)
                weights.append(0.30)
        
        # Technical indicators (if available)
        tech_cols = []
        if 'sma_20' in column_map:
            tech_cols.append(column_map['sma_20'])
        if 'sma_50' in column_map:
            tech_cols.append(column_map['sma_50'])
        
        if tech_cols:
            df['Tech_Score'] = df[tech_cols].mean(axis=1)
            tech_min = df['Tech_Score'].min()
            tech_max = df['Tech_Score'].max()
            if tech_max != tech_min:
                normalized = ((df['Tech_Score'] - tech_min) / (tech_max - tech_min) * 100)
                score_components.append(normalized * 0.20)
                weights.append(0.20)
        
        # Volume score (if available)
        if 'volume' in column_map:
            col = column_map['volume']
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                normalized = ((df[col] - col_min) / (col_max - col_min) * 100)
                score_components.append(normalized * 0.10)
                weights.append(0.10)
        
        # Calculate final momentum score
        if score_components:
            total_weight = sum(weights)
            if total_weight > 0:
                df['Momentum_Score'] = sum(score_components) / total_weight * 100
            df['Momentum_Score'] = df['Momentum_Score'].round(2)
        
        # Store column mapping for later use
        df.attrs['column_map'] = column_map
        
    except Exception as e:
        st.warning(f"Warning in momentum calculation: {str(e)}")
        # Keep default momentum score of 50.0
    
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
            
            # Show data preview
            st.sidebar.subheader("📋 Data Preview")
            st.sidebar.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Show column detection
            column_map = find_column_matches(df.columns.tolist())
            st.sidebar.write("**Detected Columns:**")
            for key, col in column_map.items():
                st.sidebar.write(f"• {key}: {col}")
            
            missing_keys = set(['ticker']) - set(column_map.keys())
            if missing_keys:
                st.sidebar.warning(f"Missing: {', '.join(missing_keys)}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")

if df is not None:
    # Calculate momentum scores
    try:
        df = calculate_momentum_score(df)
        column_map = getattr(df, 'attrs', {}).get('column_map', {})
        
        # Sidebar filters
        st.sidebar.header("🎛️ Filters")
        
        # Sector filter
        if 'sector' in column_map:
            sector_col = column_map['sector']
            sectors = ['All'] + sorted(df[sector_col].unique().tolist())
            selected_sector = st.sidebar.selectbox("Select Sector:", sectors)
            
            if selected_sector != 'All':
                df_filtered = df[df[sector_col] == selected_sector].copy()
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
                if 'perf_quarter' in column_map:
                    best_performer = top_stocks[column_map['perf_quarter']].max()
                    st.metric("Best Quarterly Performance", f"{best_performer:.1f}%")
                else:
                    st.metric("Best Quarterly Performance", "N/A")
            
            with col3:
                if 'sales_growth' in column_map:
                    avg_sales_growth = top_stocks[column_map['sales_growth']].mean()
                    st.metric("Avg Sales Growth", f"{avg_sales_growth:.1f}%")
                else:
                    st.metric("Avg Sales Growth", "N/A")
            
            with col4:
                total_stocks = len(df_filtered)
                st.metric("Total Stocks Analyzed", total_stocks)
            
            st.markdown("---")
            
            # Top picks table
            display_cols = []
            
            # Add available columns
            if 'ticker' in column_map:
                display_cols.append(column_map['ticker'])
            if 'company' in column_map:
                display_cols.append(column_map['company'])
            if 'sector' in column_map:
                display_cols.append(column_map['sector'])
            
            display_cols.append('Momentum_Score')
            
            if 'perf_quarter' in column_map:
                display_cols.append(column_map['perf_quarter'])
            if 'sales_growth' in column_map:
                display_cols.append(column_map['sales_growth'])
            if 'market_cap' in column_map:
                display_cols.append(column_map['market_cap'])
            
            display_df = top_stocks[display_cols].copy()
            
            # Format Market Cap if it exists
            if 'market_cap' in column_map and column_map['market_cap'] in display_df.columns:
                display_df[column_map['market_cap']] = display_df[column_map['market_cap']].apply(
                    lambda x: f"${x:,.0f}M" if pd.notnull(x) and x > 0 else "N/A"
                )
            
            # Rename columns for display
            display_names = {}
            if 'ticker' in column_map:
                display_names[column_map['ticker']] = 'Ticker'
            if 'company' in column_map:
                display_names[column_map['company']] = 'Company'
            if 'sector' in column_map:
                display_names[column_map['sector']] = 'Sector'
            display_names['Momentum_Score'] = 'Momentum Score'
            if 'perf_quarter' in column_map:
                display_names[column_map['perf_quarter']] = 'Q Performance (%)'
            if 'sales_growth' in column_map:
                display_names[column_map['sales_growth']] = 'Sales Growth (%)'
            if 'market_cap' in column_map:
                display_names[column_map['market_cap']] = 'Market Cap'
            
            display_df = display_df.rename(columns=display_names)
            
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
                if 'perf_quarter' in column_map and 'ticker' in column_map:
                    top_10 = df_filtered.head(10)
                    fig_bar = px.bar(
                        top_10,
                        x=column_map['ticker'],
                        y=column_map['perf_quarter'],
                        title="Top 10 Quarterly Performance",
                        color=column_map['perf_quarter'],
                        color_continuous_scale='viridis'
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Quarterly performance or ticker data not available")
            
            # Correlation heatmap
            st.subheader("📈 Performance Metrics Correlation")
            numeric_cols = ['Momentum_Score']
            
            # Add available numeric columns
            for key in ['perf_week', 'perf_month', 'perf_quarter', 'sales_growth']:
                if key in column_map:
                    numeric_cols.append(column_map[key])
            
            if len(numeric_cols) > 1:
                corr_matrix = df_filtered[numeric_cols].corr()
                
                fig_heatmap = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix of Key Metrics",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                fig_heatmap.update_layout(height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Not enough numeric columns for correlation analysis")
        
        with tab3:
            st.header("🔍 Detailed Stock Analysis")
            
            # Stock selector
            if 'ticker' in column_map:
                ticker_col = column_map['ticker']
                stock_options = df_filtered[ticker_col].tolist()
                selected_stock = st.selectbox("Select a stock for detailed analysis:", stock_options)
                
                if selected_stock:
                    stock_data = df_filtered[df_filtered[ticker_col] == selected_stock].iloc[0]
                    
                    # Stock info
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if 'company' in column_map:
                            company_name = stock_data[column_map['company']]
                            st.subheader(f"{company_name} ({selected_stock})")
                        else:
                            st.subheader(f"{selected_stock}")
                        
                        if 'sector' in column_map:
                            st.write(f"**Sector:** {stock_data[column_map['sector']]}")
                        if 'market_cap' in column_map:
                            st.write(f"**Market Cap:** ${stock_data[column_map['market_cap']]:,.0f}M")
                    
                    with col2:
                        rank = df_filtered[df_filtered[ticker_col] == selected_stock].index[0] + 1
                        st.metric(
                            "Momentum Score",
                            f"{stock_data['Momentum_Score']:.1f}",
                            delta=f"Rank #{rank}"
                        )
                    
                    # Performance metrics
                    st.subheader("📊 Performance Breakdown")
                    
                    perf_cols = st.columns(3)
                    perf_metrics = [
                        ('perf_week', 'Weekly Performance'),
                        ('perf_month', 'Monthly Performance'),
                        ('perf_quarter', 'Quarterly Performance')
                    ]
                    
                    for i, (key, display_name) in enumerate(perf_metrics):
                        with perf_cols[i]:
                            if key in column_map:
                                value = stock_data[column_map[key]]
                                st.metric(display_name, f"{value:.1f}%")
                            else:
                                st.metric(display_name, "N/A")
                    
                    # Additional metrics
                    st.subheader("📈 Additional Metrics")
                    
                    add_cols = st.columns(2)
                    with add_cols[0]:
                        if 'sales_growth' in column_map:
                            value = stock_data[column_map['sales_growth']]
                            st.metric("Sales Growth (QoQ)", f"{value:.1f}%")
                        if 'sma_20' in column_map:
                            value = stock_data[column_map['sma_20']]
                            st.metric("20-Day SMA", f"{value:.1f}%")
                    with add_cols[1]:
                        if 'sma_50' in column_map:
                            value = stock_data[column_map['sma_50']]
                            st.metric("50-Day SMA", f"{value:.1f}%")
                        if 'volume' in column_map:
                            value = stock_data[column_map['volume']]
                            st.metric("Avg Volume (3M)", f"{value:,.0f}")
            else:
                st.error("Ticker column not found in the data")
        
        with tab4:
            st.header("📈 Sector Analysis")
            
            if 'sector' in column_map:
                sector_col = column_map['sector']
                
                # Sector performance summary
                agg_dict = {'Momentum_Score': ['mean', 'count']}
                
                # Add available columns to aggregation
                if 'perf_quarter' in column_map:
                    agg_dict[column_map['perf_quarter']] = 'mean'
                if 'sales_growth' in column_map:
                    agg_dict[column_map['sales_growth']] = 'mean'
                if 'market_cap' in column_map:
                    agg_dict[column_map['market_cap']] = 'sum'
                
                sector_summary = df.groupby(sector_col).agg(agg_dict).round(2)
                
                # Flatten column names
                sector_summary.columns = ['_'.join(col).strip() for col in sector_summary.columns.values]
                sector_summary = sector_summary.rename(columns={
                    'Momentum_Score_mean': 'Avg Momentum Score',
                    'Momentum_Score_count': 'Stock Count'
                })
                
                # Rename other columns if they exist
                for key in ['perf_quarter', 'sales_growth', 'market_cap']:
                    if key in column_map:
                        old_name = f"{column_map[key]}_mean"
                        if old_name in sector_summary.columns:
                            if key == 'perf_quarter':
                                sector_summary = sector_summary.rename(columns={old_name: 'Avg Q Performance'})
                            elif key == 'sales_growth':
                                sector_summary = sector_summary.rename(columns={old_name: 'Avg Sales Growth'})
                        
                        old_name = f"{column_map[key]}_sum"
                        if old_name in sector_summary.columns:
                            if key == 'market_cap':
                                sector_summary = sector_summary.rename(columns={old_name: 'Total Market Cap'})
                
                sector_summary = sector_summary.sort_values('Avg Momentum Score', ascending=False)
                
                st.subheader("🏆 Sector Performance Rankings")
                st.dataframe(sector_summary, use_container_width=True)
                
                # Sector charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_sector_momentum = px.bar(
                        sector_summary.reset_index(),
                        x=sector_col,
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
                        names=sector_col,
                        title="Stock Distribution by Sector"
                    )
                    fig_sector_count.update_layout(height=400)
                    st.plotly_chart(fig_sector_count, use_container_width=True)
            else:
                st.info("Sector information not available in the dataset.")
    
    except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
        st.info("Please check your data format and try again.")
        
        # Complete debug information section
        if st.checkbox("Show debug information"):
            st.write("**Available columns:**", df.columns.tolist())
            st.write("**Data types:**")
            

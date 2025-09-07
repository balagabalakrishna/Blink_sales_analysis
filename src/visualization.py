import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.make_holidays import make_holidays_df
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime, timedelta
def set_background():
    st.markdown(
        """
        <style>
        /* Main app container */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(125% 125% at 50% 10%, #000 40%, #63e 100%);
            color: white;
        }
        
        /* Sidebar - now with gradient background */
        section[data-testid="stSidebar"] > div {
            background: radial-gradient(125% 125% at 50% 10%, #000 40%, #63e 100%) !important;
            border-right: 1px solid #63e;
            color: white;
        }
        
        /* Dashboard title - yellow color */
        h1 {
            color: #FFD700 !important;
            text-align: center;
            font-size: 2.5rem !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        
        /* Section headers - gradient text */
        h2 {
            background: -webkit-linear-gradient(#fff, #63e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8rem !important;
            margin-top: 1.5rem !important;
            border-bottom: 2px solid #63e;
            padding-bottom: 0.3rem;
        }
        
        /* All other text elements */
        h3, h4, h5, h6, p, div, span {
            color: white !important;
        }
        
        /* Metric cards */
        [data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #63e;
        }
        
        /* Plotly charts - fully transparent background */
        .js-plotly-plot .plotly, .plot-container.plotly {
            background: transparent !important;
        }
        
        /* Remove plotly chart backgrounds and grid */
        .modebar-container, .plotly .main-svg {
            background: transparent !important;
        }
        
        /* Remove white background from plotly graphs */
        .plotly .bg {
            fill: transparent !important;
        }
        
        /* Make plotly axes and text white */
        .plotly .xaxis-title, .plotly .yaxis-title,
        .plotly .xtitle, .plotly .ytitle,
        .plotly .xaxis text, .plotly .yaxis text {
            fill: white !important;
        }
        
        /* Make plotly grid lines more subtle */
        .plotly .gridlayer .x line, .plotly .gridlayer .y line {
            stroke: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Dataframes */
        .stDataFrame {
            background-color: rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Input widgets */
        .stTextInput, .stSelectbox, .stSlider {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }
        
        /* Make all plotly traces more visible on dark background */
        .plotly .legend text, .plotly .hovertext {
            fill: white !important;
            color: white !important;
        }
        
        /* Make plotly tooltip background dark */
        .plotly .hoverlayer .hovertext {
            background-color: rgba(0, 0, 0, 0.7) !important;
            border: 1px solid #63e !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Set page config
st.set_page_config(
    page_title="Blinkit Analytics Dashboard",
    page_icon= "https://www.blinkit.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

set_background()
# Load data with comprehensive error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\RADHA KRISHNA\Downloads\data analysis project\data\blinkit_cleaned_data.csv")
        
        # Convert datetime columns with validation
        date_cols = ['promised_delivery_time', 'feedback_date', 'actual_delivery_time']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure numeric columns
        num_cols = ['order_total', 'delivery_time_minutes', 'sentiment', 'rating']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        
        return df
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Data validation
if df.empty:
    st.warning("No data loaded. Please check the file path and format.")
    st.stop()

# Helper function for date filtering
def filter_data(timeframe):
    now = datetime.now()
    if timeframe == '1 Week':
        return df[df['promised_delivery_time'] >= (now - timedelta(days=7))]
    elif timeframe == '1 Month':
        return df[df['promised_delivery_time'] >= (now - timedelta(days=30))]
    elif timeframe == '6 Months':
        return df[df['promised_delivery_time'] >= (now - timedelta(days=180))]
    elif timeframe == '1 Year':
        return df[df['promised_delivery_time'] >= (now - timedelta(days=365))]
    return df

# =============================================
# SIDEBAR NAVIGATION
# =============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <img src="https://www.blinkit.com/favicon.ico" 
             style="height: 60px; margin-bottom: 20px;">
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Navigation")
    analysis_option = st.selectbox(
        "Select Analysis",
        ("Executive Summary", "Sales Forecasting", "Category Performance", 
         "Delivery Analytics", "Customer Insights", "Raw Data Explorer")
    )
    st.markdown("---")
    # This is the corrected footer - properly indented inside sidebar
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <p style="font-size: 0.8rem; color: #aaa;">Created  by Bala  Krishna | v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)



# =============================================
# DASHBOARD LAYOUT
# =============================================
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px;">
        <img src="https://www.blinkit.com/favicon.ico"
             alt="Blinkit Logo" 
             style="height: 50px;">
        <h1 style="color: #FFD700; margin: 0; font-size: 2.3rem;"> Blinkit Performance Analytics Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# =============================================
# CUSTOMER SENTIMENT ANALYSIS (TEXT VERSION)
# =============================================
if analysis_option == "Executive Summary":
    st.header("Executive Summary")
    
    try:
        # First Row Metrics (4 columns)
        cols = st.columns(4)
        
        with cols[0]:
            if 'order_id' in df.columns:
                total_orders = len(df)
                st.metric("Total Orders", total_orders)
        
        with cols[1]:
            if 'order_total' in df.columns:
                avg_order = df['order_total'].mean()
                st.metric("Average Order Value", f"₹{avg_order:.2f}")
        
        with cols[2]:
            if 'category' in df.columns:
                top_category = df['category'].mode()[0] if len(df['category'].mode()) > 0 else "N/A"
                st.metric("Top Category", top_category)
        
        with cols[3]:
            if 'delivery_status_x' in df.columns:
                on_time = (df['delivery_status_x'] == 'On Time').mean() * 100
                st.metric("On-Time Delivery", f"{on_time:.1f}%")

        # Second Row Metrics (4 columns)
        cols2 = st.columns(4)
        
        with cols2[0]:
            if 'customer_id_x' in df.columns:
                unique_customers = df['customer_id_x'].nunique()
                st.metric("Unique Customers", unique_customers)
        
        with cols2[1]:
            if 'rating' in df.columns:
                avg_rating = df['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/5")
        
        with cols2[2]:
            if 'promised_delivery_time' in df.columns:
                recent_date = df['promised_delivery_time'].max()
                st.metric("Most Recent Order", recent_date.strftime('%Y-%m-%d') if pd.notnull(recent_date) else "N/A")
        
        with cols2[3]:
            if 'margin_percentage' in df.columns:
                st.metric("Avg. Gross Margin", f"{df['margin_percentage'].mean():.1f}%")

        # Revenue Metrics (now 1 column)
        cols_revenue = st.columns(1)
        
        with cols_revenue[0]:
            if 'order_total' in df.columns and 'order_year' in df.columns:
                yoy_growth = df.groupby('order_year')['order_total'].sum().pct_change() * 100
                st.metric("YoY Revenue Growth", f"{yoy_growth.iloc[-1]:.1f}%" if not yoy_growth.empty else "N/A")

        # Performance Overview
        st.subheader("Performance Overview")
        
        # Customer Health (3 metrics)
        st.markdown("Customer Health")
        cols_customer = st.columns(3)  # Changed from 2 to 3 columns

        with cols_customer[0]:
            if 'customer_id_x' in df.columns:
                repeat_customers = df['customer_id_x'].duplicated().sum()
                st.metric("Repeat Customer Rate", 
                        f"{repeat_customers/len(df)*100:.1f}%",
                        help="% of customers who ordered more than once")

        with cols_customer[1]:
            if 'rating' in df.columns:
                nps_proxy = (df['rating'] >= 4).mean() * 100
                st.metric("NPS Proxy (%-4+ Stars)", f"{nps_proxy:.1f}%")

        with cols_customer[2]:
            if 'registration_date' in df.columns:
                try:
                    df['registration_date'] = pd.to_datetime(df['registration_date'])
                    new_customers = df[df['registration_date'] >= (pd.to_datetime('today') - pd.Timedelta(days=30))]['customer_id_x'].nunique()
                    st.metric("New Customers (30d)", new_customers)
                except:
                    st.metric("New Customers (30d)", "N/A")

        # Visualization columns
        col1, col2, col3 = st.columns([2,2,1])
        
        with col1:
            if 'delivery_status_x' in df.columns:
                status_counts = df['delivery_status_x'].value_counts()
                fig = px.pie(status_counts, values=status_counts.values,
                            names=status_counts.index, title="Delivery Status")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'category' in df.columns and 'order_total' in df.columns:
                category_sales = df.groupby('category')['order_total'].sum().nlargest(10)
                fig = px.bar(category_sales, title="Top Categories by Revenue",
                            labels={'value': 'Total Sales', 'index': 'Category'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("Delivery Efficiency")
            if 'delivery_duration' in df.columns and 'delivery_delay' in df.columns:
                efficiency = (df['delivery_duration'] / (df['delivery_duration'] + abs(df['delivery_delay']))).mean() * 100
                st.metric("On-Time Efficiency", f"{efficiency:.1f}%")
            
            if 'distance_km' in df.columns:
                st.metric("Avg Distance", f"{df['distance_km'].mean():.1f} km")
         # =============================================       
    except Exception as e:
        st.error(f"Executive summary error: {str(e)}")
# =============================================
# COMPLETE SALES FORECASTING DASHBOARD
# =============================================
# =============================================
# COMPLETE SALES FORECASTING DASHBOARD (STOCK-STYLE)

# =============================================
# =============================================
# COMPLETE SALES FORECASTING DASHBOARD
# =============================================
elif analysis_option == "Sales Forecasting":
    st.header("📈 Blinkit Performance Analytics Dashboard")
    
    # =============================================
    # 1. DASHBOARD CONTROLS
    # =============================================
    col1, col2 = st.columns(2)
    with col1:
        forecast_months = st.slider("Select forecast horizon (months)", 1, 8, 3)
    with col2:
        safety_buffer = st.slider("Safety stock buffer (%)", 10, 50, 20) / 100 + 1
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Analyzing trends..."):
            try:
                # =============================================
                # 2. DATA PROCESSING
                # =============================================
                df['order_date'] = pd.to_datetime(df['order_date'])
                daily = df.set_index('order_date').resample('D').agg({
                    'order_id': 'count',
                    'order_total': 'mean'
                }).rename(columns={
                    'order_id': 'actual_orders', 
                    'order_total': 'avg_value'
                }).dropna()
                
                # =============================================
                # 3. MODEL TRAINING
                # =============================================
                def create_features(df):
                    features = pd.DataFrame(index=df.index)
                    features['dayofweek'] = df.index.dayofweek
                    features['month'] = df.index.month
                    
                    for lag in [1, 7, 14, 28]:
                        features[f'lag_{lag}'] = df['actual_orders'].shift(lag)
                    
                    for window in [7, 14, 28]:
                        features[f'rolling_mean_{window}'] = df['actual_orders'].rolling(window).mean()
                    
                    return features.dropna()
                
                features = create_features(daily)
                X = features
                y = daily['actual_orders'][features.index]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42
                )
                model.fit(X_scaled, y)
                
                # =============================================
                # 4. DYNAMIC FORECAST GENERATION
                # =============================================
                forecast_days = forecast_months * 30
                last_date = features.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_days
                )
                
                forecast = []
                for i in range(forecast_days):
                    current_date = forecast_dates[i]
                    feat = {
                        'dayofweek': current_date.dayofweek,
                        'month': current_date.month,
                    }
                    
                    # Dynamic features
                    for lag in [1, 7, 14, 28]:
                        if i >= lag:
                            feat[f'lag_{lag}'] = forecast[i - lag]
                        else:
                            feat[f'lag_{lag}'] = y.iloc[-lag + i] if len(y) >= (lag - i) else y.mean()
                    
                    for window in [7, 14, 28]:
                        if i >= window:
                            window_data = forecast[i - window:i]
                        else:
                            window_data = list(y.iloc[-(window - i):]) + forecast[:i]
                        feat[f'rolling_mean_{window}'] = np.mean(window_data)
                    
                    future_df = pd.DataFrame([feat])[X.columns]
                    future_scaled = scaler.transform(future_df)
                    forecast.append(model.predict(future_scaled)[0])
                
                forecast_result = pd.DataFrame({
                    'date': forecast_dates,
                    'forecast': forecast,
                    'safety_stock': np.array(forecast) * safety_buffer
                })
                
                # Combine actual and forecast data
                combined = pd.concat([
                    daily[['actual_orders']].rename(columns={'actual_orders': 'orders'}),
                    forecast_result.set_index('date')
                ])
                
                # =============================================
                # 5. STOCK-STYLE VISUALIZATION
                # =============================================
                st.success("Forecast generated successfully!")
                st.divider()
                
                st.subheader("📈 Actual vs Forecast Trend")
                
                fig = go.Figure()
                
                # Actual data (solid line)
                fig.add_trace(go.Scatter(
                    x=combined.index,
                    y=combined['orders'],
                    name='Actual Orders',
                    line=dict(color='#636EFA', width=2),
                    mode='lines'
                ))
                
                # Forecast (dashed line)
                fig.add_trace(go.Scatter(
                    x=forecast_result['date'],
                    y=forecast_result['forecast'],
                    name='Forecast',
                    line=dict(color='#00CC96', width=2, dash='dash'),
                    mode='lines'
                ))
                
                # Safety stock (dotted line)
                fig.add_trace(go.Scatter(
                    x=forecast_result['date'],
                    y=forecast_result['safety_stock'],
                    name='Safety Stock',
                    line=dict(color='#FF6692', width=2, dash='dot'),
                    mode='lines'
                ))
                
                # Add vertical line separating actual vs forecast
                fig.add_vline(
                    x=last_date,
                    line_width=2,
                    line_dash="dash",
                    line_color="grey"
                )
                
                # Add annotation for forecast start
                fig.add_annotation(
                    x=last_date,
                    y=max(combined['orders'].max(), forecast_result['forecast'].max()),
                    text="Forecast Start",
                    showarrow=True,
                    arrowhead=1
                )
                
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Daily Orders',
                    hovermode='x unified',
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                # =============================================
                # 7. MONTHLY TRENDS WITH ACTUAL + FORECAST
                # =============================================
                st.subheader("📅 Monthly Performance")
                
                # Create monthly aggregates
                monthly_actual = daily.resample('M')['actual_orders'].sum()
                monthly_forecast = forecast_result.set_index('date').resample('M')['forecast'].sum()
                monthly_safety = forecast_result.set_index('date').resample('M')['safety_stock'].mean()
                
                fig_month = go.Figure()
                
                # Actual bars
                fig_month.add_trace(go.Bar(
                    x=monthly_actual.index,
                    y=monthly_actual,
                    name='Actual Sales',
                    marker_color='#636EFA'
                ))
                
                # Forecast bars
                fig_month.add_trace(go.Bar(
                    x=monthly_forecast.index,
                    y=monthly_forecast,
                    name='Forecast',
                    marker_color='#00CC96'
                ))
                
                # Safety stock line
                fig_month.add_trace(go.Scatter(
                    x=monthly_safety.index,
                    y=monthly_safety,
                    name='Avg Safety Stock',
                    line=dict(color='#FF6692', width=3, dash='dot'),
                    yaxis='y2'
                ))
                
                fig_month.update_layout(
                    barmode='group',
                    yaxis=dict(title='Total Orders'),
                    yaxis2=dict(
                        title='Safety Stock Level',
                        overlaying='y',
                        side='right'
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_month, use_container_width=True)
                
                # =============================================
                    # 8. STRATEGIC VALUE-ADDED INSIGHTS
                    # =============================================
                # =============================================
                # 8. STRATEGIC VALUE-ADDED INSIGHTS
                # =============================================
                # =============================================
# 8. STRATEGIC VALUE-ADDED INSIGHTS (ENHANCED)
                # =============================================
                st.subheader("💡 Strategic Value Added")

                # Calculate core metrics first
                avg_daily_demand = forecast_result['forecast'].mean()
                std_dev = forecast_result['forecast'].std()
                peak_day = forecast_result.iloc[forecast_result['forecast'].argmax()]

                # Weekly Analysis
                weekly_analysis = forecast_result.set_index('date').resample('W').agg({
                    'forecast': 'sum',
                    'safety_stock': 'mean'
                }).rename(columns={
                    'forecast': 'weekly_orders',
                    'safety_stock': 'avg_safety_stock'
                })

                # Monthly Analysis with Staffing Requirements
                monthly_analysis = forecast_result.set_index('date').resample('M').agg({
                    'forecast': 'sum',
                    'safety_stock': 'mean'
                })
                monthly_analysis['staff_increase'] = (monthly_analysis['forecast'] / monthly_analysis['forecast'].mean() - 1) * 100
                monthly_analysis['required_staff'] = (monthly_analysis['forecast'] * 1.8 / 160).round()  # Assuming 1.8 staff-hours per order, 160 hours/month

                peak_month = monthly_analysis['forecast'].idxmax().strftime('%B')
                peak_month_revenue = (monthly_analysis['forecast'].max() * daily['avg_value'].iloc[-1]) / 100000  # in lakhs

                # Calculate strategic metrics
                buffer_units = int((safety_buffer-1) * forecast_result['forecast'].sum())
                revenue_forecast = (forecast_result['forecast'] * daily['avg_value'].iloc[-1]).sum() / 100000  # in lakhs
                weekday_peaks = forecast_result.groupby(forecast_result['date'].dt.day_name())['forecast'].mean()
                best_discount_days = weekday_peaks.nsmallest(2).index.tolist()
                worst_discount_days = weekday_peaks.nlargest(2).index.tolist()
                peak_day_increase = int((peak_day['forecast']/avg_daily_demand - 1)*100)
                forecast_variance = int((std_dev/avg_daily_demand)*100)

                # Create metrics layout
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Inventory Optimization", 
                            f"{buffer_units:,} extra units",
                            f"{int((safety_buffer-1)*100)}% buffer")
                    st.caption(f"Peek week needs {weekly_analysis['weekly_orders'].max():,} units")

                with col2:
                    st.metric("Revenue Forecasting", 
                            f"₹{revenue_forecast:,.1f} lakhs",
                            f"Peak month: {peak_month} (₹{peak_month_revenue:,.1f}L)")
                    st.caption(f"±{forecast_variance}% variance | Avg ₹{daily['avg_value'].mean():.0f}/order")

                with col3:
                    st.metric("Staff Scheduling", 
                            f"{peak_day_increase}% more staff",
                            f"Peak {peak_day['date'].strftime('%A')}s")
                    st.caption(f"Max weekly staff-hours: {int(weekly_analysis['weekly_orders'].max()*1.8):,}")

                with col4:
                    st.metric("Promotion Planning",
                            f"Best: {best_discount_days[0]}",
                            f"Worst: {worst_discount_days[0]}")
                    st.caption(f"Monthly range: ₹{monthly_analysis['forecast'].min()*daily['avg_value'].mean()/100000:,.1f}L-₹{monthly_analysis['forecast'].max()*daily['avg_value'].mean()/100000:,.1f}L")

                # Staffing Requirements by Month
                st.subheader("👥 Staffing Requirements by Month")
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_staff = px.bar(monthly_analysis, 
                                    x=monthly_analysis.index.strftime('%B'),
                                    y='staff_increase',
                                    labels={'staff_increase': 'Additional Staff % Needed', 'x': 'Month'},
                                    color='staff_increase',
                                    color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig_staff, use_container_width=True)

                with col2:
                    st.markdown("**📝 Monthly Staffing Plan**")
                    for idx, row in monthly_analysis.iterrows():
                        st.write(f"""
                        **{idx.strftime('%B')}**:
                        - {int(row['staff_increase'])}% increase
                        - {int(row['required_staff'])} FTEs needed
                        """)

                # Weekly and Monthly Breakdown
                with st.expander("📅 Detailed Period Analysis", expanded=True):
                    tab1, tab2, tab3 = st.tabs(["Weekly Performance", "Monthly Revenue", "Staffing Calendar"])
                    
                    with tab1:
                        st.markdown("**📊 Weekly Order Volume**")
                        fig_week = px.bar(weekly_analysis, 
                                        x=weekly_analysis.index,
                                        y='weekly_orders',
                                        labels={'weekly_orders': 'Total Orders', 'index': 'Week'})
                        st.plotly_chart(fig_week, use_container_width=True)
                        
                    with tab2:
                        st.markdown("**💰 Monthly Revenue Breakdown**")
                        monthly_revenue = monthly_analysis['forecast'] * daily['avg_value'].iloc[-1]
                        fig_month = px.area(monthly_revenue/100000, 
                                        labels={'value': 'Revenue (Lakhs ₹)', 'index': 'Month'})
                        st.plotly_chart(fig_month, use_container_width=True)
                    
                    with tab3:
                        st.markdown("**📆 Staffing Calendar**")
                        staffing_calendar = monthly_analysis[['staff_increase', 'required_staff']]
                        staffing_calendar['current_staff'] = (staffing_calendar['required_staff'] / (1 + staffing_calendar['staff_increase']/100)).round()
                        st.dataframe(staffing_calendar.style
                                    .format({'staff_increase': '{:.0f}%', 'required_staff': '{:.0f} FTEs', 'current_staff': '{:.0f}'})
                                    .background_gradient(cmap='YlOrBr'),
                                    use_container_width=True)

                # Additional Strategic Recommendations
                st.subheader("🎯 Execution Plan")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🛒 Inventory Plan**")
                    st.write(f"""
                    - **Peak Week Prep**: {weekly_analysis['weekly_orders'].max():,} units ({weekly_analysis['weekly_orders'].idxmax().strftime('%d %b')})
                    - **Safety Stock**: Maintain {weekly_analysis['avg_safety_stock'].mean():.0f} units daily
                    - **Critical Restock**: Every {int(7/(safety_buffer))} days during {peak_month}
                                                                            """)

                with col2:
                    st.markdown("**👥 Staffing Strategy**")
                    peak_staff_month = monthly_analysis['staff_increase'].idxmax().strftime('%B')
                    st.write(f"""
                    - **Peak Hiring Month**: {peak_staff_month} ({int(monthly_analysis['staff_increase'].max())}% increase)
                    - **New Hires Needed**: {int(monthly_analysis['required_staff'].max() - monthly_analysis['required_staff'].min())} temporary staff
                    - **Training Period**: Start in {pd.to_datetime(monthly_analysis['staff_increase'].idxmax() - pd.DateOffset(months=1)).strftime('%B')}
                    - **Shift Allocation**: {int(monthly_analysis.loc[monthly_analysis['staff_increase'].idxmax(), 'required_staff']*0.6)} day / {int(monthly_analysis.loc[monthly_analysis['staff_increase'].idxmax(), 'required_staff']*0.4)} night staff
                    """)

                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

elif analysis_option == "Category Performance":
    st.header(" Category Performance")
    
    try:
        if 'category' in df.columns and 'promised_delivery_time' in df.columns and 'order_total' in df.columns:
            # Timeframe selector
            timeframe = st.selectbox("Select Timeframe", 
                                   ['1 Week', '1 Month', '6 Months', '1 Year', 'All Time'],
                                   index=1)
            
            filtered_df = filter_data(timeframe)
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_category = st.selectbox("Select Category", filtered_df['category'].unique())
                
                # Aggregate data by selected timeframe
                if timeframe in ['1 Week', '1 Month']:
                    # Daily view for short timeframes
                    category_sales = filtered_df[filtered_df['category'] == selected_category].groupby(
                        filtered_df['promised_delivery_time'].dt.date
                    )['order_total'].sum().reset_index()
                    category_sales.columns = ['ds', 'y']
                    
                    if len(category_sales) > 1:
                        fig = px.line(category_sales, x='ds', y='y',
                                     title=f"Daily Sales for {selected_category} ({timeframe})",
                                     labels={'ds': 'Date', 'y': 'Sales Amount'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Insufficient daily data for {selected_category} in {timeframe}")
                else:
                    # Monthly view for longer timeframes
                    category_sales = filtered_df[filtered_df['category'] == selected_category].groupby(
                        [filtered_df['promised_delivery_time'].dt.to_period('M').astype(str)]
                    )['order_total'].sum().reset_index()
                    category_sales.columns = ['Month', 'Sales']
                    
                    if len(category_sales) > 1:
                        fig = px.bar(category_sales, x='Month', y='Sales',
                                    title=f"Monthly Sales for {selected_category} ({timeframe})")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Insufficient monthly data for {selected_category} in {timeframe}")
            
            with col2:
                # Top products in category
                if 'product_name' in df.columns:
                    top_products = filtered_df[filtered_df['category'] == selected_category].groupby(
                        'product_name')['order_total'].sum().nlargest(10).reset_index()
                    
                    if len(top_products) > 0:
                        fig2 = px.bar(top_products, x='order_total', y='product_name',
                                     title=f"Top Products in {selected_category}",
                                     orientation='h',
                                     labels={'order_total': 'Total Revenue', 'product_name': 'Product'})
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning(f"No product data available for {selected_category}")

            # Add download button for category data (properly indented)
            st.markdown("---")
            st.subheader("Export Category Data")
            category_data = filtered_df[filtered_df['category'] == selected_category]
            csv = category_data.to_csv(index=False)
            st.download_button(
                label="Download Category Data as CSV",
                data=csv,
                file_name=f"{selected_category}_data.csv",
                mime="text/csv"
            )


        else:
            st.warning("Required columns for category analysis not found")
    except Exception as e:
        
        st.error(f"Category analysis error: {str(e)}")

elif analysis_option == "Delivery Analytics":
    st.header("Delivery Analytics")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'delivery_status_x' in df.columns:
                st.subheader("Delivery Status")
                status_counts = df['delivery_status_x'].value_counts()
                if len(status_counts) > 0:
                    fig = px.pie(status_counts, 
                                values=status_counts.values,
                                names=status_counts.index,
                                title="Delivery Status Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No delivery status data available")
        
        with col2:
            if 'delivery_time_minutes' in df.columns:
                st.subheader("Delivery Time Distribution")
                delivery_times = df['delivery_time_minutes'].dropna()
                if len(delivery_times) > 0:
                    fig = px.histogram(delivery_times, 
                                     nbins=20,
                                     title="Delivery Time in Minutes",
                                     labels={'value': 'Minutes'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Delivery Time", f"{delivery_times.mean():.1f} mins")
                    with col2:
                        st.metric("On-Time Rate", 
                                 f"{(df['delivery_status_x'] == 'On Time').mean()*100:.1f}%")
                else:
                    st.warning("No delivery time data available")
        
        # Delivery partner performance
        st.subheader("Delivery Partner Performance")
        if 'delivery_partner_id' in df.columns and 'delivery_time_minutes' in df.columns:
            # Calculate average performance metrics
            partner_stats = df.groupby("delivery_partner_id").agg({
                "delivery_time_minutes": "mean",
                "delivery_status_x": lambda x: (x == 'On Time').mean()
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Fastest partners
                fastest = partner_stats.nsmallest(10, "delivery_time_minutes")
                fig1 = px.bar(fastest, x='delivery_partner_id', y='delivery_time_minutes',
                             title="Top 10 Fastest Delivery Partners",
                             labels={'delivery_time_minutes': 'Avg Delivery Time (minutes)'})
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Most reliable partners
                most_reliable = partner_stats.nlargest(10, "delivery_status_x")
                fig2 = px.bar(most_reliable, x='delivery_partner_id', y='delivery_status_x',
                             title="Top 10 Most Reliable Partners",
                             labels={'delivery_status_x': 'On-Time Delivery Rate'})
                st.plotly_chart(fig2, use_container_width=True)
        
        # Area-wise delivery performance
        if 'area' in df.columns:
            st.subheader("Area-wise Delivery Performance")
            area_stats = df.groupby("area").agg({
                "delivery_time_minutes": "mean",
                "delivery_status_x": lambda x: (x == 'On Time').mean()
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig3 = px.bar(area_stats.nsmallest(10, "delivery_time_minutes"),
                             x='area', y='delivery_time_minutes',
                             title="Fastest Delivery Areas",
                             labels={'delivery_time_minutes': 'Avg Delivery Time (minutes)'})
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                fig4 = px.bar(area_stats.nlargest(10, "delivery_status_x"),
                             x='area', y='delivery_status_x',
                             title="Most Reliable Delivery Areas",
                             labels={'delivery_status_x': 'On-Time Delivery Rate'})
                st.plotly_chart(fig4, use_container_width=True)
    
    except Exception as e:
        st.error(f"Delivery analytics error: {str(e)}")

elif analysis_option == "Customer Insights":
    st.header(" Customer Insights")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer segmentation analysis
            if 'customer_segment' in df.columns:
                st.subheader("Customer Segmentation")
                
                segment_orders = df.groupby("customer_segment")['order_total'].agg(['count', 'mean'])
                segment_orders.columns = ['Total Orders', 'Average Order Value']
                
                fig1 = px.bar(segment_orders, 
                             y='Total Orders',
                             title="Total Orders by Customer Segment",
                             labels={'index': 'Segment', 'value': 'Count'})
                st.plotly_chart(fig1, use_container_width=True)
                
                fig2 = px.bar(segment_orders, 
                             y='Average Order Value',
                             title="Average Order Value by Segment",
                             labels={'index': 'Segment', 'value': 'Amount ($)'})
                st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Customer clustering
            st.subheader("Customer Clustering")
            
            if st.checkbox("Enable Advanced Customer Clustering"):
                # Select features for clustering
                features = st.multiselect(
                    "Select features for clustering",
                    ['total_orders', 'avg_order_value', 'order_total'],
                    default=['order_total']
                )
                
                if features:
                    X = df[features].dropna()
                    
                    if len(X) > 0:
                        # Select number of clusters
                        n_clusters = st.slider("Number of clusters", 2, 5, 3)
                        
                        # Perform clustering
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        X['cluster'] = kmeans.fit_predict(X)
                        
                        # Plot clusters
                        if len(features) >= 2:
                            fig3 = px.scatter(X, x=features[0], y=features[1], 
                                             color='cluster', title="Customer Clusters")
                            st.plotly_chart(fig3, use_container_width=True)
                        else:
                            st.warning("Select at least 2 features for 2D visualization")
        
        # Customer retention analysis
        if 'customer_id_x' in df.columns and 'promised_delivery_time' in df.columns:
            st.subheader("Customer Retention")
            
            # Calculate repeat customers
            customer_orders = df.groupby('customer_id_x')['order_id'].nunique()
            repeat_customers = (customer_orders > 1).sum()
            new_customers = (customer_orders == 1).sum()
            
            fig4 = px.pie(values=[repeat_customers, new_customers],
                          names=['Repeat Customers', 'New Customers'],
                          title="Customer Retention Rate")
            st.plotly_chart(fig4, use_container_width=True)
    
    except Exception as e:
        st.error(f"Customer insights error: {str(e)}")

elif analysis_option == "Raw Data Explorer":
    st.header(" Data Explorer")
    
    # Show raw data with filters
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    # Data statistics
    st.subheader("Data Statistics")
    st.write(df.describe())
    
    # Column explorer
    st.subheader("Column Explorer")
    selected_column = st.selectbox("Select column to explore", df.columns)
    
    if selected_column:
        if df[selected_column].dtype in ['object', 'category']:
            # Categorical column
            st.write("Value Counts:")
            st.write(df[selected_column].value_counts())
        else:
            # Numerical column
            fig = px.histogram(df, x=selected_column)
            st.plotly_chart(fig, use_container_width=True)

# Add some styling

# Add some styling
# ======================
st.markdown("""
<style>
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stHeader {
        color: #2c3e50;
    }
    .metric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)
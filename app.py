import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="DataViz Pro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background and color scheme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Custom card styling */
    .metric-card {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        text-align: center;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Animated title */
    .animated-title {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glowing buttons */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown('<h1 class="animated-title">📊 DataViz Pro Dashboard</h1>', unsafe_allow_html=True)

# Sidebar navigation with emojis and styling
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px;">
    <h2 style="color: #4ecdc4;">🚀 Navigation</h2>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Choose Your Adventure:",
    ["🏠 Dashboard Home", "📈 Analytics Hub", "🎨 Visualization Studio", "🤖 AI Insights", "⚙️ Settings"]
)

# Dashboard Home
if page == "🏠 Dashboard Home":
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, #667eea, #764ba2); 
                    padding: 30px; border-radius: 20px; margin: 20px 0;">
            <h2 style="color: white; margin-bottom: 20px;">🌟 Welcome to DataViz Pro!</h2>
            <p style="color: #f8f9fa; font-size: 1.2rem;">
                Transform your data into stunning visualizations with our AI-powered dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive user input section
    st.markdown("### 👤 Personalization Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("🏷️ Your Name:", placeholder="Enter your name here...")
        if name:
            st.markdown(f"""
            <div class="success-message">
                🎉 Hello, {name}! Welcome to your personalized dashboard!
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        favorite_color = st.selectbox(
            "🎨 Choose Your Theme Color:",
            ["🔴 Vibrant Red", "🔵 Ocean Blue", "🟢 Nature Green", "🟡 Sunny Yellow", "🟣 Royal Purple"]
        )
    
    # Dynamic metrics with animations
    st.markdown("### 📊 Real-Time Metrics")
    
    # Generate dynamic data
    if 'metrics_data' not in st.session_state:
        st.session_state.metrics_data = {
            'users': random.randint(1000, 5000),
            'revenue': random.randint(50000, 200000),
            'growth': random.uniform(5, 25)
        }
    
    # Auto-refresh button
    if st.button("🔄 Refresh Metrics", key="refresh_metrics"):
        st.session_state.metrics_data = {
            'users': random.randint(1000, 5000),
            'revenue': random.randint(50000, 200000),
            'growth': random.uniform(5, 25)
        }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="👥 Active Users",
            value=f"{st.session_state.metrics_data['users']:,}",
            delta=f"{random.randint(50, 200)} new today"
        )
    
    with col2:
        st.metric(
            label="💰 Revenue",
            value=f"${st.session_state.metrics_data['revenue']:,}",
            delta=f"{random.uniform(5, 15):.1f}% this month"
        )
    
    with col3:
        st.metric(
            label="📈 Growth Rate",
            value=f"{st.session_state.metrics_data['growth']:.1f}%",
            delta=f"{random.uniform(1, 5):.1f}% vs last week"
        )
    
    # Interactive feature showcase
    st.markdown("### ✨ Feature Showcase")
    
    feature_choice = st.radio(
        "Explore our amazing features:",
        ["🎯 Smart Analytics", "🎨 Custom Visualizations", "🤖 AI Predictions"],
        horizontal=True
    )
    
    if feature_choice == "🎯 Smart Analytics":
        st.info("🎯 **Smart Analytics**: AI-powered insights that automatically detect patterns in your data!")
        
        # Sample analytics
        progress_text = "Analyzing your data patterns..."
        progress_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1, text=progress_text)
        
        st.success("✅ Analysis complete! Found 5 key insights in your data.")
        
    elif feature_choice == "🎨 Custom Visualizations":
        st.info("🎨 **Custom Visualizations**: Create beautiful, interactive charts with just a few clicks!")
        
        # Mini visualization
        sample_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'Sales': [random.randint(1000, 5000) for _ in range(5)]
        })
        
        fig = px.bar(sample_data, x='Month', y='Sales', 
                    title="📊 Sample Sales Data",
                    color='Sales',
                    color_continuous_scale='viridis')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("🤖 **AI Predictions**: Machine learning models that forecast trends and outcomes!")
        
        # Prediction simulation
        with st.spinner("🤖 AI is making predictions..."):
            time.sleep(2)
        
        prediction = random.uniform(75, 95)
        st.success(f"🎯 AI Prediction: {prediction:.1f}% chance of meeting your targets!")

# Analytics Hub
elif page == "📈 Analytics Hub":
    st.markdown("# 📈 Advanced Analytics Hub")
    
    # Generate comprehensive sample data
    @st.cache_data
    def generate_analytics_data():
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        return pd.DataFrame({
            'Date': dates,
            'Revenue': np.random.normal(10000, 2000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 1000,
            'Users': np.random.poisson(500, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
            'Conversion_Rate': np.random.beta(2, 8, len(dates)) * 100,
            'Category': np.random.choice(['Product A', 'Product B', 'Product C'], len(dates)),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
        })
    
    data = generate_analytics_data()
    
    # Date range selector
    st.markdown("### 📅 Time Period Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2024, 12, 31))
    
    # Filter data
    mask = (data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))
    filtered_data = data.loc[mask]
    
    # KPI Cards
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_data['Revenue'].sum()
        st.metric("💰 Total Revenue", f"${total_revenue:,.0f}", f"{(total_revenue/1000000):.1f}M")
    
    with col2:
        avg_users = filtered_data['Users'].mean()
        st.metric("👥 Avg Daily Users", f"{avg_users:,.0f}", f"{(avg_users/1000):.1f}K")
    
    with col3:
        avg_conversion = filtered_data['Conversion_Rate'].mean()
        st.metric("📊 Avg Conversion", f"{avg_conversion:.1f}%", "📈 +2.3%")
    
    with col4:
        total_days = len(filtered_data)
        st.metric("📅 Days Analyzed", f"{total_days}", f"{total_days} days")
    
    # Advanced Charts
    st.markdown("### 📊 Advanced Analytics")
    
    # Revenue trend with forecasting
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trend', 'User Distribution', 'Conversion Funnel', 'Regional Performance'),
        specs=[[{"secondary_y": True}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Revenue trend
    fig.add_trace(
        go.Scatter(x=filtered_data['Date'], y=filtered_data['Revenue'],
                  name='Revenue', line=dict(color='#ff6b6b', width=3)),
        row=1, col=1
    )
    
    # Add moving average
    filtered_data['Revenue_MA'] = filtered_data['Revenue'].rolling(window=30).mean()
    fig.add_trace(
        go.Scatter(x=filtered_data['Date'], y=filtered_data['Revenue_MA'],
                  name='30-Day MA', line=dict(color='#4ecdc4', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Pie chart for categories
    category_data = filtered_data.groupby('Category')['Revenue'].sum()
    fig.add_trace(
        go.Pie(labels=category_data.index, values=category_data.values,
               name="Category Distribution", marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1']),
        row=1, col=2
    )
    
    # Bar chart for regions
    region_data = filtered_data.groupby('Region')['Users'].mean()
    fig.add_trace(
        go.Bar(x=region_data.index, y=region_data.values,
               name="Avg Users by Region", marker_color='#96ceb4'),
        row=2, col=1
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=filtered_data['Users'], y=filtered_data['Revenue'],
                  mode='markers', name='Users vs Revenue',
                  marker=dict(color=filtered_data['Conversion_Rate'], 
                            colorscale='viridis', size=8)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Comprehensive Analytics Dashboard",
                     plot_bgcolor='rgba(0,0,0,0)',
                     paper_bgcolor='rgba(0,0,0,0)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table with search
    st.markdown("### 🔍 Data Explorer")
    
    search_term = st.text_input("🔍 Search data:", placeholder="Enter search term...")
    
    if search_term:
        mask = filtered_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        display_data = filtered_data[mask]
    else:
        display_data = filtered_data.head(100)
    
    st.dataframe(display_data, use_container_width=True)
    
    # Download option
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f'analytics_data_{start_date}_to_{end_date}.csv',
        mime='text/csv'
    )

# Visualization Studio
elif page == "🎨 Visualization Studio":
    st.markdown("# 🎨 Interactive Visualization Studio")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff9a9e, #fecfef); 
                padding: 20px; border-radius: 15px; margin: 20px 0;">
        <h3 style="color: #2c3e50; text-align: center;">
            🎨 Create Stunning Visualizations in Real-Time!
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart type selection
    chart_type = st.selectbox(
        "🎯 Choose Visualization Type:",
        ["📊 Interactive Bar Chart", "📈 Dynamic Line Chart", "🥧 Animated Pie Chart", 
         "🎭 3D Scatter Plot", "🌊 Heatmap", "📉 Candlestick Chart"]
    )
    
    # Data customization
    st.markdown("### ⚙️ Customize Your Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_points = st.slider("📊 Number of Data Points:", 10, 100, 50)
    with col2:
        noise_level = st.slider("🌪️ Noise Level:", 0.0, 2.0, 0.1)
    with col3:
        trend_strength = st.slider("📈 Trend Strength:", 0.0, 5.0, 1.0)
    
    # Generate custom data based on selections
    np.random.seed(42)
    x_data = np.linspace(0, 10, data_points)
    y_data = trend_strength * x_data + noise_level * np.random.randn(data_points) + np.sin(x_data) * 2
    
    # Create visualizations based on selection
    if chart_type == "📊 Interactive Bar Chart":
        categories = [f"Category {i+1}" for i in range(data_points)]
        fig = px.bar(
            x=categories[:20], y=y_data[:20],
            title="🎨 Customized Bar Chart",
            color=y_data[:20],
            color_continuous_scale='rainbow'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
    elif chart_type == "📈 Dynamic Line Chart":
        fig = px.line(
            x=x_data, y=y_data,
            title="📈 Dynamic Trend Analysis",
            markers=True
        )
        fig.update_traces(line=dict(color='#ff6b6b', width=4))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
    elif chart_type == "🥧 Animated Pie Chart":
        pie_data = np.abs(y_data[:8])
        pie_labels = [f"Segment {i+1}" for i in range(8)]
        fig = px.pie(
            values=pie_data, names=pie_labels,
            title="🥧 Interactive Pie Chart",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
    elif chart_type == "🎭 3D Scatter Plot":
        z_data = np.random.randn(data_points) * trend_strength
        fig = px.scatter_3d(
            x=x_data, y=y_data, z=z_data,
            title="🎭 3D Data Exploration",
            color=z_data,
            size=[abs(val) + 1 for val in z_data],
            color_continuous_scale='viridis'
        )
        
    elif chart_type == "🌊 Heatmap":
        # Create 2D data for heatmap
        heatmap_data = np.random.randn(10, 10) * noise_level + trend_strength
        fig = px.imshow(
            heatmap_data,
            title="🌊 Data Heatmap",
            color_continuous_scale='rainbow',
            aspect='auto'
        )
        
    else:  # Candlestick chart
        # Generate OHLC data
        dates = pd.date_range(start='2024-01-01', periods=data_points, freq='D')
        ohlc_data = pd.DataFrame({
            'Date': dates,
            'Open': y_data,
            'High': y_data + np.abs(np.random.randn(data_points) * noise_level),
            'Low': y_data - np.abs(np.random.randn(data_points) * noise_level),
            'Close': y_data + np.random.randn(data_points) * noise_level * 0.5
        })
        
        fig = go.Figure(data=go.Candlestick(
            x=ohlc_data['Date'],
            open=ohlc_data['Open'],
            high=ohlc_data['High'],
            low=ohlc_data['Low'],
            close=ohlc_data['Close'],
            name="OHLC Data"
        ))
        fig.update_layout(
            title="📉 Financial Candlestick Chart",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Chart customization options
    st.markdown("### 🎨 Chart Styling Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        color_theme = st.selectbox(
            "🌈 Color Theme:",
            ["Rainbow", "Ocean", "Sunset", "Forest", "Monochrome"]
        )
    
    with col2:
        animation_speed = st.slider("⚡ Animation Speed:", 100, 2000, 500)
    
    if st.button("🚀 Apply Styling"):
        st.success("✨ Styling applied! Your chart looks amazing!")

# AI Insights
elif page == "🤖 AI Insights":
    st.markdown("# 🤖 AI-Powered Insights Engine")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                padding: 25px; border-radius: 20px; margin: 20px 0; text-align: center;">
        <h2 style="color: white; margin-bottom: 15px;">🧠 Artificial Intelligence at Work</h2>
        <p style="color: #f8f9fa; font-size: 1.1rem;">
            Let our AI analyze your data and provide intelligent recommendations!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Analysis Options
    analysis_type = st.radio(
        "🎯 Choose AI Analysis Type:",
        ["🔍 Pattern Detection", "📊 Trend Forecasting", "🎯 Anomaly Detection", "💡 Smart Recommendations"],
        horizontal=True
    )
    
    if analysis_type == "🔍 Pattern Detection":
        st.markdown("### 🔍 AI Pattern Detection")
        
        if st.button("🚀 Run Pattern Analysis"):
            with st.spinner("🤖 AI is analyzing patterns..."):
                time.sleep(3)
            
            # Simulate AI findings
            patterns = [
                "📈 Strong upward trend detected in Q3-Q4 data",
                "🔄 Cyclical pattern every 7 days - likely weekly seasonality",
                "⚡ Sudden spike in activity during weekends",
                "🎯 High correlation between user engagement and revenue",
                "📊 Three distinct customer segments identified"
            ]
            
            st.success("✅ Pattern analysis complete!")
            
            for i, pattern in enumerate(patterns, 1):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e, #38ef7d); 
                            padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>Pattern {i}:</strong> {pattern}
                </div>
                """, unsafe_allow_html=True)
    
    elif analysis_type == "📊 Trend Forecasting":
        st.markdown("### 📊 AI Trend Forecasting")
        
        forecast_period = st.slider("📅 Forecast Period (days):", 7, 90, 30)
        
        if st.button("🔮 Generate Forecast"):
            with st.spinner("🤖 AI is predicting the future..."):
                time.sleep(2)
            
            # Generate sample forecast data
            dates = pd.date_range(start=datetime.now(), periods=forecast_period, freq='D')
            forecast = np.random.randn(forecast_period).cumsum() + 100
            confidence_upper = forecast + np.random.rand(forecast_period) * 10
            confidence_lower = forecast - np.random.rand(forecast_period) * 10
            
            # Create forecast chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color='#ff6b6b', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(confidence_upper) + list(confidence_lower[::-1]),
                fill='toself',
                fillcolor='rgba(255, 107, 107, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title="🔮 AI-Generated Forecast",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"🎯 Forecast generated for next {forecast_period} days with 85% confidence!")
    
    elif analysis_type == "🎯 Anomaly Detection":
        st.markdown("### 🎯 AI Anomaly Detection")
        
        sensitivity = st.slider("🎚️ Detection Sensitivity:", 0.1, 1.0, 0.5)
        
        if st.button("🕵️ Detect Anomalies"):
            with st.spinner("🤖 AI is hunting for anomalies..."):
                time.sleep(2)
            
            # Generate sample data with anomalies
            normal_data = np.random.normal(50, 10, 100)
            anomalies = np.random.choice([0, 1], 100, p=[0.95, 0.05])
            data_with_anomalies = normal_data + anomalies * np.random.normal(0, 30, 100)
            
            # Create anomaly detection chart
            fig = go.Figure()
            
            normal_indices = np.where(anomalies == 0)[0]
            anomaly_indices = np.where(anomalies == 1)[0]
            
            fig.add_trace(go.Scatter(
                x=normal_indices,
                y=data_with_anomalies[normal_indices],
                mode='markers',
                name='Normal Data',
                marker=dict(color='#4ecdc4', size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=anomaly_indices,
                y=data_with_anomalies[anomaly_indices],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#ff6b6b', size=12, symbol='x')
            ))
            
            fig.update_layout(
                title="🎯 Anomaly Detection Results",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            num_anomalies = len(anomaly_indices)
            st.warning(f"⚠️ Found {num_anomalies} potential anomalies in your data!")
    
    else:  # Smart Recommendations
        st.markdown("### 💡 AI Smart Recommendations")
        
        business_type = st.selectbox(
            "🏢 Business Type:",
            ["E-commerce", "SaaS", "Manufacturing", "Healthcare", "Finance"]
        )
        
        if st.button("🧠 Generate AI Recommendations"):
            with st.spinner("🤖 AI is crafting personalized recommendations..."):
                time.sleep(3)
            
            recommendations = {
                "E-commerce": [
                    "🛒 Implement abandoned cart recovery campaigns",
                    "📱 Optimize mobile checkout experience",
                    "🎯 Use dynamic pricing based on demand patterns",
                    "📦 Offer same-day delivery in high-density areas"
                ],
                "SaaS": [
                    "👥 Focus on user onboarding optimization",
                    "📈 Implement usage-based pricing tiers",
                    "🔄 Create automated customer success workflows",
                    "💬 Add in-app messaging for user engagement"
                ],
                "Manufacturing": [
                    "🏭 Implement predictive maintenance schedules",
                    "📊 Optimize supply chain inventory levels",
                    "⚡ Reduce energy consumption during peak hours",
                    "🔧 Automate quality control processes"
                ],
                "Healthcare": [
                    "👩‍⚕️ Implement telemedicine capabilities",
                    "📅 Optimize appointment scheduling systems",
                    "💊 Use AI for medication adherence tracking",
                    "📊 Implement population health analytics"
                ],
                "Finance": [
                    "🔒 Enhance fraud detection algorithms",
                    "📱 Develop mobile-first customer experiences",
                    "🤖 Implement robo-advisory services",
                    "📊 Use AI for credit risk assessment"
                ]
            }
            
            st.success("✅ AI recommendations generated!")
            
            for i, rec in enumerate(recommendations[business_type], 1):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                            padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>Recommendation {i}:</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # ROI Estimation
            st.markdown("### 💰 Estimated ROI Impact")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Revenue Increase", "15-25%", "🔥 High Impact")
            with col2:
                st.metric("💸 Cost Reduction", "10-20%", "💪 Medium Impact")
            with col3:
                st.metric("⏱️ Time Savings", "30-40%", "⚡ High Impact")

# Settings Page
elif page == "⚙️ Settings":
    st.markdown("# ⚙️ Dashboard Settings & Configuration")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ffeaa7, #fab1a0); 
                padding: 25px; border-radius: 20px; margin: 20px 0; text-align: center;">
        <h2 style="color: #2d3436; margin-bottom: 15px;">⚙️ Customize Your Experience</h2>
        <p style="color: #636e72; font-size: 1.1rem;">
            Personalize your dashboard to match your workflow and preferences!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme Settings
    st.markdown("### 🎨 Theme & Appearance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme_choice = st.selectbox(
            "🌈 Dashboard Theme:",
            ["🌈 Rainbow", "🌊 Ocean Blue", "🌅 Sunset", "🌲 Forest Green", "🌙 Dark Mode"]
        )
        
        font_size = st.slider("📝 Font Size:", 12, 24, 16)
        
    with col2:
        chart_style = st.selectbox(
            "📊 Chart Style:",
            ["Modern", "Classic", "Minimalist", "Vibrant"]
        )
        
        animation_enabled = st.checkbox("✨ Enable Animations", value=True)
    
    # Notification Settings
    st.markdown("### 🔔 Notification Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox("📧 Email Notifications", value=True)
        push_notifications = st.checkbox("📱 Push Notifications", value=False)
        
    with col2:
        alert_frequency = st.selectbox(
            "⏰ Alert Frequency:",
            ["Real-time", "Hourly", "Daily", "Weekly"]
        )
    
    # Data Settings
    st.markdown("### 📊 Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh Data", value=True)
        if auto_refresh:
            refresh_interval = st.slider("⏱️ Refresh Interval (minutes):", 1, 60, 15)
    
    with col2:
        data_retention = st.selectbox(
            "🗄️ Data Retention Period:",
            ["30 days", "90 days", "1 year", "Forever"]
        )
    
    # Export Settings
    st.markdown("### 📥 Export Preferences")
    
    export_format = st.multiselect(
        "📋 Default Export Formats:",
        ["CSV", "Excel", "PDF", "JSON", "PNG"],
        default=["CSV", "PNG"]
    )
    
    include_metadata = st.checkbox("📝 Include Metadata in Exports", value=True)
    
    # Privacy Settings
    st.markdown("### 🔒 Privacy & Security")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analytics_tracking = st.checkbox("📊 Allow Analytics Tracking", value=True)
        data_sharing = st.checkbox("🤝 Allow Data Sharing for Improvements", value=False)
    
    with col2:
        session_timeout = st.slider("⏰ Session Timeout (hours):", 1, 24, 8)
    
    # Save Settings
    st.markdown("### 💾 Save Configuration")
    
    if st.button("💾 Save All Settings", key="save_settings"):
        # Simulate saving settings
        with st.spinner("💾 Saving your preferences..."):
            time.sleep(2)
        
        st.success("✅ Settings saved successfully!")
        
        # Show current configuration
        st.markdown("### 📋 Current Configuration")
        
        settings_summary = f"""
        **Theme Settings:**
        - Theme: {theme_choice}
        - Font Size: {font_size}px
        - Chart Style: {chart_style}
        - Animations: {'Enabled' if animation_enabled else 'Disabled'}
        
        **Notifications:**
        - Email: {'Enabled' if email_notifications else 'Disabled'}
        - Push: {'Enabled' if push_notifications else 'Disabled'}
        - Frequency: {alert_frequency}
        
        **Data Configuration:**
        - Auto-refresh: {'Enabled' if auto_refresh else 'Disabled'}
        {'- Refresh Interval: ' + str(refresh_interval) + ' minutes' if auto_refresh else ''}
        - Data Retention: {data_retention}
        
        **Export Settings:**
        - Formats: {', '.join(export_format)}
        - Include Metadata: {'Yes' if include_metadata else 'No'}
        
        **Privacy & Security:**
        - Analytics Tracking: {'Enabled' if analytics_tracking else 'Disabled'}
        - Data Sharing: {'Enabled' if data_sharing else 'Disabled'}
        - Session Timeout: {session_timeout} hours
        """
        
        st.markdown(settings_summary)
    
    # Reset to Defaults
    if st.button("🔄 Reset to Default Settings"):
        st.warning("⚠️ This will reset all settings to their default values.")
        if st.button("✅ Confirm Reset"):
            st.success("🔄 Settings reset to defaults!")

# Footer with additional info
st.markdown("---")

# Social proof and stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #4ecdc4;">🚀</h3>
        <p><strong>Fast Performance</strong></p>
        <p>Lightning-fast analytics</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #ff6b6b;">🔒</h3>
        <p><strong>Secure & Private</strong></p>
        <p>Enterprise-grade security</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #45b7d1;">📊</h3>
        <p><strong>Advanced Analytics</strong></p>
        <p>AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #96ceb4;">🌟</h3>
        <p><strong>User Friendly</strong></p>
        <p>Intuitive interface</p>
    </div>
    """, unsafe_allow_html=True)

# Final footer
st.markdown("""
<div style="text-align: center; padding: 30px; 
            background: linear-gradient(135deg, #667eea, #764ba2); 
            border-radius: 15px; margin: 20px 0;">
    <h3 style="color: white; margin-bottom: 15px;">🎉 Thank You for Using DataViz Pro!</h3>
    <p style="color: #f8f9fa; font-size: 1.1rem; margin-bottom: 20px;">
        Built with ❤️ using Streamlit | Version 2.0 | © 2024 DataViz Pro
    </p>
    <div style="display: flex; justify-content: center; gap: 20px;">
        <span style="color: #4ecdc4;">📧 support@datavizpro.com</span>
        <span style="color: #ff6b6b;">🌐 www.datavizpro.com</span>
        <span style="color: #feca57;">📞 +1-800-DATAVIZ</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Add some interactive elements at the bottom
if st.button("🎊 Celebrate Success!", key="celebrate"):
    st.balloons()
    st.success("🎉 Congratulations! You've built an amazing Streamlit app!")

# Easter egg
if st.button("🎲 Random Fun Fact", key="fun_fact"):
    fun_facts = [
        "🚀 Streamlit was founded in 2018 and is now used by millions of developers!",
        "📊 Data visualization can increase comprehension by up to 400%!",
        "🤖 AI-powered analytics can reduce decision-making time by 5x!",
        "🌈 The human eye can distinguish about 10 million different colors!",
        "📈 Interactive dashboards improve user engagement by 70%!"
    ]
    
    fact = random.choice(fun_facts)
    st.info(fact)
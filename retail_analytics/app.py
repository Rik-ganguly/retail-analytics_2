import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv('data/sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    return df

df = load_data()

# ================================
# SIDEBAR
# ================================
st.sidebar.image("https://img.icons8.com/color/96/shop.png")
st.sidebar.title("🛒 Retail Analytics")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("📌 Navigate", [
    "🏠 Overview",
    "📊 Sales Analysis",
    "📈 Forecasting",
    "🏪 Store Segments"
])

st.sidebar.markdown("---")
region_filter = st.sidebar.multiselect(
    "Filter by Region",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

df_filtered = df[df['Region'].isin(region_filter)]

# ================================
# PAGE 1 — OVERVIEW
# ================================
if page == "🏠 Overview":
    st.title("🛒 Retail Sales Analytics Dashboard")
    st.markdown("### Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Revenue",
                  f"₹{df_filtered['Revenue'].sum()/1e6:.1f}M")
    with col2:
        st.metric("📦 Total Sales",
                  f"₹{df_filtered['Total_Sales'].sum()/1e6:.1f}M")
    with col3:
        st.metric("🏪 Total Stores",
                  df_filtered['Store_ID'].nunique())
    with col4:
        st.metric("📦 Total Products",
                  df_filtered['Product_ID'].nunique())

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Revenue by Region")
        reg = df_filtered.groupby('Region')['Revenue'].sum().reset_index()
        fig = px.bar(reg, x='Region', y='Revenue',
                     color='Region', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🛍️ Revenue by Category")
        cat = df_filtered.groupby('Product_Category')['Revenue'].sum().reset_index()
        fig = px.pie(cat, names='Product_Category',
                     values='Revenue', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# ================================
# PAGE 2 — SALES ANALYSIS
# ================================
elif page == "📊 Sales Analysis":
    st.title("📊 Sales Analysis")

    st.subheader("📈 Monthly Revenue Trend")
    monthly = df_filtered.groupby(['Year','Month'])['Revenue'].sum().reset_index()
    monthly['Period'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
    fig = px.line(monthly, x='Period', y='Revenue',
                  markers=True, template='plotly_dark',
                  title='Monthly Revenue Trend')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏪 Top 10 Stores")
        top = df_filtered.groupby('Store_ID')['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top, x='Store_ID', y='Revenue',
                     color='Revenue', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💳 Payment Mode")
        pay = df_filtered.groupby('Payment_Mode')['Revenue'].sum().reset_index()
        fig = px.pie(pay, names='Payment_Mode',
                     values='Revenue', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📦 Revenue by Product Category & Region")
    pivot = df_filtered.groupby(['Region','Product_Category'])['Revenue'].sum().reset_index()
    fig = px.bar(pivot, x='Product_Category', y='Revenue',
                 color='Region', barmode='group', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# ================================
# PAGE 3 — FORECASTING
# ================================
elif page == "📈 Forecasting":
    st.title("📈 Sales Forecasting")

    daily = df_filtered.groupby('Date')['Revenue'].sum().reset_index()
    daily = daily.sort_values('Date')
    daily['Day'] = daily['Date'].dt.day
    daily['Month'] = daily['Date'].dt.month
    daily['Year'] = daily['Date'].dt.year
    daily['Day_of_Week'] = daily['Date'].dt.dayofweek
    daily['Quarter'] = daily['Date'].dt.quarter
    daily['Lag_7'] = daily['Revenue'].shift(7)
    daily['Lag_30'] = daily['Revenue'].shift(30)
    daily['Rolling_7'] = daily['Revenue'].rolling(7).mean()
    daily['Rolling_30'] = daily['Revenue'].rolling(30).mean()
    daily.dropna(inplace=True)

    features = ['Day','Month','Year','Day_of_Week',
                'Quarter','Lag_7','Lag_30','Rolling_7','Rolling_30']
    X = daily[features]
    y = daily['Revenue']
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(n_estimators=300, learning_rate=0.05)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("🎯 Actual vs Predicted Revenue")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test.values, name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=y_pred, name='Predicted', line=dict(color='red', dash='dash')))
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 90-Day Future Forecast")
    future_dates = pd.date_range(start=daily['Date'].max()+pd.Timedelta(days=1), periods=90)
    future = pd.DataFrame({'Date': future_dates})
    future['Day'] = future['Date'].dt.day
    future['Month'] = future['Date'].dt.month
    future['Year'] = future['Date'].dt.year
    future['Day_of_Week'] = future['Date'].dt.dayofweek
    future['Quarter'] = future['Date'].dt.quarter
    future['Lag_7'] = daily['Revenue'].iloc[-7:].mean()
    future['Lag_30'] = daily['Revenue'].iloc[-30:].mean()
    future['Rolling_7'] = daily['Revenue'].iloc[-7:].mean()
    future['Rolling_30'] = daily['Revenue'].iloc[-30:].mean()
    future_pred = model.predict(future[features])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=daily['Date'], y=daily['Revenue'], name='Historical', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=future_dates, y=future_pred, name='Forecast', line=dict(color='green', dash='dash')))
    fig2.update_layout(template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

# ================================
# PAGE 4 — STORE SEGMENTS
# ================================
elif page == "🏪 Store Segments":
    st.title("🏪 Store Segmentation")

    store_features = df_filtered.groupby('Store_ID').agg(
        Total_Revenue=('Revenue', 'sum'),
        Avg_Daily_Sales=('Revenue', 'mean'),
        Volatility=('Revenue', 'std'),
        Total_Units=('Units_Sold', 'sum'),
        Avg_Rating=('Store_Rating', 'mean'),
        Avg_Discount=('Discount_Percentage', 'mean')
    ).reset_index()

    numeric_cols = ['Total_Revenue','Avg_Daily_Sales',
                    'Volatility','Total_Units',
                    'Avg_Rating','Avg_Discount']
    scaler = StandardScaler()
    X = scaler.fit_transform(store_features[numeric_cols])
    kmeans = KMeans(n_clusters=4, random_state=42)
    store_features['Cluster'] = kmeans.fit_predict(X)
    labels = {0:'🌟 High Performers', 1:'📈 Growing Stores',
              2:'⚠️ Underperformers', 3:'🔄 Seasonal Stores'}
    store_features['Segment'] = store_features['Cluster'].map(labels)

    fig = px.scatter(store_features,
                     x='Total_Revenue', y='Avg_Rating',
                     color='Segment', size='Total_Units',
                     hover_data=['Store_ID'],
                     template='plotly_dark',
                     title='Store Segments')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Store Details")
    st.dataframe(store_features[['Store_ID','Total_Revenue','Avg_Rating','Segment']])

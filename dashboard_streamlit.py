"""
Electricity Consumption Analytics Dashboard
============================================
Interactive Streamlit dashboard for exploring energy consumption patterns
and ML model predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Electricity Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .stMetric {
        background: rgba(102, 126, 234, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">⚡ Electricity Consumption Analytics</p>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('smart_home_energy_consumption_large.csv')
    return df

df = load_data()

# Preprocessing
@st.cache_data
def preprocess_data(df):
    df = df.copy()

    # Time features
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
    df['Hour'] = df['Time'].dt.hour
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    def categorize_time(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    df['TimeOfDay'] = df['Hour'].apply(categorize_time)
    df = df.dropna(subset=['Date', 'Time'])

    # Temperature deviation
    df['Temp_Deviation_From_Comfort'] = abs(df['Outdoor Temperature (°C)'] - 22)

    return df

df_processed = preprocess_data(df)

# Sidebar
st.sidebar.header("⚙️ Filters")

# Season filter
season_filter = st.sidebar.multiselect(
    "Select Season:",
    options=df_processed['Season'].unique(),
    default=df_processed['Season'].unique()
)

# Appliance filter
appliance_filter = st.sidebar.multiselect(
    "Select Appliance Type:",
    options=df_processed['Appliance Type'].unique(),
    default=df_processed['Appliance Type'].unique()
)

# Apply filters
filtered_df = df_processed[
    (df_processed['Season'].isin(season_filter)) &
    (df_processed['Appliance Type'].isin(appliance_filter))
]

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(filtered_df):,}")

with col2:
    st.metric("Mean Consumption", f"{filtered_df['Energy Consumption (kWh)'].mean():.2f} kWh")

with col3:
    st.metric("Max Consumption", f"{filtered_df['Energy Consumption (kWh)'].max():.2f} kWh")

with col4:
    st.metric("Min Consumption", f"{filtered_df['Energy Consumption (kWh)'].min():.2f} kWh")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Consumption Analysis",
    "🌡️ Temperature Impact",
    "⏰ Temporal Patterns",
    "🤖 ML Model",
    "📈 Predictions"
])

with tab1:
    st.header("Consumption Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Energy by appliance
        appliance_avg = filtered_df.groupby('Appliance Type')['Energy Consumption (kWh)'].mean().sort_values()
        fig = px.bar(appliance_avg.values, x=appliance_avg.index, y=appliance_avg.values,
                     title="Average Energy by Appliance", color=appliance_avg.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Energy by season
        season_avg = filtered_df.groupby('Season')['Energy Consumption (kWh)'].agg(['mean', 'std'])
        fig = px.bar(season_avg['mean'].values, x=season_avg.index, y=season_avg['mean'].values,
                     title="Average Energy by Season", color=season_avg['mean'].values,
                     color_continuous_scale='Plasma', error_y=season_avg['std'].values)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Distribution
    fig = px.histogram(filtered_df, x='Energy Consumption (kWh)', nbins=50,
                       title="Energy Consumption Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Temperature Impact Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot
        sample_df = filtered_df.sample(min(10000, len(filtered_df)))
        fig = px.scatter(sample_df, x='Outdoor Temperature (°C)', y='Energy Consumption (kWh)',
                         color='Hour', color_continuous_scale='Viridis',
                         title="Temperature vs Energy (colored by Hour)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Temperature deviation
        fig = px.scatter(filtered_df.sample(min(5000, len(filtered_df))),
                         x='Temp_Deviation_From_Comfort', y='Energy Consumption (kWh)',
                         trendline='ols', title="Energy vs Temperature Deviation from Comfort (22°C)")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    corr_cols = ['Energy Consumption (kWh)', 'Outdoor Temperature (°C)', 'Hour', 'Month',
                 'Household Size', 'Temp_Deviation_From_Comfort']
    corr_df = filtered_df[corr_cols].corr()
    fig = px.imshow(corr_df, text_auto='.2f', title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Temporal Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Hourly pattern
        hourly = filtered_df.groupby('Hour')['Energy Consumption (kWh)'].agg(['mean', 'std']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly['Hour'], y=hourly['mean'], mode='lines+markers',
                                  name='Mean', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=hourly['Hour'], y=hourly['mean']+hourly['std'],
                                  fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                  name='Std Dev'))
        fig.add_trace(go.Scatter(x=hourly['Hour'], y=hourly['mean']-hourly['std'],
                                  fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                  name='Std Dev'))
        fig.update_layout(title="Hourly Energy Consumption Pattern", xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Monthly pattern
        monthly = filtered_df.groupby('Month')['Energy Consumption (kWh)'].agg(['mean', 'std']).reset_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=monthly['mean'], marker_color=monthly['mean'],
                              error_y=monthly['std']))
        fig.update_layout(title="Monthly Energy Consumption Pattern", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Day of week pattern
    dow = filtered_df.groupby('DayOfWeek')['Energy Consumption (kWh)'].mean()
    dow.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig = px.line(dow, x=dow.index, y=dow.values, markers=True,
                  title="Day of Week Pattern")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Machine Learning Model")

    # Prepare features
    feature_cols = ['Outdoor Temperature (°C)', 'Household Size', 'Hour', 'Month',
                    'DayOfWeek', 'IsWeekend', 'Temp_Deviation_From_Comfort']

    # Encode categorical
    df_ml = filtered_df.copy()
    appliance_dummies = pd.get_dummies(df_ml['Appliance Type'], prefix='Appliance')
    df_ml = pd.concat([df_ml, appliance_dummies], axis=1)
    feature_cols.extend(appliance_dummies.columns.tolist())

    X = df_ml[feature_cols].fillna(0)
    y = df_ml['Energy Consumption (kWh)']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    col3.metric("R² Score", f"{r2:.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title="Top 15 Feature Importances", color='Importance',
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual vs Predicted', 'Residuals'))
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', alpha=0.1), row=1, col=1)
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                              mode='lines', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_pred, y=y_test.values - y_pred, mode='markers', alpha=0.1), row=1, col=2)
    fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=2)
    fig.update_layout(title="Model Performance")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Make Predictions")

    st.subheader("Interactive Prediction Tool")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider("Outdoor Temperature (°C)", -10.0, 45.0, 25.0)
        humidity = st.slider("Household Size", 1, 6, 3)
        hour = st.slider("Hour of Day", 0, 23, 12)
        month = st.slider("Month", 1, 12, 6)

    with col2:
        appliance = st.selectbox("Appliance Type", df['Appliance Type'].unique())
        season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])

    # Simple prediction based on averages
    similar = df_processed[
        (abs(df_processed['Outdoor Temperature (°C)'] - temp) < 5) &
        (df_processed['Hour'] == hour) &
        (df_processed['Appliance Type'] == appliance)
    ]

    if len(similar) > 0:
        predicted = similar['Energy Consumption (kWh)'].mean()
        st.success(f"Estimated Energy Consumption: **{predicted:.2f} kWh**")
        st.info(f"Based on {len(similar)} similar historical records")
    else:
        appliance_avg = df_processed[df_processed['Appliance Type'] == appliance]['Energy Consumption (kWh)'].mean()
        st.warning(f"No exact matches. Average for {appliance}: **{appliance_avg:.2f} kWh**")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data: Smart Home Energy Consumption Dataset")

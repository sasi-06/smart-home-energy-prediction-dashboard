import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Setup Streamlit Config
st.set_page_config(page_title="Energy Predictor ML", page_icon="⚡", layout="wide")

# Custom CSS for modern dynamic aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #38bdf8;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5);
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #fbbf24;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Tamil Nadu Electricity Bill Calculator based on New Tariff
def calculate_electricity_bill(units):
    if units <= 500:
        if units <= 100:
            return 0.0
        elif units <= 200:
            return (units - 100) * 2.25
        elif units <= 400:
            return (100 * 2.25) + (units - 200) * 4.50
        else:
            return (100 * 2.25) + (200 * 4.50) + (units - 400) * 6.00
    else:
        base_cost_500 = (300 * 4.50) + (100 * 6.00)
        if units <= 600:
            return base_cost_500 + (units - 500) * 8.00
        elif units <= 800:
            return base_cost_500 + (100 * 8.00) + (units - 600) * 9.00
        elif units <= 1000:
            return base_cost_500 + (100 * 8.00) + (200 * 9.00) + (units - 800) * 10.00
        else:
            return base_cost_500 + (100 * 8.00) + (200 * 9.00) + (200 * 10.00) + (units - 1000) * 11.00

# Helper function to load models safely
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        pca = joblib.load('models/pca.pkl')
        model = joblib.load('models/model.pkl')
        return preprocessor, pca, model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run the pipeline script first.")
        return None, None, None

preprocessor, pca, model = load_models()

# Dashboard Title
st.markdown("<h1 style='text-align: center;'>⚡ Smart Home Energy Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Machine learning system analyzing electricity consumption and forecasting usage.</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploratory Data Analysis", "📈 Model Evaluation", "🎯 Energy Predictor", "🧮 Precise Bill Calculator"])

with tab1:
    st.header("Dataset Overview")
    
    @st.cache_data
    def load_data():
        df = pd.read_csv("smart_home_energy_consumption_large.csv")
        # Load sample for performance if huge, but let's just use 10k random rows for fast visualization
        if df.shape[0] > 10000:
            return df.sample(10000, random_state=42)
        return df

    try:
        df_display = load_data()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Records Shown</div><div class='metric-value'>{df_display.shape[0]:,}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Consumption (kWh)</div><div class='metric-value'>{df_display['Energy Consumption (kWh)'].mean():.2f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Unique Appliance Types</div><div class='metric-value'>{df_display['Appliance Type'].nunique()}</div></div>", unsafe_allow_html=True)
            
        st.write("### Data Preview")
        st.dataframe(df_display.head(10), width="stretch" if hasattr(st, "dataframe") else None)
        
        # Plot distribution
        st.write("### Energy Consumption Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df_display['Energy Consumption (kWh)'], bins=50, kde=True, color='#8b5cf6', ax=ax)
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Dataset not found. Please ensure 'smart_home_energy_consumption_large.csv' is in the root directory.")

with tab2:
    st.header("Model Evaluation & Pipeline")
    st.write("These charts were automatically generated during the pipeline execution (Data Collection -> Standardisation -> PCA -> Random Forest).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("plots/actual_vs_predicted.png"):
            st.image("plots/actual_vs_predicted.png", caption="Model Accuracy mapping Predictions to Actual Values")
        else:
            st.info("Run the pipeline script to generate this plot.")
            
    with col2:
        if os.path.exists("plots/pca_variance.png"):
            st.image("plots/pca_variance.png", caption="Dimensionality Reduction (PCA explained variance > 95%)")
        else:
            st.info("Run the pipeline script to generate this plot.")
            
    if os.path.exists("plots/appliance_consumption.png"):
        st.write("### Appliance Analysis")
        st.image("plots/appliance_consumption.png")

with tab3:
    st.header("Predict Future Energy Usage")
    st.write("Adjust the parameters below to simulate a scenario and predict the specific item's Energy Consumption in kWh.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            appliances = st.multiselect("Appliance Types", ["Fridge", "Oven", "Dishwasher", "Heater", "Microwave", "Air Conditioning", "Computer", "TV", "Lights", "Washing Machine"], default=["Oven"])
            temp = st.slider("Outdoor Temperature (°C)", -20.0, 50.0, 15.0)
            hh_size = st.number_input("Household Size", min_value=1, max_value=10, value=3)
            
        with col2:
            season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
            time_str = st.time_input("Time of Day", value=pd.to_datetime("14:30").time())
            date_input = st.date_input("Date")
            
        submit = st.form_submit_button("Predict Energy Consumption")
        
    if submit:
        if model is None:
            st.error("Model artifacts missing. Cannot predict.")
        else:
            with st.spinner("Analyzing parameters through PCA pipeline..."):
                hour = time_str.hour
                month = date_input.month
                day_of_week = date_input.weekday()
                
                if not appliances:
                    st.warning("Please select at least one appliance.")
                else:
                    try:
                        total_prediction = 0
                        predictions = {}
                        
                        for app in appliances:
                            input_data = pd.DataFrame([{
                                'Outdoor Temperature (°C)': temp,
                                'Household Size': hh_size,
                                'Hour': hour,
                                'Month': month,
                                'DayOfWeek': day_of_week,
                                'Appliance Type': app,
                                'Season': season
                            }])
                            
                            processed = preprocessor.transform(input_data)
                            pca_transformed = pca.transform(processed)
                            pred = model.predict(pca_transformed)[0]
                            total_prediction += pred
                            predictions[app] = pred
                        
                        # Save to session state so it can be sent to Precise Bill Calculator
                        st.session_state.ml_predictions = predictions
                        st.success("Prediction complete! You can now import this data in the 'Precise Bill Calculator' tab.")
                        
                        # Project for 30 days to get a realistic estimated bill, instead of a 0 value for 1 day
                        bill_amount = calculate_electricity_bill(total_prediction * 30)
                        
                        st.markdown(f"""
                            <div style='display: flex; gap: 20px; margin-bottom: 20px;'>
                                <div style='flex: 1; background: linear-gradient(135deg, #10b981, #059669); padding: 30px; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                                    <h2 style='color: white; margin: 0;'>Total Predicted Consumption</h2>
                                    <p style='font-size: 3em; margin: 10px 0; font-weight: bold;'>{total_prediction:.2f} <span style='font-size: 0.4em;'>kWh/day</span></p>
                                    <p style='margin: 0; font-size: 0.9em; opacity: 0.8;'>Based on: {len(appliances)} appliances | {temp}°C | {season}</p>
                                </div>
                                <div style='flex: 1; background: linear-gradient(135deg, #f59e0b, #d97706); padding: 30px; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                                    <h2 style='color: white; margin: 0;'>Estimated Cost (TN Tariff)</h2>
                                    <p style='font-size: 3em; margin: 10px 0; font-weight: bold;'>₹{bill_amount:.2f}</p>
                                    <p style='margin: 0; font-size: 0.9em; opacity: 0.8;'>Based on 1-day projection</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("### Breakdown by Appliance")
                        for i in range(0, len(appliances), 3):
                            cols = st.columns(3)
                            for j in range(3):
                                if i + j < len(appliances):
                                    app = appliances[i + j]
                                    pred = predictions[app]
                                    with cols[j]:
                                        st.markdown(f"""
                                            <div class='metric-card' style='background-color: #1e293b; margin-bottom: 10px;'>
                                                <div class='metric-label'>{app}</div>
                                                <div class='metric-value' style='font-size: 1.5em; color: #38bdf8;'>{pred:.2f} <span style='font-size: 0.6em;'>kWh</span></div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                        
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

with tab4:
    st.header("🧮 Precise Bill Calculator")
    st.write("Calculate your exact Tamil Nadu electricity bill based on specific appliance wattages and daily usage. The data table below is editable—feel free to add, modify, or delete rows!")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        billing_days = st.number_input("Billing Cycle (Days)", min_value=1, max_value=365, value=30, help="For Tamil Nadu EB bi-monthly billing, enter 60.")
        
        # Option to import from ML Predictions tab
        has_ml_data = 'ml_predictions' in st.session_state and st.session_state.ml_predictions
        if has_ml_data:
            if st.button("📥 Import ML Predictions", help="Replace table with your calculated ML predictions"):
                st.session_state.use_ml_data = True
                
        if st.session_state.get('use_ml_data'):
            if st.button("↺ Reset Table", help="Revert to default items"):
                st.session_state.use_ml_data = False
                st.rerun()
                
    with col1:
        if st.session_state.get('use_ml_data') and has_ml_data:
            apps, qty, watts, hours = [], [], [], []
            for app, pred_kwh in st.session_state.ml_predictions.items():
                apps.append(f"{app} (Predicted)")
                qty.append(1)
                watts.append(1000.0) # Using 1000W so 1 unit * 1000W * Hours / 1000 = Hours = kWh
                hours.append(float(pred_kwh))
            
            df_calc = pd.DataFrame({
                "Appliance": apps,
                "Quantity": qty,
                "Wattage (W)": watts,
                "Hours Used/Day": hours
            })
            table_key = "cost_calculator_ml_mode"
        else:
            # Default dataframe
            default_data = {
                "Appliance": ["Air Conditioner", "Refrigerator", "Television", "Ceiling Fan", "LED Light", "Washing Machine"],
                "Quantity": [1, 1, 1, 4, 10, 1],
                "Wattage (W)": [1500.0, 200.0, 100.0, 75.0, 12.0, 500.0],
                "Hours Used/Day": [8.0, 24.0, 4.0, 12.0, 6.0, 1.0]
            }
            df_calc = pd.DataFrame(default_data)
            table_key = "cost_calculator_default"
            
        # Make the dataframe editable
        if hasattr(st, "data_editor"):
            edited_df = st.data_editor(
                df_calc, 
                num_rows="dynamic",
                width="stretch",
                key=table_key
            )
        else:
            edited_df = df_calc
            st.info("Update Streamlit to enable table editing.")
        
    # Validation & Computation
    if not edited_df.empty:
        try:
            # Calculate daily kWh per row: (Quantity * Wattage * Hours) / 1000
            edited_df["Daily kWh"] = (edited_df["Quantity"] * edited_df["Wattage (W)"] * edited_df["Hours Used/Day"]) / 1000.0
            
            # Aggregate totals
            total_daily_kwh = edited_df["Daily kWh"].sum()
            total_cycle_kwh = total_daily_kwh * billing_days
            
            bill_amount = calculate_electricity_bill(total_cycle_kwh)
            
            st.markdown(f"""
                <hr>
                <div style='display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;'>
                    <div style='flex: 1; background-color: #1e293b; padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #334155;'>
                        <h3 style='color: #94a3b8; font-size: 1.1em; text-transform: uppercase; margin: 0;'>Daily Consumption</h3>
                        <p style='font-size: 2.5em; color: #38bdf8; margin: 10px 0; font-weight: bold;'>{total_daily_kwh:.2f} <span style='font-size: 0.4em; color: #94a3b8;'>kWh/day</span></p>
                    </div>
                    <div style='flex: 1; background-color: #1e293b; padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #334155;'>
                        <h3 style='color: #94a3b8; font-size: 1.1em; text-transform: uppercase; margin: 0;'>Total Cycle Units</h3>
                        <p style='font-size: 2.5em; color: #fbbf24; margin: 10px 0; font-weight: bold;'>{total_cycle_kwh:.2f} <span style='font-size: 0.4em; color: #94a3b8;'>units ({billing_days} days)</span></p>
                    </div>
                    <div style='flex: 1; background: linear-gradient(135deg, #f59e0b, #d97706); padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>
                        <h3 style='color: white; font-size: 1.1em; text-transform: uppercase; margin: 0;'>Estimated Bill</h3>
                        <p style='font-size: 2.5em; color: white; margin: 10px 0; font-weight: bold;'>₹{bill_amount:.2f}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error("Error calculating costs. Please ensure all inputs are valid numbers.")
            st.error(str(e))

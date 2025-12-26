import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Engine Rating Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dashboard Look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6; 
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
    }
    /* Dark mode adjustment if needed, usually streamlit handles this */
</style>
""", unsafe_allow_html=True)

# Load Model and Columns
@st.cache_resource
def load_resources():
    try:
        with open('final_model_lgbm.pickle', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        return model, model_columns
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}. Please ensure 'final_model_lgbm.pickle' and 'model_columns.json' exist.")
        return None, None

model, model_columns = load_resources()

def main():
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.title("🔧 Configuration")
        st.markdown("Adjust the parameters below to simulate a car inspection.")
        
        st.divider()
        st.subheader("Inspection Details")
        
        inspection_datetime = st.date_input("Inspection Date")
        inspection_time = st.time_input("Inspection Time")
        inspection_start_time = pd.to_datetime(f"{inspection_datetime} {inspection_time}")
        
        odometer_reading = st.number_input("Odometer Reading (km)", min_value=0, value=50000, step=1000)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"])
        
        st.divider()
        st.subheader("Diagnostics")
        st.caption("Mark observed issues:")
        
        battery_jump_start = st.checkbox("🔋 Battery Jump Start")
        engine_oil_leak = st.checkbox("🛢️ Oil Leakage")
        engine_sound_abnormal = st.checkbox("🔊 Abnormal Sound")
        exhaust_smoke_white = st.checkbox("💨 White Smoke")
        clutch_hard = st.checkbox("⚙️ Hard Clutch")
        gear_shifting_hard = st.checkbox("🕹️ Hard Gear Shift")
        
        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    # --- Main Dashboard ---
    st.title("🚗 Engine Rating Analysis Dashboard")
    
    with st.expander("ℹ️ About this Project"):
        st.markdown("""
        **Objective:** Predict the quality rating (0-5) of a car's engine based on inspection data using a LightGBM regressor model.
        
        **How it works:**
        1.  **Input Data:** Enter details like Odometer reading, Fuel Type, and specific observed issues in the **Sidebar**.
        2.  **Model:** The app uses a pre-trained Gradient Boosting model to analyse these features.
        3.  **Output:** Returns a predicted rating and visually highlights potential component health issues.
        
        **Key Drivers:** Odometer Reading, Age (from Date), and Diagnosed Issues (Oil leaks, Noise, etc.).
        """)
    
    if not model:
        return

    # Prepare Data
    input_data = {
        'inspectionStartTime': inspection_start_time,
        'odometer_reading': odometer_reading,
        'fuel_type': fuel_type,
    }
    
    input_features = {col: 0 for col in model_columns}
    
    # 2. Time features
    input_features['inspection_hour'] = input_data['inspectionStartTime'].hour
    input_features['inspection_mon'] = input_data['inspectionStartTime'].month
    
    # Odometer
    input_features['odometer_reading'] = input_data['odometer_reading']
    
    # Fuel & Categorical Mapping
    # map fuel type selection to checking availability
    if fuel_type == "Petrol":
        fuel_col = "fuel_type_Petrol"
    elif fuel_type == "CNG":
        fuel_col = "fuel_type_Petrol___CNG"
    elif fuel_type == "LPG":
        fuel_col = "fuel_type_Petrol___LPG"
    elif fuel_type == "Electric":
        fuel_col = "fuel_type_Electric"
    elif fuel_type == "Hybrid":
        fuel_col = "fuel_type_Hybrid"
    else:
        fuel_col = None # Diesel or others might be baseline (all 0) or simply not present as specific col
    
    if fuel_col and fuel_col in input_features:
        input_features[fuel_col] = 1
        
    checkbox_map = {
        battery_jump_start: 'engineTransmission_battery_cc_value_Jump_Start', 
        engine_oil_leak: 'engineTransmission_engineOil_cc_value_Leaking',
        engine_sound_abnormal: 'engineTransmission_engineSound_cc_value_Engine_Auxiliary_Noise',
        exhaust_smoke_white: 'engineTransmission_exhaustSmoke_cc_value_White',
        clutch_hard: 'engineTransmission_clutch_cc_value_Hard',
        gear_shifting_hard: 'engineTransmission_gearShifting_cc_value_Not_Engaging'
    }
    
    for is_checked, col_name in checkbox_map.items():
        if is_checked and col_name in input_features:
            input_features[col_name] = 1
            
    X_input = pd.DataFrame([input_features])[model_columns]
    
    # Prediction
    try:
        prediction = model.predict(X_input)[0]
        rating_star = round(prediction)
        
        st.markdown("---")
        st.header("📊 Inspection Report")
        
        # --- Section 1: Health Profile ---
        st.subheader("1. Vehicle Health Profile")
        st.caption("This radar chart visualizes the condition of key vehicle systems based on your inspection inputs. A full polygon indicates perfect health, while dents indicate reported issues.")
        
        # ... (Radar Chart Logic) ...
        # Calculate scores
        # Logic matches previous turn, just visualizing nicely
        categories = ['Battery', 'Oil System', 'Sound', 'Exhaust', 'Clutch', 'Transmission']
        r_values = [
            100 if not battery_jump_start else 40,
            100 if not engine_oil_leak else 30,
            100 if not engine_sound_abnormal else 50,
            100 if not exhaust_smoke_white else 40,
            100 if not clutch_hard else 60,
            100 if not gear_shifting_hard else 60
        ]
        
        col_radar_chart, col_radar_text = st.columns([2, 1])
        with col_radar_chart:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=categories,
                fill='toself',
                name='Current Status',
                line_color='deepskyblue'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                margin=dict(l=40, r=40, t=30, b=30),
                height=300
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_radar_text:
            st.markdown(f"**Health Summary:**")
            issues_count = len([x for x in r_values if x < 100])
            if issues_count == 0:
                st.success("✅ No specific issues reported.")
            else:
                st.warning(f"⚠️ {issues_count} system(s) flagged for attention.")
                for cat, val in zip(categories, r_values):
                    if val < 100:
                        st.markdown(f"- **{cat}**: Issue Detected")

        st.markdown("---")

        # --- Section 2: Prediction ---
        st.subheader("2. AI Engine Rating")
        st.caption(f"The LightGBM model has analyzed the inputs (including {fuel_type} engine type, {odometer_reading} km usage) to predict the standardized rating.")

        # Metric Cards Row
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Predicted Score", value=f"{prediction:.2f} / 5.0")
        with kpi2:
             status_map = {
                5: ("Excellent", "green"),
                4: ("Very Good", "green"),
                3: ("Average", "orange"),
                2: ("Below Average", "orange"),
                1: ("Poor", "red"),
                0: ("Critical", "red")
            }
             p_int = max(0, min(5, int(round(prediction))))
             state_text, state_color = status_map.get(p_int, ("Unknown", "grey"))
             st.metric(label="Condition", value=state_text)
        with kpi3:
             st.metric(label="Mileage Impact", value="High" if odometer_reading > 100000 else "Moderate" if odometer_reading > 50000 else "Low")

        # Gauge Chart centered
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Rating Confidence"},
            gauge = {
                'axis': {'range': [0, 5], 'tickwidth': 1},
                'bar': {'color': state_color},
                'steps': [
                    {'range': [0, 2], 'color': 'rgba(255, 0, 0, 0.1)'},
                    {'range': [2, 4], 'color': 'rgba(255, 165, 0, 0.1)'},
                    {'range': [4, 5], 'color': 'rgba(0, 128, 0, 0.1)'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction}
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")

        # --- Section 3: Drivers ---
        st.subheader("3. What Drove This Score?")
        st.caption("The graph below shows the 'Feature Importance', indicating which factors the model heavily relies on when calculating ratings generally.")
        
        if hasattr(model, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'Feature': model_columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(8)
            
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                            color='Importance', color_continuous_scale='Bluered')
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_fi, use_container_width=True)
            
        st.info("💡 **Note:** High odometer reading and presence of 'Abnormal Sound' or 'Oil Leaks' typically penalize the score significantly.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()

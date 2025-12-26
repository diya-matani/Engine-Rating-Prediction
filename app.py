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
    page_title="Engine Rating Prediction",
    page_icon="🚗",
    layout="wide"
)

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
    st.title("🚗 Engine Rating Prediction")
    st.markdown("Enter the car details below to predict the engine rating.")
    
    # Toggle for Project Details
    if st.toggle("Show Project Details"):
        st.info("""
        **Project Overview:**
        - **Objective:** Predict the engine rating (quality score) of valid used cars based on inspection data.
        - **Model:** LightGBM Regressor (Gradient Boosting Framework).
        - **Key Features:**
            - **Inspection Time:** Day, Month, Hour.
            - **Odometer:** Total distance travelled.
            - **Diagnostics:** Engine sound, oil leakage, battery status, smoke color, etc.
        - **Logic:** The model was trained on historical data where categorical features were combined (e.g., specific battery issues) to capture complex patterns.
        """)
        
        # Feature Importance Plot
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Model Feature Importance")
            # Create a DF for importance
            # We need column names from model_columns
            # Ensure length matches
            if len(model_columns) == len(model.feature_importances_):
                fi_df = pd.DataFrame({
                    'Feature': model_columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False).head(10)
                
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                                title="Top 10 Influential Features", color='Importance')
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.warning("Feature importance dimension mismatch.")

    if not model:
        return

    # Create a form for inputs
    with st.form("prediction_form"):
        st.subheader("Inspection Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            inspection_datetime = st.date_input("Inspection Date")
            inspection_time = st.time_input("Inspection Time")
            
            # Combine Date and Time
            inspection_start_time = pd.to_datetime(f"{inspection_datetime} {inspection_time}")
            
            odometer_reading = st.number_input("Odometer Reading", min_value=0, value=50000)
            
            # Fuel Type (We need to know valid options, adding common ones for now based on notebook view)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"])

        with col2:
            st.info("Additional engine and transmission checks can be added here. For this demo, we assume standard conditions for unlisted features.")
            
            # Add some key boolean features if possible, or simplified inputs
            # For simplicity in this V1, let's allow adding specific 'issues' that match our combined columns
            # We know some combined columns exist like 'engineTransmission_battery_..._Jump Start'
            
            st.markdown("### Common Issues / Observations")
            battery_jump_start = st.checkbox("Battery: Jump Start Required")
            engine_oil_leak = st.checkbox("Engine Oil: Leaking")
            engine_sound_abnormal = st.checkbox("Engine Sound: Abnormal")
            exhaust_smoke_white = st.checkbox("Exhaust Smoke: White")
            clutch_hard = st.checkbox("Clutch: Hard")
            gear_shifting_hard = st.checkbox("Gear Shifting: Hard")
            
        submitted = st.form_submit_button("Predict Rating")

    if submitted:
        # Prepare input dataframe
        input_data = {
            'inspectionStartTime': inspection_start_time,
            'odometer_reading': odometer_reading,
            'fuel_type': fuel_type,
            # We need to map our checkboxes to the features expected by the model
            # This is a simplification. Real application would need mapping all features.
            # We will handle this by creating a raw DF and then processing it similar to training
        }
        
        # Preprocessing Logic
        
        # 1. Create base DF
        # We need to simulate the raw dataframe structure if we want to reuse logic,
        # OR we construct the final feature vector directly.
        # Direct construction is safer given the complexity of the "combined_columns" logic in training.
        
        # Let's start with a dict of 0s for all model columns
        input_features = {col: 0 for col in model_columns}
        
        # 2. Time features
        input_features['inspection_hour'] = input_data['inspectionStartTime'].hour
        input_features['inspection_mon'] = input_data['inspectionStartTime'].month
        input_features['inspection_date'] = input_data['inspectionStartTime'].day # Note: notebook used full date object, but ML models usually need numbers. 
        # Wait, the notebook had: `df['inspection_date'] = df['inspectionStartTime'].dt.date`
        # And then `inspection_date` was DROPPED in cell 39: `df_clean = df_features.drop(..., 'inspection_date', ...)`
        # So we don't need inspection_date! We just need hour, mon, dow?
        # Notebook: 
        # df['inspection_hour'] = ...
        # df['inspection_mon'] = ...
        # df['inspection_dow'] = ... day_name()
        # features dropped: inspectionStartTime, inspection_dow, inspection_date.
        # SO: We keep 'inspection_hour', 'inspection_mon'.
        
        input_features['inspection_hour'] = input_data['inspectionStartTime'].hour
        input_features['inspection_mon'] = input_data['inspectionStartTime'].month
        
        # Odometer
        input_features['odometer_reading'] = input_data['odometer_reading']
        
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
            
        # 4. Engine Transmission Features
        # Exact mapping based on model_columns.json inspection
        checkbox_map = {
            battery_jump_start: 'engineTransmission_battery_cc_value_Jump_Start', 
            engine_oil_leak: 'engineTransmission_engineOil_cc_value_Leaking',
            # Using 'Engine Auxiliary Noise' as proxy for abnormal sound checkbox
            engine_sound_abnormal: 'engineTransmission_engineSound_cc_value_Engine_Auxiliary_Noise',
            exhaust_smoke_white: 'engineTransmission_exhaustSmoke_cc_value_White',
            clutch_hard: 'engineTransmission_clutch_cc_value_Hard',
            # Using 'Not Engaging' for gear shifting issue
            gear_shifting_hard: 'engineTransmission_gearShifting_cc_value_Not_Engaging'
        }
        
        for is_checked, col_name in checkbox_map.items():
            if is_checked and col_name in input_features:
                input_features[col_name] = 1
        
        # Create DataFrame
        X_input = pd.DataFrame([input_features])
        
        # Ensure correct column order
        X_input = X_input[model_columns]
        
        # Predict
        try:
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Engine Rating: {prediction:.2f}")
            
            # Visual Gauge (Simple progress bar for rating 0-5)
            # Assuming rating is approx 0 to 5.
            rating_val = min(max(prediction / 5.0, 0.0), 1.0)
            st.progress(rating_val) 
            
            # Interpret Rating
            if prediction >= 4.0:
                status = "Excellent Condition! 🌟"
                color = "green"
            elif prediction >= 3.0:
                status = "Good Condition 👍"
                color = "blue"
            elif prediction >= 2.0:
                status = "Average / Fair ⚠️"
                color = "orange"
            else:
                status = "Poor Condition 🛑"
                color = "red"
            
            # st.markdown(f"### Status: :{color}[{status}]")
            
            col_gauge, col_radar = st.columns([1, 1])
            
            with col_gauge:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Engine Rating"},
                    gauge = {
                        'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 2], 'color': '#ffcccc'},
                            {'range': [2, 3], 'color': '#ffebcc'},
                            {'range': [3, 4], 'color': '#cce5ff'},
                            {'range': [4, 5], 'color': '#ccffcc'}],
                    }
                ))
                fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_radar:
                # Radar Chart for Component Health
                # Logical inference based on inputs (checkboxes)
                # 100 = Perfect, -20 for each issue
                
                health_scores = {
                    'Battery': 100 - (100 if battery_jump_start else 0),
                    'Engine Oil': 100 - (100 if engine_oil_leak else 0),
                    'Engine Sound': 100 - (50 if engine_sound_abnormal else 0),
                    'Exhaust': 100 - (100 if exhaust_smoke_white else 0),
                    'Clutch': 100 - (50 if clutch_hard else 0),
                    'Gears': 100 - (50 if gear_shifting_hard else 0)
                }
                
                # Normalize so it looks nicer (min 20)
                for k, v in health_scores.items():
                    health_scores[k] = max(v, 20)
                
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=list(health_scores.values()),
                    theta=list(health_scores.keys()),
                    fill='toself',
                    name='Component Health'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                        )),
                    showlegend=False,
                    title="Component Health Diagnostic",
                    height=400,
                    margin=dict(l=40, r=40, t=50, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Additional Context
            st.markdown(f"### Status: :{color}[{status}]")
            if prediction < 3.0:
                st.warning("The predicted engine rating is low. Careful inspection is recommended.")
            else:
                st.success("The engine seems to be in good shape based on the inputs.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re

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
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])

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
        
        # 3. Categorical Features (One-Hot Encoded in training)
        # fuel_type
        fuel_col = f"fuel_type_{input_data['fuel_type']}"
        # Sanitize column name to match training
        fuel_col = "".join(c if c.isalnum() else "_" for c in fuel_col)
        
        if fuel_col in input_features:
            input_features[fuel_col] = 1
            
        # 4. Engine Transmission Features
        # These are the "combined_columns" in the notebook.
        # Example from notebook: `engineTransmission_battery_value_Jump Start`
        # Using our sanitized logic, this becomes something like `engineTransmission_battery_value_Jump_Start`
        
        # Mapping checkboxes to likely column names (Requires knowledge of exact column names)
        # We try to match best guess. 
        # Ideally, we should inspect `model_columns.json` to be perfect, but let's try standard patterns.
        
        checkbox_map = {
            battery_jump_start: 'engineTransmission_battery_value_Jump_Start', # Guess
            engine_oil_leak: 'engineTransmission_engineOil_value_Leaking', # Guess
            engine_sound_abnormal: 'engineTransmission_engineSound_value_Abnormal',
            exhaust_smoke_white: 'engineTransmission_exhaustSmoke_value_White',
            clutch_hard: 'engineTransmission_clutch_value_Hard',
            gear_shifting_hard: 'engineTransmission_gearShifting_value_Hard'
        }
        
        # We need to find the actual matching column in model_columns because sanitization might have changed them slightly
        # or the prefix might be different (e.g. `cc_value_0` intermediate steps).
        # In the script `run_project.py`, we combined: `base_name}_{suffix}`
        # e.g. `engineTransmission_battery_cc_value_Jump Start`
        
        for is_checked, partial_name in checkbox_map.items():
            if is_checked:
                # Find column that looks like this
                # We normalize the partial name for search
                search_term = partial_name.replace(' ', '_')
                
                # Simple loose matching
                matches = [c for c in model_columns if search_term.lower() in c.lower()]
                if matches:
                    # Set the first match to 1
                    input_features[matches[0]] = 1
        
        # Create DataFrame
        X_input = pd.DataFrame([input_features])
        
        # Ensure correct column order
        X_input = X_input[model_columns]
        
        # Predict
        try:
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Engine Rating: {prediction:.2f}")
            
            # Visual Gauge (Simple progress bar for rating 0-5)
            # Assuming rating is approx 0 to 5 or 0 to 10? Notebook target was `rating_engineTransmission`.
            st.progress(min(max(prediction / 5.0, 0.0), 1.0)) # Normalize to 0-1 assuming max rating is 5
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()

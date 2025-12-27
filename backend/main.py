from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Engine Rating Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Resources
# We look for the model files in the current directory or the parent directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model_lgbm.pickle")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "model_columns.json")

# If not found in backend/, try parent directory (to avoid duplicating large files if running locally)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final_model_lgbm.pickle")
if not os.path.exists(COLUMNS_PATH):
    COLUMNS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_columns.json")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print("Error: final_model_lgbm.pickle not found.")
    model = None

try:
    with open(COLUMNS_PATH, 'r') as f:
        model_columns = json.load(f)
    print(f"Model columns loaded from {COLUMNS_PATH}")
except FileNotFoundError:
    print("Error: model_columns.json not found.")
    model_columns = []

class InspectionData(BaseModel):
    inspectionStartTime: str
    odometer_reading: int
    fuel_type: str
    diagnostics: dict = {}

@app.get("/")
def read_root():
    return {"status": "online", "message": "Engine Rating Prediction API is running"}

@app.post("/predict")
def predict_rating(data: InspectionData):
    if not model or not model_columns:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Parse Input
        input_start_time = pd.to_datetime(data.inspectionStartTime)
        
        # Initialize features with 0
        input_features = {col: 0 for col in model_columns}
        
        # Time features
        input_features['inspection_hour'] = input_start_time.hour
        input_features['inspection_mon'] = input_start_time.month
        
        # Odometer
        input_features['odometer_reading'] = data.odometer_reading
        
        # Fuel Type Mapping
        fuel_map = {
            "Petrol": "fuel_type_Petrol",
            "CNG": "fuel_type_Petrol___CNG",
            "LPG": "fuel_type_Petrol___LPG",
            "Electric": "fuel_type_Electric",
            "Hybrid": "fuel_type_Hybrid"
        }
        
        fuel_col = fuel_map.get(data.fuel_type)
        if fuel_col and fuel_col in input_features:
            input_features[fuel_col] = 1
            
        # Diagnostics Mapping
        # Keys match the frontend state names, Values match the model column names
        diagnostic_map = {
            "battery_jump_start": 'engineTransmission_battery_cc_value_Jump_Start', 
            "engine_oil_leak": 'engineTransmission_engineOil_cc_value_Leaking',
            "engine_sound_abnormal": 'engineTransmission_engineSound_cc_value_Engine_Auxiliary_Noise',
            "exhaust_smoke_white": 'engineTransmission_exhaustSmoke_cc_value_White',
            "clutch_hard": 'engineTransmission_clutch_cc_value_Hard',
            "gear_shifting_hard": 'engineTransmission_gearShifting_cc_value_Not_Engaging'
        }
        
        for key, col_name in diagnostic_map.items():
            if data.diagnostics.get(key, False) and col_name in input_features:
                input_features[col_name] = 1

        # Create DataFrame
        X_input = pd.DataFrame([input_features])[model_columns]
        
        # Predict
        prediction = model.predict(X_input)[0]
        prediction = max(0, min(5, prediction)) # Clamp between 0 and 5
        
        return {
            "prediction": prediction,
            "rating_text": get_rating_params(prediction)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def get_rating_params(score):
    if score >= 4.5: return "Excellent"
    if score >= 3.5: return "Very Good"
    if score >= 2.5: return "Average"
    if score >= 1.5: return "Below Average"
    return "Poor"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

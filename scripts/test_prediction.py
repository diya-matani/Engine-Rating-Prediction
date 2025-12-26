import os
import sys
import pickle
import pandas as pd
import sklearn

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import prepare_input_df

def test_prediction():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'model_pipeline.pkl')
    data_path = os.path.join(base_dir, 'data.xlsx')

    print(f"Python Executable: {sys.executable}")
    print(f"Scikit-learn Version: {sklearn.__version__}")
    print(f"Model Path: {model_path}")

    if not os.path.exists(model_path):
        print("ERROR: Model file not found!")
        return

    import joblib
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return

    # Create Dummy Input
    # We need to load data just to get the schema for prepare_input_df defaults
    from src.preprocess import load_data, basic_cleaning
    
    print("Loading data for schema...")
    try:
        df = load_data(data_path)
        df_clean = basic_cleaning(df)
        features_df = df_clean.drop(columns=['rating_engineTransmission']) if 'rating_engineTransmission' in df_clean.columns else df_clean
    except Exception as e:
        print(f"WARNING: Could not load data for schema, creating minimal dummy DF. Error: {e}")
        # Create a dummy features_df with at least one column to avoid crash if possible, 
        # but the model expects specific columns.
        print("Cannot proceed without schema for robust test.")
        return

    # Create a dummy user input
    user_inputs = {
        'year': 2015,
        'month': 6,
        'odometer_reading': 50000,
        'fuel_type': 'Petrol'
    }
    
    print("Preparing input dataframe...")
    try:
        input_df = prepare_input_df(user_inputs, features_df, "⚡ Quick Scan")
        print("Input DataFrame shape:", input_df.shape)
    except Exception as e:
        print(f"ERROR: Failed to prepare input df: {e}")
        return

    # Predict
    print("Attempting prediction...")
    try:
        pred = model.predict(input_df)
        print(f"SUCCESS! Prediction Result: {pred[0]}")
    except Exception as e:
        print(f"################################################")
        print(f"PREDICTION FAILED: {e}")
        print(f"################################################")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()

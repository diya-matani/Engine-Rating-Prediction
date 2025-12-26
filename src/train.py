import os
import pickle
import pandas as pd
import sys

# Ensure src can be imported if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocess import load_data, basic_cleaning, get_pipeline

def train_model():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data.xlsx')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print("Error: data.xlsx not found.")
        return
    
    print("Cleaning data...")
    df_clean = basic_cleaning(df)
    
    # Target and Features
    target_col = 'rating_engineTransmission'
    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    print(f"Dataset shape after cleaning: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build Pipeline
    print("Building pipeline...")
    preprocessor = get_pipeline(X_train)
    
    # Using RandomForest as per documentation ("Council of Mechanics")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    # Save
    import joblib
    model_path = os.path.join(models_dir, 'model_pipeline.pkl')
    
    # 1. Force remove old file to ensure no stale handle
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            print(f"Removed old model file at {model_path}")
        except Exception as e:
            print(f"WARNING: Could not remove old file: {e}")
            
    # 2. Save with joblib (better for sklearn)
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()

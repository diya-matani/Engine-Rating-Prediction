import pandas as pd
import numpy as np
import os
import sys
import pickle
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Add project root to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_data, basic_cleaning, get_pipeline

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data.xlsx')
    model_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(model_dir, 'model_pipeline.pkl')
    
    print("Loading data...")
    df = load_data(data_path)
    
    print("Cleaning data...")
    df_clean = basic_cleaning(df)
    
    target_col = 'rating_engineTransmission'
    if target_col not in df_clean.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    print("Shape:", X.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get preprocessor
    print("Building pipeline...")
    preprocessor = get_pipeline(X_train)
    
    # Define Model
    model = LGBMRegressor(random_state=42)
    
    # Full Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print(f"Training Score: {pipeline.score(X_train, y_train):.4f}")
    print(f"Test Score: {pipeline.score(X_test, y_test):.4f}")
    
    # Save
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print("Done.")

if __name__ == "__main__":
    train()

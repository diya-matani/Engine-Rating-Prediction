import pandas as pd
import numpy as np
import re
import pickle
import warnings
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def load_data(filepath):
    """Loads the dataset from a CSV or Excel file."""
    import os
    if not os.path.exists(filepath):
        # Try finding data.xlsx if default name wasn't found
        if filepath == 'Car_Features.csv' and os.path.exists('data.xlsx'):
            print(f"'{filepath}' not found. Loading 'data.xlsx' instead...")
            return pd.read_excel('data.xlsx', sheet_name='data')
        else:
            raise FileNotFoundError(f"File not found: {filepath}")
            
    print(f"Loading data from {filepath}...")
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath, sheet_name='data')
    else:
        # Fallback to csv if unknown extension but we'll try
        return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Performs data preprocessing and feature engineering.
    Includes time-based feature extraction, categorical encoding,
    and custom column combination logic.
    """
    print("Preprocessing data...")
    
    # Time based features
    # Convert inspectionStartTime to datetime if not already
    if df['inspectionStartTime'].dtype == 'O':
        df['inspectionStartTime'] = pd.to_datetime(df['inspectionStartTime'])
        
    df['inspection_hour'] = df['inspectionStartTime'].dt.hour
    df['inspection_mon'] = df['inspectionStartTime'].dt.month
    df['inspection_date'] = df['inspectionStartTime'].dt.date
    df['inspection_dow'] = df['inspectionStartTime'].dt.day_name()
    
    # Identify initial categorical columns related to engine transmission
    cat_cols = [label for label in df.columns if any(x in label for x in ['engineTransmission_'])]
    
    # Create a separate dataframe for these categorical columns
    df_cat = df[cat_cols]
    
    # One-hot encoding
    df_cat_encoded = pd.get_dummies(df_cat, dtype=bool, drop_first=True)
    
    # Custom Logic: Combine encoded columns based on suffixes
    # This logic matches cell 35 of the notebook
    suffixes = df_cat_encoded.columns.str.extract(r'.*_(.*)$')[0].unique()
    
    for suffix in suffixes:
        if pd.isna(suffix):
            continue
            
        matching_columns = [col for col in df_cat_encoded.columns if f'_{suffix}' in col]
        
        if not matching_columns:
            continue
            
        # Create base name by removing the numeric part and suffix
        # e.g., 'engineTransmission_battery_value_0_Jump Start' -> 'engineTransmission_battery_value'
        # The regex pattern in notebook was: r'_\d+_' + re.escape(suffix) + '$'
        # But we need to be careful to match the first column's pattern to get the base name correctly
        # Let's try to infer base name from the first matching column
        first_col = matching_columns[0]
        # In the notebook: base_name = re.sub(r'_\d+_' + re.escape(suffix) + '$', '', matching_columns[0])
        base_name = re.sub(r'_\d+_' + re.escape(suffix) + '$', '', first_col)
        
        # If the regex didn't change anything (no numeric part), we might need to adjust or just skip if it's already a base col
        # In the notebook, `df_cat_encoded` came from `df_cat` which was `df[cat_cols]`.
        # `cat_cols` had names like `engineTransmission_battery_cc_value_0`.
        # So `pd.get_dummies` would produce `engineTransmission_battery_cc_value_0_Suffix`.
        
        # Combine the selected columns (using OR operation)
        new_col_name = f'{base_name}_{suffix}'
        df_cat_encoded[new_col_name] = df_cat_encoded[matching_columns].any(axis=1).astype(int)
        
        # Drop the original individual columns after combining
        df_cat_encoded.drop(columns=matching_columns, inplace=True)
        
    # Combine encoded columns back with the original dataframe (minus original cat cols)
    # We use index alignment which is automatic in pandas
    df_features = pd.concat([df.drop(columns=cat_cols), df_cat_encoded], axis=1)
    
    # Identification of "combined_columns" for reference (from notebook)
    # combined_columns = [label for label in df_features.columns if any(x in label for x in ['engineTransmission_'])]
    
    # Cleaning steps from cell 39
    cols_to_drop = [
        'appointmentId',
        'inspectionStartTime',
        'inspection_dow',
        'inspection_date',
        'index', # In case it exists
        'engineTransmission_battery_cc_value_yes', # constant variable
        'engineTransmission_exhaustSmoke_cc_value_Leakage from manifold'
    ]
    
    # Drop only columns that actually exist
    existing_cols_to_drop = [c for c in cols_to_drop if c in df_features.columns]
    df_clean = df_features.drop(columns=existing_cols_to_drop, axis=1)
    
    # Final Feature Separation
    if 'rating_engineTransmission' not in df_clean.columns:
        raise ValueError("Target column 'rating_engineTransmission' not found/dropped unexpectedly.")
        
    X = df_clean.drop('rating_engineTransmission', axis=1)
    y = df_clean['rating_engineTransmission']
    
    # One-hot encode remaining categorical features (cell 41)
    # Identify categorical columns (object or category)
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        print(f"One-hot encoding remaining categorical features: {categorical_features}")
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
    # Sanitize column names for LightGBM to avoid "Do not support special JSON characters in feature name" error
    # Replace non-alphanumeric characters with underscores
    X.columns = ["".join (c if c.isalnum() else "_" for c in str(col)) for col in X.columns]
        
    return X, y

def train_model(X, y):
    """Trains the LightGBM model."""
    print("Splitting data and training model...")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Further split for validation if needed, but for final training usually we use more data?
    # The notebook re-combines X_train and X_val before final fit in Cell 51.
    # Cell 41: X_train, X_test... then X_train, X_val...
    # Cell 51: X_train_final = pd.concat((X_train, X_val), axis=0) ... fit on this.
    # This effectively means training on 80% of data (Train+Val) and testing on 20% (Test).
    # Since train_test_split above gives 80/20, X_train here IS (Train+Val) effectively.
    
    # Initialize Model
    # Using default LGBMRegressor as per Cell 51 of the notebook
    model = LGBMRegressor()
    
    # Fit Model
    # Notebook casts target to int: y_train_final.astype(int)
    # This might be for classification-like behavior or ordinal regression, 
    # but the problem is framed as regression (rating).
    # We will follow the notebook's instruction.
    model.fit(X_train, y_train.astype(int))
    
    # Evaluation
    train_score = model.score(X_train, y_train.astype(int))
    test_score = model.score(X_test, y_test.astype(int))
    
    print(f"Training R2 Score: {train_score:.4f}")
    print(f"Test R2 Score: {test_score:.4f}")
    
    return model

def save_model(model, filepath):
    """Saves the trained model to a pickle file."""
    print(f"Saving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Engine Rating Prediction Project")
    parser.add_argument('--data_path', type=str, default='Car_Features.csv', help='Path to the input CSV data')
    parser.add_argument('--model_path', type=str, default='final_model_lgbm.pickle', help='Path to save the trained model')
    args = parser.parse_args()
    
    try:
        df = load_data(args.data_path)
        X, y = preprocess_data(df)
        model = train_model(X, y)
        save_model(model, args.model_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

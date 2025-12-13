import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(filepath):
    """Loads the dataset from the specified path."""
    try:
        df = pd.read_excel(filepath, sheet_name='data', engine='openpyxl')
    except ValueError:
        df = pd.read_excel(filepath, engine='openpyxl')
    return df

def basic_cleaning(df):
    """Performs initial data cleaning steps found in the notebook."""
    df = df.copy()
    
    # 1. Drop known empty/useless columns
    cols_to_drop = [
        'engineTransmission_engine_cc_value_10', 
        'engineTransmission_engineOil_cc_value_9'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # 2. Drop comment columns
    comment_cols = [c for c in df.columns if 'comments' in c]
    df.drop(columns=comment_cols, inplace=True)
    
    # 3. Impute 'engineTransmission' columns with 'yes' (as per notebook/data dict)
    # Finding columns that likely need this imputation (categorical ones related to engine)
    # A simple heuristic from the notebook was checking for 'engineTransmission' in name.
    eng_trans_cols = [c for c in df.columns if 'engineTransmission' in c and c in df.columns]
    
    # We only want to fill NA with 'yes' for the categorical/object columns in this set
    # The notebook did: df[cols] = df[cols].fillna('yes')
    # Use fillna only on object columns to avoid errors if any are numeric (though most seem object)
    obj_eng_cols = df[eng_trans_cols].select_dtypes(include=['object']).columns
    df[obj_eng_cols] = df[obj_eng_cols].fillna('yes')
    
    # 4. Drop IDs if present (appointmentId) - usually not useful for prediction
    if 'appointmentId' in df.columns:
        df.drop(columns=['appointmentId'], inplace=True)
        
    # 5. Handle date columns? inspectionStartTime, year, month
    # For a baseline model, we might drop checking time or extract features. 
    # Let's keep 'year' and 'month', drop 'inspectionStartTime'
    if 'inspectionStartTime' in df.columns:
        df.drop(columns=['inspectionStartTime'], inplace=True)
        
    return df

def get_pipeline(df_train):
    """
    Creates a sklearn pipeline for preprocessing.
    Requires a sample df (df_train) to identify column types.
    """
    numeric_features = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from features if present
    target = 'rating_engineTransmission'
    if target in numeric_features:
        numeric_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

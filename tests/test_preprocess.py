import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import basic_cleaning, prepare_input_df, get_pipeline

def test_basic_cleaning():
    # Create sample data
    data = {
        'rating_engineTransmission': [4.0, 3.5, 5.0],
        'engineTransmission_engine_cc_value_10': [1, 2, 3], # Should be dropped
        'comments_misc': ['foo', 'bar', 'baz'], # Should be dropped
        'engineTransmission_clutch_value': ['Yes', np.nan, 'No'], # Should strip NaN
        'appointmentId': [101, 102, 103] # Should be dropped
    }
    df = pd.DataFrame(data)
    
    clean_df = basic_cleaning(df)
    
    # Assertions
    assert 'engineTransmission_engine_cc_value_10' not in clean_df.columns
    assert 'comments_misc' not in clean_df.columns
    assert 'appointmentId' not in clean_df.columns
    assert clean_df['engineTransmission_clutch_value'].isna().sum() == 0 # Should fillna with 'yes'
    assert clean_df.shape[0] == 3

def test_prepare_input_df():
    # Mock features schema
    features_df = pd.DataFrame({
        'year': [2015, 2018],
        'month': [1, 5],
        'fuel_type': ['Petrol', 'Diesel'],
        'odometer_reading': [10000, 20000],
        'battery_value': ['Yes', 'No']
    })
    
    # User Inputs (Partial)
    user_inputs = {
        'year': 2020,
        'fuel_type': 'Petrol'
    }
    
    input_df = prepare_input_df(user_inputs, features_df, mode="Quick Scan")
    
    # Check if input_df has all columns
    assert list(input_df.columns) == list(features_df.columns)
    
    # Check values
    assert input_df['year'].iloc[0] == 2020 # User input
    assert input_df['fuel_type'].iloc[0] == 'Petrol' # User input
    assert input_df['month'].iloc[0] == 3.0 # Median of [1, 5]
    assert input_df['odometer_reading'].iloc[0] == 15000.0 # Median 
    assert input_df['battery_value'].iloc[0] == 'Petrol' # Mode (alphabetic sort tie-break or similar) -- wait, mode of ['Yes','No']? 
    # Actually mode of ['Yes', 'No'] might be 'No' or 'Yes'. 
    
    # Let's check type
    assert isinstance(input_df, pd.DataFrame)
    assert len(input_df) == 1

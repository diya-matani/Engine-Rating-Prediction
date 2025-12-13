import pandas as pd
import numpy as np
import os
import sys

def load_and_inspect(filepath):
    print(f"Loading data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    try:
        # The notebook specified sheet_name='data', so we'll try that potentially, 
        # but let's just read default first or try-catch.
        try:
            df = pd.read_excel(filepath, sheet_name='data', engine='openpyxl')
        except ValueError:
             # Fallback if sheet name doesn't match
            print("Sheet 'data' not found, reading first sheet.")
            df = pd.read_excel(filepath, engine='openpyxl')
            
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print("First 5 rows:")
        print(df.head())
        print("\nInfo:")
        print(df.info())
        
        print("\nVerifying library imports...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import scipy
        import statsmodels
        import sklearn
        import xgboost
        import lightgbm
        print("All libraries imported successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Script is in scripts/, data is in root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data.xlsx')
    load_and_inspect(data_path)

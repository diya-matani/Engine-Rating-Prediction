import streamlit as st
import pandas as pd
import os
import sys
import joblib
import sklearn


# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import load_data, basic_cleaning, prepare_input_df, get_pipeline
from src import ui, analytics
import importlib
import src.preprocess
import src.ui
import src.analytics
importlib.reload(src.preprocess)
importlib.reload(src.ui)
importlib.reload(src.analytics)

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Set page config must be the very first Streamlit command
st.set_page_config(page_title="Engine Rating AI", layout="wide", initial_sidebar_state="expanded", page_icon="🏎️")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data.xlsx')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_v2.joblib')

@st.cache_data
def get_data():
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        return df
    return None

def retrain_model_in_app():
    """Retrains the model on the fly using available data."""
    try:
        df = load_data(DATA_PATH)
        df_clean = basic_cleaning(df)
        target_col = 'rating_engineTransmission'
        if target_col in df_clean.columns:
            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]
            
            # Simple Train (no split needed for final app model really, but sticking to protocol)
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.01, random_state=42) # Use almost all data
            
            preprocessor = get_pipeline(X_train)
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)) # Faster training
            ])
            model.fit(X_train, y_train)
            return model
    except Exception as e:
        st.error(f"Auto-retraining failed: {e}")
    return None

@st.cache_resource
def load_model_pipeline():
    model = None
    # 1. Try Loading from Disk
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            
            # --- VALIDATION (Smoketest) ---
            # We must verify the model actually works with the current scikit-learn version.
            # Loading is not enough; we must predict.
            try:
                df = load_data(DATA_PATH)
                df_clean = basic_cleaning(df)
                if 'rating_engineTransmission' in df_clean.columns:
                    X_val = df_clean.drop(columns=['rating_engineTransmission']).iloc[:5] # Test on 5 rows
                else:
                    X_val = df_clean.iloc[:5]
                
                # Try prediction
                model.predict(X_val)
                # If we get here, the model is valid!
            except Exception as val_e:
                print(f"Model validation failed ({val_e}). Discarding model.")
                model = None # Force retraining
                
        except Exception as e:
            st.warning(f"Could not load pre-trained model ({e}). Retraining...")
            model = None
    
    # 2. If missing or failed validations, RETRAIN
    if model is None:
        with st.spinner("⚠️ Incompatible Model Detected. Auto-retraining to fix..."):
            model = retrain_model_in_app()
            if model:
                st.success("✅ Model repaired successfully!")
                
    return model

def main():
    # 1. Setup Styles
    ui.render_style()
    
    # DEBUG BANNER
    st.warning(f"⚠️ DEBUG MODE: Running Scikit-learn Version: {sklearn.__version__} (Target: 1.3.1)")
    
    # 2. Load Resources
    df = get_data()
    model = load_model_pipeline()
    
    # 3. Process Data for Schema
    if df is not None:
        df_clean = basic_cleaning(df)
        if 'rating_engineTransmission' in df_clean.columns:
            features_df = df_clean.drop(columns=['rating_engineTransmission'])
        else:
            features_df = df_clean
    else:
        features_df = pd.DataFrame()
        df_clean = None

    # 4. Render Header
    ui.render_header()

    # 5. Render Sidebar & Capture Inputs
    user_inputs, mode, predict_btn = ui.render_sidebar(features_df)

    # 5b. Debug Info in Sidebar
    with st.sidebar:
        with st.expander("🔧 Runtime Diagnostics", expanded=True):
            st.write(f"**Python**: {sys.version.split()[0]}")
            st.write(f"**Scikit-learn**: {sklearn.__version__}")
            st.write(f"**Joblib**: {joblib.__version__}")
            st.write(f"**Executable**: {sys.executable}")
            st.write(f"**Model Path**: {MODEL_PATH}")
            
            # File Integrity Check
            if os.path.exists(MODEL_PATH):
                import hashlib
                with open(MODEL_PATH, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                file_size = os.path.getsize(MODEL_PATH)
                st.write(f"**Model Size**: {file_size/1024:.2f} KB")
                st.code(f"Hash: {file_hash[:10]}...", language="text")
            else:
                st.error("Model File Not Found!")

    # 6. Render Top Metrics
    ui.render_metrics(df_clean)

    # 7. Main Layout
    col_res, col_vis = st.columns([1, 2], gap="large")

    with col_res:
        st.subheader("Prediction")
        
        if predict_btn:
            if model is None:
                st.error("Model missing. Please train the model first.")
            else:
                with st.spinner("Processing..."):
                    try:
                        if mode == "📂 Batch Processing":
                            if 'batch_file' not in user_inputs:
                                st.error("Please upload a file first.")
                            else:
                                # Load batch data
                                batch_df = pd.read_csv(user_inputs['batch_file']) if user_inputs['batch_file'].name.endswith('.csv') else pd.read_excel(user_inputs['batch_file'])
                                
                                # We need to ensure batch_df has the same columns as features_df
                                # For simplicity, let's assume the user uploads valid data or we try to match
                                # We can use basic_cleaning on it just in case?
                                batch_clean = basic_cleaning(batch_df)
                                
                                # Align columns (missing -> median/mode from training data)
                                # iterating rows is slow, better to use vector ops
                                
                                # 1. Ensure all model features exist in batch_clean
                                for col in features_df.columns:
                                    if col not in batch_clean.columns:
                                        # Fill with default from features_df
                                        if features_df[col].dtype == 'object':
                                            batch_clean[col] = features_df[col].mode()[0]
                                        else:
                                            batch_clean[col] = features_df[col].median()
                                
                                # 2. Select only relevant columns
                                X_batch = batch_clean[features_df.columns]
                                
                                # Predict
                                preds = model.predict(X_batch)
                                batch_clean['Predicted Rating'] = preds
                                
                                st.success(f"Processed {len(batch_clean)} records.")
                                st.dataframe(batch_clean[['Predicted Rating'] + list(batch_clean.columns[:5])]) # Show rating and first few cols
                                
                                # Download
                                csv = batch_clean.to_csv(index=False).encode('utf-8')
                                st.download_button("Download Results", csv, "predictions.csv", "text/csv")
                        
                        else:
                            # Single Prediction
                            input_df = prepare_input_df(user_inputs, features_df, mode)
                            pred_val = model.predict(input_df)[0]
                            ui.render_prediction_result(pred_val)
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        # st.write("Input Debug:", user_inputs)
        else:
            ui.render_empty_state()

    with col_vis:
        analytics.render_analytics_tabs(df_clean)

    # 8. Footer
    ui.render_footer_explanation()

if __name__ == "__main__":
    main()

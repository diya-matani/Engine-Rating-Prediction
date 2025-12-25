import streamlit as st
import pandas as pd
import os
import pickle
from src.preprocess import load_data, basic_cleaning, prepare_input_df
from src import ui, analytics

# Set page config must be the very first Streamlit command
st.set_page_config(page_title="Engine Rating AI", layout="wide", initial_sidebar_state="expanded", page_icon="🏎️")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data.xlsx')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_pipeline.pkl')

@st.cache_data
def get_data():
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        return df
    return None

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

def main():
    # 1. Setup Styles
    ui.render_style()
    
    # 2. Load Resources
    df = get_data()
    model = get_model()
    
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
                        # Prepare Input
                        input_df = prepare_input_df(user_inputs, features_df, mode)
                        
                        # Predict
                        pred_val = model.predict(input_df)[0]
                        
                        # Show Result
                        ui.render_prediction_result(pred_val)
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.write("Input Debug:", user_inputs)
        else:
            ui.render_empty_state()

    with col_vis:
        analytics.render_analytics_tabs(df_clean)

    # 8. Footer
    ui.render_footer_explanation()

if __name__ == "__main__":
    main()

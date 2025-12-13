import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import load_data, basic_cleaning

# Set page config
st.set_page_config(page_title="Engine Rating AI", layout="wide", initial_sidebar_state="expanded", page_icon="🏎️")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #fcefe9; /* Very light warm tint */
    }
    .stApp header {
        background-color: transparent;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {

    }
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.3);
        transform: translateY(-2px);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', serif;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

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
    # Load resources
    df = get_data()
    model = get_model()
    
    # Process data to get schema
    if df is not None:
        df_clean = basic_cleaning(df)
        if 'rating_engineTransmission' in df_clean.columns:
            features_df = df_clean.drop(columns=['rating_engineTransmission'])
        else:
            features_df = df_clean
    else:
        features_df = pd.DataFrame() # Fallback

    # --- Sidebar ---
    with st.sidebar:
        st.title("Mode")
        
        # MODE SWITCHER
        mode = st.radio(
            "Analysis Mode",
            ["⚡ Quick Scan", "🔬 Detailed Inspection"],
            captions=["5 Key Parameters", "Full 60+ Point Check"]
        )
        
        st.markdown("---")
        
        with st.form("prediction_form"):
            user_inputs = {}
            st.subheader("Vehicle Parameters")
            
            # --- QUICK MODE INPUTS ---
            # These are always collected, but in Quick Mode they are the ONLY ones collected explicitly.
            # In Detailed Mode, we might want to group them differently, but for simplicity let's request them first.
            
            # Defaults
            def_year = int(features_df['year'].median()) if not features_df.empty else 2015
            def_month = int(features_df['month'].median()) if not features_df.empty else 6
            def_odo = int(features_df['odometer_reading'].median()) if not features_df.empty else 50000
            
            col_sb1, col_sb2 = st.columns(2)
            with col_sb1:
                input_year = st.number_input("Year", 2000, 2024, def_year)
            with col_sb2:
                input_month = st.number_input("Month", 1, 12, def_month)
                
            input_odo = st.number_input("Odometer (km)", 0, 500000, def_odo, step=1000)
            
            opts_fuel = features_df['fuel_type'].unique().tolist() if not features_df.empty else ["Petrol"]
            input_fuel = st.selectbox("Fuel Type", opts_fuel)
            
            # For Quick Mode, we might want one condition variable, e.g. Battery
            # For Detailed, we show everything.
            
            if mode == "⚡ Quick Scan":
                # Only one extra key feature
                input_battery = st.selectbox("Battery Condition", ["Yes", "No"], index=0, help="Is the battery working?")
                
                # Store these specific quick inputs
                user_inputs['year'] = input_year
                user_inputs['month'] = input_month
                user_inputs['odometer_reading'] = input_odo
                user_inputs['fuel_type'] = input_fuel
                # We need to map 'battery' to the actual column name used in training
                # Heuristic: find a column with 'battery_value'
                batt_col = next((c for c in features_df.columns if 'battery_value' in c), None)
                if batt_col:
                    user_inputs[batt_col] = input_battery
                    
                st.info("Other technical features are assumed 'Standard/Median'.")
                
            else:
                # DETAILED MODE
                # We need to capture the inputs above + all others
                user_inputs['year'] = input_year
                user_inputs['month'] = input_month
                user_inputs['odometer_reading'] = input_odo
                user_inputs['fuel_type'] = input_fuel
                
                # Generate dynamic inputs for the rest
                # Exclude the ones we just asked for
                # Also exclude ID or similar if any left
                
                skip_cols = ['year', 'month', 'odometer_reading', 'fuel_type', 'appointmentId']
                
                # Grouping Logic
                grouped_cols = {}
                others = []
                
                for col in features_df.columns:
                    if col in skip_cols:
                        continue
                    parts = col.split('_')
                    if len(parts) > 1 and parts[0] == 'engineTransmission':
                        group_name = parts[1].capitalize()
                        if group_name not in grouped_cols:
                            grouped_cols[group_name] = []
                        grouped_cols[group_name].append(col)
                    else:
                        others.append(col)
                
                # Render Groups
                for group in sorted(grouped_cols.keys()):
                    with st.expander(f"{group}", expanded=False):
                        for col in grouped_cols[group]:
                            label = col.replace(f"engineTransmission_{group.lower()}_", "").replace("_", " ").title()
                            label = label if label else col
                            
                            if features_df[col].dtype == 'object':
                                opts = features_df[col].unique().tolist()
                                opts = [x for x in opts if pd.notna(x)]
                                # Default to mode
                                pk = features_df[col].mode()[0] if not features_df[col].mode().empty else opts[0]
                                idx = opts.index(pk) if pk in opts else 0
                                user_inputs[col] = st.selectbox(label, opts, index=idx, key=col)
                            else:
                                mid = float(features_df[col].median())
                                user_inputs[col] = st.number_input(label, value=mid, key=col)
                
                if others:
                    with st.expander("Miscellaneous"):
                        for col in others:
                             if features_df[col].dtype == 'object':
                                opts = features_df[col].unique().tolist()
                                user_inputs[col] = st.selectbox(col, opts, key=col)
                             else:
                                user_inputs[col] = st.number_input(col, value=float(features_df[col].median()), key=col)

            predict_btn = st.form_submit_button("Start Analysis")

    # --- Main Area ---
    st.markdown("""
        <div style="text-align: left;">
            <h1 style="margin-bottom: 0;">Engine Rating AI</h1>
            <p style="color: #666; font-size: 1.1rem;">Advanced Diagnostic & Prediction System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # --- Top Metrics Row ---
    if df is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Records", f"{len(df_clean):,}")
        with c2:
            avg_rtg = df_clean['rating_engineTransmission'].mean()
            delta_rtg = avg_rtg - 3.5 # dummy baseline
            st.metric("Avg Rating", f"{avg_rtg:.2f}", delta=f"{delta_rtg:.2f}")
        with c3:
            st.metric("Avg Odometer", f"{int(df_clean['odometer_reading'].mean()):,} km")
        with c4:
            top_fuel = df_clean['fuel_type'].mode()[0]
            st.metric("Dominant Fuel", top_fuel)
        with c5:
            yr_max = df_clean['year'].max()
            st.metric("Newest Model", int(yr_max))
            
    st.markdown("---")

    # --- Layout Split ---
    col_res, col_vis = st.columns([1, 2], gap="large")

    with col_res:
        st.subheader("Prediction")
        
        if predict_btn:
            if model is None:
                st.error("Model missing. Train first.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Prepare input vector
                        if mode == "⚡ Quick Scan":
                            # We have partial user_inputs. Need to fill the rest from defaults.
                            # We take a 'template' row (e.g. median/mode of entire dataset)
                            # Effectively, creating a dict of defaults
                            
                            full_input = {}
                            for col in features_df.columns:
                                if col in user_inputs:
                                    full_input[col] = user_inputs[col]
                                else:
                                    # Fallback
                                    if features_df[col].dtype == 'object':
                                        full_input[col] = features_df[col].mode()[0]
                                    else:
                                        full_input[col] = features_df[col].median()
                            input_df = pd.DataFrame([full_input])
                        else:
                            # Detailed Mode: we tried to collect everything in Sidebar
                            # user_inputs should cover all relevant keys if our loops were correct.
                            # Just to be safe, fill any missing keys from defaults as well
                             full_input = {}
                             for col in features_df.columns:
                                if col in user_inputs:
                                    full_input[col] = user_inputs[col]
                                else:
                                    if features_df[col].dtype == 'object':
                                        full_input[col] = features_df[col].mode()[0]
                                    else:
                                        full_input[col] = features_df[col].median()
                             input_df = pd.DataFrame([full_input])
                        
                        # Reorder columns to match training if necessary (Pipeline usually handles by name, but good practice)
                        input_df = input_df[features_df.columns]

                        pred_val = model.predict(input_df)[0]
                        
                        # Visualization of Result
                        st.markdown(f"""
                        <div style="
                            padding: 30px;
                            border-radius: 15px;
                            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                            text-align: center;
                            border: 1px solid #ddd;
                        ">
                            <h3 style="margin:0; color:#555;">Engine Health Score</h3>
                            <h1 style="font-size: 72px; color: #ff4b4b; margin: 10px 0;">{pred_val:.1f}</h1>
                            <p style="color:#888;">Range: 1.0 (Worst) - 5.0 (Best)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(min(pred_val/5.0, 1.0))
                        
                        if pred_val >= 4.0:
                            st.success("Verdict: Excellent Condition")
                        elif pred_val >= 3.0:
                            st.info("Verdict: Good / Average")
                        else:
                            st.error("Verdict: Poor Condition")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.write("Input Debug:", user_inputs)
        else:
            st.info("Select Mode & Analyze")
            
            # Fun placeholder
            st.markdown("""
            <div style="padding: 20px;border-radius: 10px; border: 1px dashed #ccc; text-align: center; color: #aaa;">
                Waiting for input...
            </div>
            """, unsafe_allow_html=True)

    with col_vis:
        # All visualizations as requested
        st.subheader("Deep Dive Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlations", "Trends", "Diagnostics"])
        
        if df is not None:
            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Ratings Histogram**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df_clean['rating_engineTransmission'], bins=10, kde=True, color='salmon', ax=ax)
                    st.pyplot(fig)
                with col_b:
                    st.write("**Ratings by Fuel**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x='fuel_type', y='rating_engineTransmission', data=df_clean, palette='Set2', hue='fuel_type', legend=False, ax=ax)
                    st.pyplot(fig)
                    
            with tab2:
                st.write("**Feature Correlation Heatmap** (Top numerical features)")
                num_df = df_clean.select_dtypes(include=['float64', 'int64'])
                if not num_df.empty:
                    corr = num_df.corr().round(2)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
            
            with tab3:
                st.write("**Yearly Quality Trend**")
                trend_data = df_clean.groupby('year')['rating_engineTransmission'].mean().reset_index()
                st.line_chart(trend_data, x='year', y='rating_engineTransmission')
                
            with tab4:
                st.write("**Component Failure Rates**")
                # Simple analysis: Count of non-'yes' in engineTransmission columns if logical
                # Heuristic: 'No' or 'Weak' or other values might indicate failure vs 'Yes' (if yes=good) or vice versa
                # Actually, data dictionary said nulls imputed as 'yes', often meaning 'Issue Present' is the explicit value?
                # Without deep domain knowledge, just showing frequency of a key column like 'engineSound'
                
                snd_col = next((c for c in df_clean.columns if 'engineSound_value' in c), None)
                if snd_col:
                    st.write(f"**Engine Sound Distribution**")
                    st.bar_chart(df_clean[snd_col].value_counts())

    st.markdown("---")
    st.subheader("How the AI Works")
    st.markdown("""
    **The "Magic" Explained Simply**
    
    Imagine you have a council of **100 expert mechanics**. Each mechanic has seen thousands of cars and knows exactly what to look for—engine sound, smoke color, oil leaks, and more.
    
    When you enter your car's details:
    1. **Data Collection**: We gather all the clues you provided (mileage, year, specific engine conditions).
    2. **The Council Votes**: Each of the 100 digital mechanics (trees in our "Random Forest") independently looks at the data and gives their rating.
    3. **Final Verdict**: We average all their votes to give you a single, highly accurate score.
    
    **Why this is better:**
    - **No Bias**: It relies on data, not just one person's opinion.
    - **Consensus**: One mistake by a single "mechanic" is corrected by the 99 others.
    - **Experience**: The model has "seen" patterns from the entire dataset that a human might miss.
    """)
    
    st.markdown("---")
    st.subheader("Under the Hood: Technical Deep Dive")
    st.write("For those interested in the engineering behind the prediction, here is the complete logic flow.")
    
    # Flowchart using Graphviz
    graph = """
    digraph ModelFlow {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="#f0f2f6", fontname="Segoe UI"];
        edge [color="#555"];
        
        Data [label="User Input\n(Features)", fillcolor="#ffe8cc"];
        Pre [label="Preprocessing\n(Scaling & Encoding)"];
        Forest [label="Random Forest\n(100 Decision Trees)", fillcolor="#d4edda"];
        Vote [label="Aggregated Vote\n(Averaging)"];
        Output [label="Final Rating\n(1.0 - 5.0)", fillcolor="#ffcccb"];
        
        Data -> Pre;
        Pre -> Forest;
        Forest -> Vote;
        Vote -> Output;
        
        subgraph cluster_0 {
            label = "Decision Tree Logic (Example)";
            style = dashed;
            color = grey;
            node [style=filled, fillcolor="white"];
            
            Q1 [label="Mileage < 50k?"];
            Q2 [label="Oil Leak?"];
            R1 [label="High Rating"];
            R2 [label="Low Rating"];
            
            Q1 -> Q2 [label="Yes"];
            Q2 -> R1 [label="No"];
            Q2 -> R2 [label="Yes"];
        }
    }
    """
    st.graphviz_chart(graph)
    
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.markdown("""
        ### Input Features
        The model analyzes **over 60 data points** across key categories:
        - **Core Specs**: Year, Month, Mileage, Fuel Type.
        - **Engine Condition**: Starter motor, Engine Oil, Coolant, Mounts, Sound, Smoke.
        - **Transmission**: Gear shifting, Clutch feel/slippage.
        - **Peripherals**: Battery voltage, Radiator fan.
        """)
        
    with c_d2:
        st.markdown("""
        ### The Algorithm: Random Forest
        We use an ensemble learning method called **Random Forest Regressor**.
        
        - **Ensemble**: Instead of one complex formula, we use 100 simple decision trees.
        - **Bootstrapping**: Each tree is trained on a random subset of the data, ensuring diversity.
        - **Robustness**: This method is highly resistant to "overfitting" (memorizing the data) and handles missing values well.
        - **Training**: It was trained on thousands of historical inspection reports where the final engine rating was known.
        """)

if __name__ == "__main__":
    main()

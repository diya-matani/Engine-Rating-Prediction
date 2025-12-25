import streamlit as st
import pandas as pd
import numpy as np

def render_style():
    """Renders usage-specific CSS."""
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

def render_header():
    """Renders the main application header."""
    st.markdown("""
        <div style="text-align: left;">
            <h1 style="margin-bottom: 0;">Engine Rating AI</h1>
            <p style="color: #666; font-size: 1.1rem;">Advanced Diagnostic & Prediction System</p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar(features_df):
    """
    Renders the sidebar and returns the user input dictionary and selected mode.
    
    Args:
        features_df: DataFrame containing the feature columns to generate inputs for.
    
    Returns:
        tuple: (user_inputs, mode, predict_btn)
    """
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
            
            if mode == "⚡ Quick Scan":
                input_battery = st.selectbox("Battery Condition", ["Yes", "No"], index=0, help="Is the battery working?")
                
                # Store these specific quick inputs
                user_inputs['year'] = input_year
                user_inputs['month'] = input_month
                user_inputs['odometer_reading'] = input_odo
                user_inputs['fuel_type'] = input_fuel
                
                batt_col = next((c for c in features_df.columns if 'battery_value' in c), None)
                if batt_col:
                    user_inputs[batt_col] = input_battery
                    
                st.info("Other technical features are assumed 'Standard/Median'.")
                
            else:
                # DETAILED MODE
                user_inputs['year'] = input_year
                user_inputs['month'] = input_month
                user_inputs['odometer_reading'] = input_odo
                user_inputs['fuel_type'] = input_fuel
                
                skip_cols = ['year', 'month', 'odometer_reading', 'fuel_type', 'appointmentId']
                
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
                    # Use a clean key for expander to avoid duplicates if any
                    with st.expander(f"{group}", expanded=False):
                        for col in grouped_cols[group]:
                            label = col.replace(f"engineTransmission_{group.lower()}_", "").replace("_", " ").title()
                            label = label if label else col
                            
                            if features_df[col].dtype == 'object':
                                opts = features_df[col].unique().tolist()
                                opts = [x for x in opts if pd.notna(x)]
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
            
    return user_inputs, mode, predict_btn

def render_metrics(df_clean):
    """Renders the top row of summary metrics."""
    if df_clean is not None:
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

def render_prediction_result(pred_val):
    """Renders the visual result of the prediction."""
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

def render_empty_state():
    """Renders the placeholder state before prediction."""
    st.info("Select Mode & Analyze")
    st.markdown("""
    <div style="padding: 20px;border-radius: 10px; border: 1px dashed #ccc; text-align: center; color: #aaa;">
        Waiting for input...
    </div>
    """, unsafe_allow_html=True)
    
def render_footer_explanation():
    """Renders the 'How it works' and 'Technical Deep Dive' sections."""
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

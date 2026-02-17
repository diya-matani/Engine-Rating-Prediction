import streamlit as st
import traceback

# Set page config MUST be the first streamlit command
st.set_page_config(
    page_title="Engine Rating Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    import matplotlib
    matplotlib.use('Agg') # Prevent GUI errors on headless server
    import pandas as pd
    import numpy as np
    import pickle
    import json
    import re
    import plotly.graph_objects as go
    import plotly.express as px
except Exception as e:
    st.error(f"Failed to import dependencies: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Custom CSS for Dashboard Look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6; 
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
    }
    /* Dark mode adjustment if needed, usually streamlit handles this */
</style>
""", unsafe_allow_html=True)

# Load Model and Columns
@st.cache_resource
def load_resources():
    try:
        with open('final_model_lgbm.pickle', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        return model, model_columns
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}. Please ensure 'final_model_lgbm.pickle' and 'model_columns.json' exist.")
        return None, None

@st.cache_data
def load_data():
    """Loads the historical dataset for visualization."""
    try:
        # Try CSV first
        df = pd.read_csv('Car_Features.csv')
    except:
        try:
            # Fallback to Excel
            df = pd.read_excel('data.xlsx', sheet_name='data')
        except Exception as e:
            return None
    
    # Preprocessing for text optimization
    if 'inspectionStartTime' in df.columns:
        df['inspectionStartTime'] = pd.to_datetime(df['inspectionStartTime'])
        df['inspection_date'] = df['inspectionStartTime'].dt.date
        df['inspection_mon'] = df['inspectionStartTime'].dt.month_name()
        df['inspection_month_num'] = df['inspectionStartTime'].dt.month
        df['inspection_hour'] = df['inspectionStartTime'].dt.hour
        df['inspection_dow'] = df['inspectionStartTime'].dt.day_name()
    return df

try:
    model, model_columns = load_resources()
    df_history = load_data()
except Exception as e:
    st.error(f"Critical Error during initialization: {e}")
    # We don't stop here, but main will likely fail if model is None
    model, model_columns = None, None
    df_history = None

def main():
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.title("🔧 Configuration")
        st.markdown("Adjust the parameters below to simulate a car inspection.")
        
        st.divider()
        st.subheader("Inspection Details")
        
        inspection_datetime = st.date_input("Inspection Date")
        inspection_time = st.time_input("Inspection Time")
        inspection_start_time = pd.to_datetime(f"{inspection_datetime} {inspection_time}")
        
        odometer_reading = st.number_input("Odometer Reading (km)", min_value=0, value=50000, step=1000)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"])
        
        st.divider()
        st.subheader("Diagnostics")
        st.caption("Mark observed issues:")
        
        battery_jump_start = st.checkbox("🔋 Battery Jump Start")
        engine_oil_leak = st.checkbox("🛢️ Oil Leakage")
        engine_sound_abnormal = st.checkbox("🔊 Abnormal Sound")
        exhaust_smoke_white = st.checkbox("💨 White Smoke")
        clutch_hard = st.checkbox("⚙️ Hard Clutch")
        gear_shifting_hard = st.checkbox("🕹️ Hard Gear Shift")
        
        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    # --- Main Dashboard ---
    
    # Initialize Session State for Theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'

    def toggle_theme():
        if st.session_state.theme == 'dark':
            st.session_state.theme = 'light'
        else:
            st.session_state.theme = 'dark'

    # Determine current mode
    is_light_mode = st.session_state.theme == 'light'
    
    # Dynamic CSS for App Background & Toggle Button
    if is_light_mode:
        app_bg_color = "#fcfbf4"
        text_color = "#2c3e50"
        toggle_icon = "🌙" # Button to switch TO dark
        btn_tooltip = "Switch to Dark Mode"
    else:
        app_bg_color = "#0e1117" # Standard Streamlit Dark or custom #1a1a1a
        text_color = "#e0e0e0"
        toggle_icon = "☀️" # Button to switch TO light
        btn_tooltip = "Switch to Light Mode"

    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {app_bg_color};
            color: {text_color};
        }}
        /* Force text color on metrics and headers if needed */
        h1, h2, h3, h4, h5, h6, p, div, span {{
            color: {text_color} !important;
        }}
        /* Metric Cards */
        .metric-card {{
            background-color: {'#ffffff' if is_light_mode else '#262730'}; 
            border-left: 5px solid #ff4b4b;
        }}
        /* Button specific styling - targeted to Main area (Theme Toggle) */
        section[data-testid="stMain"] div.stButton > button:first-child {{
            background-color: transparent;
            border: 1px solid {text_color};
            color: {text_color};
            border-radius: 50%;
            height: 45px;
            width: 45px;
            font-size: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Header Layout
    col_header, col_toggle = st.columns([8, 1])
    with col_header:
        st.title("🚗 Engine Rating Analysis Dashboard")
    with col_toggle:
        st.button(toggle_icon, on_click=toggle_theme, help=btn_tooltip)

    with st.expander("ℹ️ About this Project"):
        st.markdown("""
        **Objective:** Predict the quality rating (0-5) of a car's engine based on inspection data using a LightGBM regressor model.
        
        **How it works:**
        1.  **Input Data:** Enter details like Odometer reading, Fuel Type, and specific observed issues in the **Sidebar**.
        2.  **Model:** The app uses a pre-trained Gradient Boosting model to analyse these features.
        3.  **Output:** Returns a predicted rating and visually highlights potential component health issues.
        
        **Key Drivers:** Odometer Reading, Age (from Date), and Diagnosed Issues (Oil leaks, Noise, etc.).
        """)
    
    if not model:
        return

    # Prepare Data
    input_data = {
        'inspectionStartTime': inspection_start_time,
        'odometer_reading': odometer_reading,
        'fuel_type': fuel_type,
    }
    
    input_features = {col: 0 for col in model_columns}
    
    # 2. Time features
    input_features['inspection_hour'] = input_data['inspectionStartTime'].hour
    input_features['inspection_mon'] = input_data['inspectionStartTime'].month
    
    # Odometer
    input_features['odometer_reading'] = input_data['odometer_reading']
    
    # Fuel & Categorical Mapping
    # map fuel type selection to checking availability
    if fuel_type == "Petrol":
        fuel_col = "fuel_type_Petrol"
    elif fuel_type == "CNG":
        fuel_col = "fuel_type_Petrol___CNG"
    elif fuel_type == "LPG":
        fuel_col = "fuel_type_Petrol___LPG"
    elif fuel_type == "Electric":
        fuel_col = "fuel_type_Electric"
    elif fuel_type == "Hybrid":
        fuel_col = "fuel_type_Hybrid"
    else:
        fuel_col = None # Diesel or others might be baseline (all 0) or simply not present as specific col
    
    if fuel_col and fuel_col in input_features:
        input_features[fuel_col] = 1
        
    checkbox_map = {
        battery_jump_start: 'engineTransmission_battery_cc_value_Jump_Start', 
        engine_oil_leak: 'engineTransmission_engineOil_cc_value_Leaking',
        engine_sound_abnormal: 'engineTransmission_engineSound_cc_value_Engine_Auxiliary_Noise',
        exhaust_smoke_white: 'engineTransmission_exhaustSmoke_cc_value_White',
        clutch_hard: 'engineTransmission_clutch_cc_value_Hard',
        gear_shifting_hard: 'engineTransmission_gearShifting_cc_value_Not_Engaging'
    }
    
    for is_checked, col_name in checkbox_map.items():
        if is_checked and col_name in input_features:
            input_features[col_name] = 1
            
    X_input = pd.DataFrame([input_features])[model_columns]
    
    # Prediction
    try:
        prediction = model.predict(X_input)[0]
        
        # --- Helper for Honeywell Aesthetic ---
        def apply_honeywell_style(fig, title=None, height=300, light_mode=False):
            if light_mode:
                # Cream Mode Colors
                paper_bg = '#fcfbf4'
                plot_bg = '#fcfbf4'
                font_color = '#2c3e50'
                grid_color = '#ebdccb'
            else:
                # Dark Mode Colors (Deep Gray)
                paper_bg = 'rgba(0,0,0,0)'
                plot_bg = 'rgba(0,0,0,0)'
                font_color = '#e0e0e0'
                grid_color = '#333333'

            fig.update_layout(
                title=title,
                paper_bgcolor=paper_bg,
                plot_bgcolor=plot_bg,
                font=dict(color=font_color),
                height=height,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True, gridcolor=grid_color),
                yaxis=dict(showgrid=True, gridcolor=grid_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return fig

        # --- Top-Level Tabs ---
        tab_inspection, tab_trends, tab_data = st.tabs(["🔍 Current Inspection", "📈 Market Intelligence", "📄 Inspection Records"])
        
        # ================= TAB 1: Real-time Analysis =================
        with tab_inspection:
            st.markdown("### Real-time Quality Monitor")
            
            # --- Row 1: KPI Overview ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Predicted Output", f"{prediction:.2f} / 5.0")
            with kpi2:
                 status_map = {5: "Excellent", 4: "Very Good", 3: "Average", 2: "Below Average", 1: "Poor", 0: "Critical"}
                 p_int = max(0, min(5, int(round(prediction))))
                 state_text = status_map.get(p_int, "Unknown")
                 st.metric("Quality Grade", state_text)
            with kpi3:
                 total_checks = 6
                 failed_checks = sum([1 for x in [battery_jump_start, engine_oil_leak, engine_sound_abnormal, exhaust_smoke_white, clutch_hard, gear_shifting_hard] if x])
                 health_pct = ((total_checks - failed_checks) / total_checks) * 100
                 st.metric("System Health", f"{health_pct:.0f}%", delta=f"-{failed_checks} Flags" if failed_checks > 0 else "Optimal")
            with kpi4:
                 st.metric("Odometer Impact", "High" if odometer_reading > 100000 else "Normal")

            st.divider()

            # --- Row 2: Trend Analysis ---
            st.subheader("📉 Predictive Degradation Analysis")
            st.caption("Projected rating decay if usage continues without maintenance.")
            
            future_odo = [odometer_reading + i*10000 for i in range(6)]
            future_preds = []
            for odo in future_odo:
                feat_copy = input_features.copy()
                feat_copy['odometer_reading'] = odo
                X_temp = pd.DataFrame([feat_copy])[model_columns]
                pred_temp = model.predict(X_temp)[0]
                future_preds.append(max(0, pred_temp))
                
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=future_odo, y=future_preds, mode='lines+markers', name='Projected', line=dict(color='#ff4b4b', width=3)))
            apply_honeywell_style(fig_trend, "Future Rating Prediction", height=320, light_mode=is_light_mode)
            st.plotly_chart(fig_trend, use_container_width=True)
            st.info(f"ℹ️ Prediction drops to **{future_preds[-1]:.2f}** after +50k km.")

            st.divider()

            # --- Row 3: Component Analysis ---
            col_pie, col_radar = st.columns(2)
            
            with col_pie:
                st.subheader("System Status")
                labels = ['Healthy', 'Issues']
                values = [total_checks - failed_checks, failed_checks]
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=['#00cc96', '#ef553b']))])
                apply_honeywell_style(fig_pie, height=300, light_mode=is_light_mode)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_radar:
                st.subheader("Health Profile")
                categories = ['Battery', 'Oil System', 'Sound', 'Exhaust', 'Clutch', 'Transmission']
                r_values = [
                    100 if not battery_jump_start else 40,
                    100 if not engine_oil_leak else 30,
                    100 if not engine_sound_abnormal else 50,
                    100 if not exhaust_smoke_white else 40,
                    100 if not clutch_hard else 60,
                    100 if not gear_shifting_hard else 60
                ]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=r_values, theta=categories, fill='toself', name='Score', line_color='#636efa'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor= 'rgba(0,0,0,0)' if not is_light_mode else '#f0f2f6'))
                apply_honeywell_style(fig_radar, height=300, light_mode=is_light_mode)
                st.plotly_chart(fig_radar, use_container_width=True)

            # --- Row 4: Explainability ---
            st.divider()
            st.subheader("🧠 Decision Factors")
            if hasattr(model, 'feature_importances_'):
                fi_df = pd.DataFrame({'Feature': model_columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance')
                apply_honeywell_style(fig_fi, height=350, light_mode=is_light_mode)
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_fi, use_container_width=True)


        # ================= TAB 2: Historical Trends (Linear & Explained) =================
        with tab_trends:
            if df_history is not None:
                st.markdown("### 📈 Market Intelligence Hub")
                st.markdown("Detailed analytics derived from the historical inspection database. Scroll down for sequential analysis.")
                
                # 1. Volume Trend
                st.divider()
                st.subheader("1. Timeline Analysis")
                st.markdown("**Insight:** Monitors the daily volume of inspections over time. The orange line represents the 7-day moving average, smoothing out daily fluctuations to reveal the true underlying trend.")
                if 'inspection_date' in df_history.columns:
                    ts = df_history.groupby('inspection_date').size().reset_index(name='Original')
                    ts['Mean'] = ts['Original'].rolling(7).mean()
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=ts['inspection_date'], y=ts['Original'], name='Daily', line=dict(color='rgba(135, 206, 235, 0.5)', width=1)))
                    fig_ts.add_trace(go.Scatter(x=ts['inspection_date'], y=ts['Mean'], name='7-Day Avg', line=dict(color='#FFA15A', width=2)))
                    apply_honeywell_style(fig_ts, "Daily Inspection Volume Trend", height=380, light_mode=is_light_mode)
                    st.plotly_chart(fig_ts, use_container_width=True)
                
                # 2. Seasonality
                st.divider()
                st.subheader("2. Seasonal Patterns")
                st.markdown("**Insight:** Aggregates inspections by month to highlight seasonal peaks. Identifying these high-activity periods helps in resource planning and predicting market influx.")
                if 'inspection_mon' in df_history.columns:
                    mc = df_history.groupby(['inspection_month_num', 'inspection_mon']).size().reset_index(name='count').sort_values('inspection_month_num')
                    fig_mon = px.bar(mc, x='inspection_mon', y='count', color='count', color_continuous_scale='Blues')
                    apply_honeywell_style(fig_mon, "Monthly Inspection Volume", height=380, light_mode=is_light_mode)
                    st.plotly_chart(fig_mon, use_container_width=True)

                # 3. Vintage
                st.divider()
                st.subheader("3. Asset Vintage")
                st.markdown("**Insight:** Break down of inspected vehicles by their registration year. This distribution reveals the age profile of the fleet being serviced, indicating whether the market is dominated by newer or older vehicles.")
                year_col = 'year' if 'year' in df_history.columns else ('registrationYear' if 'registrationYear' in df_history.columns else None)
                if year_col:
                    rc = df_history[year_col].value_counts().sort_index()
                    fig_reg = px.bar(x=rc.index, y=rc.values, labels={'x':'Year', 'y':'Count'}, color_discrete_sequence=['#636efa'])
                    apply_honeywell_style(fig_reg, "Vehicle Registration Year Distribution", height=380, light_mode=is_light_mode)
                    st.plotly_chart(fig_reg, use_container_width=True)
                else:
                    st.info("Registration Year data unavailable.")

                # 4. Intensity
                st.divider()
                st.subheader("4. Operational Intensity")
                st.markdown("**Insight:** A histogram showing the frequency of daily inspection counts. It answers the question: 'How often do we perform X inspections a day?', helping to benchmark operational capacity.")
                if 'inspection_date' in df_history.columns:
                    dc = df_history.groupby('inspection_date').size().reset_index(name='count')
                    fig_hist = px.histogram(dc, x="count", marginal="violin", nbins=30, color_discrete_sequence=['#00cc96'])
                    apply_honeywell_style(fig_hist, "Distribution of Daily Inspection Counts", height=380, light_mode=is_light_mode)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # 5. Odometer
                st.divider()
                st.subheader("5. Usage Profile (Odometer)")
                st.markdown("**Insight:** Visualizes the spread of mileage across all inspected cars. The boxplot highlights the median usage and identifies 'outliers'—vehicles with exceptionally high or low mileage.")
                if 'odometer_reading' in df_history.columns:
                    fig_box = px.box(df_history, y="odometer_reading", points="outliers", color_discrete_sequence=['#AB63FA'])
                    apply_honeywell_style(fig_box, "Odometer Reading Distribution", height=400, light_mode=is_light_mode)
                    st.plotly_chart(fig_box, use_container_width=True)
                        
                # 6. Heatmap
                st.divider()
                st.subheader("6. Correlation Analysis")
                st.markdown("**Insight:** A heatmap revealing hidden relationships between variables. Darker or distinct colors indicate strong positive or negative correlations, guiding which factors (like age or mileage) most strongly influence ratings.")
                cols = ['year', 'inspection_month_num', 'odometer_reading', 'rating_engineTransmission']
                ex = [c for c in cols if c in df_history.columns]
                if len(ex)>1: 
                    corr = df_history[ex].corr()
                    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', origin='lower', aspect="auto")
                    apply_honeywell_style(fig_corr, "Feature Correlation Matrix", height=450, light_mode=is_light_mode)
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Historical data not available for trends.")


        # ================= TAB 3: Data Table (Refined) =================
        with tab_data:
            st.markdown("### 📄 Inspection Records Database")
            if df_history is not None:
                # Filter Control
                show_weak = st.checkbox("⚠️ Show Weak Engines Only (Rating ≤ 2)", value=False, help="Filter to show only cars with poor engine ratings.")
                
                # Column Configuration & Renaming
                # Map raw names to clean names
                col_mapping = {
                    'inspectionStartTime': 'Inspection Date',
                    'year': 'Year', 
                    'rating_engineTransmission': 'Engine Rating',
                    'odometer_reading': 'Odometer (km)',
                    'fuel_type': 'Fuel Type'
                }
                
                # Create view with specific columns
                available_cols = [c for c in col_mapping.keys() if c in df_history.columns]
                df_view = df_history[available_cols].rename(columns=col_mapping).copy()
                
                # Apply Filter
                if show_weak and 'Engine Rating' in df_view.columns:
                    df_view = df_view[df_view['Engine Rating'] <= 2]
                
                # Ensure 'Inspection Date' is sorted descending by default if user hasn't sorted manually
                if 'Inspection Date' in df_view.columns:
                     df_view = df_view.sort_values('Inspection Date', ascending=False)

                # Display with proper formatting
                # Using st.column_config for better UX
                st.dataframe(
                    df_view.style.background_gradient(cmap='Reds', subset=['Engine Rating'], vmin=0, vmax=5).format({'Engine Rating': '{:.1f}'}),
                    use_container_width=True,
                    height=500,
                    column_config={
                        "Engine Rating": st.column_config.NumberColumn(
                            "Engine Rating",
                            help="AI Predicted Rating (0-5)",
                            min_value=0,
                            max_value=5,
                            step=0.1,
                            format="%.1f ⭐",
                        ),
                        "Inspection Date": st.column_config.DatetimeColumn(
                            "Inspection Date",
                            format="D MMM YYYY, HH:mm",
                        ),
                    },
                    hide_index=True
                )
                
                # Download Button
                csv = df_view.to_csv(index=False).encode('utf-8')
                st.download_button("Download Data (CSV)", csv, "inspection_data.csv", "text/csv")
            else:
                st.info("No historical data to display.")

    except Exception as e:
        import traceback
        st.error(f"Prediction Error: {e}")
        st.markdown(f"```\n{traceback.format_exc()}\n```")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Engine Rating Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

model, model_columns = load_resources()
df_history = load_data()

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
    st.title("🚗 Engine Rating Analysis Dashboard")
    
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
        
        # Calculate r_values needed for charts (Fixing NameError)
        categories = ['Battery', 'Oil System', 'Sound', 'Exhaust', 'Clutch', 'Transmission']
        r_values = [
            100 if not battery_jump_start else 40,
            100 if not engine_oil_leak else 30,
            100 if not engine_sound_abnormal else 50,
            100 if not exhaust_smoke_white else 40,
            100 if not clutch_hard else 60,
            100 if not gear_shifting_hard else 60
        ]
        
        # --- Top-Level Tabs (Honeywell Style) ---
        tab_inspection, tab_trends, tab_data = st.tabs(["🔍 Current Inspection", "📈 Market Intelligence", "📄 Inspection Records"])
        
        # TAB 1: Real-time Analysis
        with tab_inspection:
            st.markdown("### Real-time Quality Monitor")
            # KPIs
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Predicted Output", f"{prediction:.2f} / 5.0")
            with kpi2:
                 status_map = {5: "Excellent", 4: "Very Good", 3: "Average", 2: "Below Average", 1: "Poor", 0: "Critical"}
                 p_int = max(0, min(5, int(round(prediction))))
                 state_text = status_map.get(p_int, "Unknown")
                 st.metric("Quality Grade", state_text, delta_color="normal")
            with kpi3:
                 total_checks = 6
                 failed_checks = sum([1 for x in [battery_jump_start, engine_oil_leak, engine_sound_abnormal, exhaust_smoke_white, clutch_hard, gear_shifting_hard] if x])
                 health_pct = ((total_checks - failed_checks) / total_checks) * 100
                 st.metric("System Health", f"{health_pct:.0f}%", delta=f"-{failed_checks} Flags" if failed_checks > 0 else "Optimal")
            with kpi4:
                 st.metric("Odometer Impact", "High" if odometer_reading > 100000 else "Normal")

            st.divider()

            # Trend Analysis (Degradation)
            st.subheader("📉 Predictive Degradation Analysis")
            st.caption("Projected rating decay over future mileage.")
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
            fig_trend.update_layout(title="Future Rating Prediction", xaxis_title="Odometer", yaxis_title="Rating", height=300, margin=dict(t=30, b=20))
            st.plotly_chart(fig_trend, use_container_width=True)
            st.info(f"ℹ️ Prediction drops to **{future_preds[-1]:.2f}** after +50k km.")

            st.divider()

            # Component Analysis
            col_pie, col_radar = st.columns(2)
            with col_pie:
                st.subheader("System Status")
                labels = ['Healthy', 'Issues']
                values = [total_checks - failed_checks, failed_checks]
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=['#00cc96', '#ef553b']))])
                fig_pie.update_layout(height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_radar:
                st.subheader("Health Profile")
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=r_values, theta=categories, fill='toself', name='Score', line_color='#636efa'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig_radar, use_container_width=True)
                
            # Explainability
            st.divider()
            st.subheader("🧠 Decision Factors")
            if hasattr(model, 'feature_importances_'):
                fi_df = pd.DataFrame({'Feature': model_columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance')
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=300, margin=dict(t=0, b=0))
                st.plotly_chart(fig_fi, use_container_width=True)

        # TAB 2: Historical Trends
        with tab_trends:
            if df_history is not None:
                st.markdown("### 📈 Historical Market Analysis")
                
                # 1. Reg Year
                st.subheader("1. Vehicle Vintage")
                if 'registrationYear' in df_history.columns:
                    rc = df_history['registrationYear'].value_counts().sort_index()
                    st.plotly_chart(px.bar(x=rc.index, y=rc.values, labels={'x':'Year', 'y':'Count'}).update_layout(height=300), use_container_width=True)
                
                c1, c2 = st.columns(2)
                with c1: 
                    st.subheader("2. Daily Intensity")
                    if 'inspection_date' in df_history.columns:
                        dc = df_history.groupby('inspection_date').size().reset_index(name='count')
                        st.plotly_chart(px.histogram(dc, x="count", marginal="violin").update_layout(height=300), use_container_width=True)
                with c2:
                    st.subheader("3. Seasonality")
                    if 'inspection_mon' in df_history.columns:
                        mc = df_history.groupby(['inspection_month_num', 'inspection_mon']).size().reset_index(name='count').sort_values('inspection_month_num')
                        st.plotly_chart(px.bar(mc, x='inspection_mon', y='count').update_layout(height=300), use_container_width=True)
                        
                st.subheader("4. Volume Trends")
                if 'inspection_date' in df_history.columns:
                    ts = df_history.groupby('inspection_date').size().reset_index(name='Original')
                    ts['Mean'] = ts['Original'].rolling(7).mean()
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=ts['inspection_date'], y=ts['Original'], name='Daily'))
                    fig_ts.add_trace(go.Scatter(x=ts['inspection_date'], y=ts['Mean'], name='7-Day Avg'))
                    fig_ts.update_layout(height=350, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                st.markdown("---")
                st.caption("Deep Dive Distributions")
                c3, c4 = st.columns(2)
                with c3:
                    if 'odometer_reading' in df_history.columns:
                        st.plotly_chart(px.box(df_history, y="odometer_reading", title="Odometer Outliers").update_layout(height=350), use_container_width=True)
                with c4:
                    cols = ['year', 'inspection_month_num', 'odometer_reading', 'rating_engineTransmission']
                    ex = [c for c in cols if c in df_history.columns]
                    if len(ex)>1: st.plotly_chart(px.imshow(df_history[ex].corr(), text_auto=True).update_layout(height=350), use_container_width=True)
            else:
                st.warning("Historical data not available.")

        # TAB 3: Data Table
        with tab_data:
            st.markdown("### 📄 Inspection Records")
            if df_history is not None:
                st.dataframe(df_history.head(200), use_container_width=True)
                csv = df_history.head(1000).to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "data.csv")
            else:
                st.info("No data.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()

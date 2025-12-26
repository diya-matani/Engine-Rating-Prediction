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
        rating_star = round(prediction)
        
        st.markdown("---")
        st.header("🏭 Engine Analytics & Quality Assurance")
        
        # --- Row 1: KPI Overview ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric("Predicted Output", f"{prediction:.2f} / 5.0", delta_color="normal")
        with kpi2:
             status_map = {
                5: ("Excellent", "green"),
                4: ("Very Good", "green"),
                3: ("Average", "orange"),
                2: ("Below Average", "orange"),
                1: ("Poor", "red"),
                0: ("Critical", "red")
            }
             p_int = max(0, min(5, int(round(prediction))))
             state_text, state_color = status_map.get(p_int, ("Unknown", "grey"))
             st.metric("Quality Grade", state_text)
        with kpi3:
             # Calculate a simple "Confidence Health %"
             total_checks = 6
             failed_checks = sum([1 for x in [battery_jump_start, engine_oil_leak, engine_sound_abnormal, exhaust_smoke_white, clutch_hard, gear_shifting_hard] if x])
             health_pct = ((total_checks - failed_checks) / total_checks) * 100
             st.metric("System Health", f"{health_pct:.0f}%", delta=f"-{failed_checks} Flags" if failed_checks > 0 else "Optimal")
        with kpi4:
             st.metric("Odometer Impact", "High" if odometer_reading > 100000 else "Normal")

        st.markdown("---")

        # --- Row 2: Trend Analysis (The "Honeywell" Line Chart style) ---
        st.subheader("📉 Predictive Degradation Analysis")
        st.caption("**What is this?** This graph simulates the future engine rating if the car continues to be driven without repairs. It shows the expected 'decay' of the rating over the next 50,000 km.")
        
        # logical simulation: Predict rating for current odometer + 10k, 20k...
        future_odo = [odometer_reading + i*10000 for i in range(6)]
        type_future = "Future Prediction"
        
        # We need to construct input DFs for these future points
        future_preds = []
        for odo in future_odo:
            # Copy base features
            feat_copy = input_features.copy()
            feat_copy['odometer_reading'] = odo
            # Make prediction
            X_temp = pd.DataFrame([feat_copy])[model_columns]
            pred_temp = model.predict(X_temp)[0]
            future_preds.append(max(0, pred_temp)) # Clamp at 0
            
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=future_odo, y=future_preds, mode='lines+markers', name='Projected Rating', line=dict(color='#ff4b4b', width=3)))
        fig_trend.update_layout(
            title="Engine Rating vs. Odometer Reading (Projected)",
            xaxis_title="Odometer (km)",
            yaxis_title="Predicted Rating (0-5)",
            height=350,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col_trend_info, _ = st.columns([3, 1])
        with col_trend_info:
            st.info(f"ℹ️ **Interpretation:** At the current usage, the engine rating is projected to drop to **{future_preds[-1]:.2f}** after another 50,000 km. Regular maintenance can flatten this curve.")

        st.markdown("---")
        
        # --- Row 3: Component Analysis (Pie + Radar) ---
        st.subheader("🔍 Diagnostics & Health Distribution")
        
        col_pie, col_radar = st.columns(2)
        
        with col_pie:
            st.markdown("##### System Status Distribution")
            st.caption("**Explanation:** Shows the proportion of systems passing inspection vs. those flagged with issues.")
            
            # Pie Data
            labels = ['Healthy Systems', 'Flagged Issues']
            values = [total_checks - failed_checks, failed_checks]
            colors = ['#00cc96', '#ef553b'] # Greenish, Reddish
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
            fig_pie.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_radar:
            st.markdown("##### Component Health Profile")
            st.caption("**Explanation:** A balanced polygon indicates a well-maintained vehicle. Dents (inward points) identify specific failing components.")
            
            # Reuse calculated r_values for Radar
            # categories = ['Battery', 'Oil', 'Sound', 'Exhaust', 'Clutch', 'Gears']
            # r_values calculated previously
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=categories,
                fill='toself',
                name='Health Score',
                line_color='#636efa'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # --- Row 4: Algorithm Explainability ---
        st.subheader("🧠 Model Decision Factors")
        st.caption("**Why this result?** The bar chart below ranks the features that most influenced the AI's decision for this specific prediction context.")
        
        if hasattr(model, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'Feature': model_columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False).head(10)
            
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
            st.plotly_chart(fig_fi, use_container_width=True)
            
        # --- Section 4: Historical Data Insights (New Request) ---
        if df_history is not None:
            st.markdown("---")
            st.header("📈 Historical Market Trends")
            st.caption("Insights derived from the historical dataset of car inspections. These graphs help understand the broader market context.")

            # 1. Registration Year Distribution
            st.subheader("1. Vehicle Age Distribution")
            st.caption("Shows the volume of inspections based on the car's **Registration Year**. This helps identify the most common vintage of cars being inspected.")
            
            if 'registrationYear' in df_history.columns:
                reg_counts = df_history['registrationYear'].value_counts().sort_index()
                fig_reg = px.bar(x=reg_counts.index, y=reg_counts.values, 
                                 labels={'x': 'Registration Year', 'y': 'Count of Inspections'})
                fig_reg.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_reg, use_container_width=True)

            # 2. Daily Inspections (Histogram/KDE)
            st.subheader("2. Daily Inspection Intensity")
            st.caption("Distribution of the number of inspections performed per day. Peaks indicate high-traffic operational days.")
            
            if 'inspection_date' in df_history.columns:
                daily_counts = df_history.groupby('inspection_date').size().reset_index(name='count')
                fig_hist = px.histogram(daily_counts, x="count", nbins=30, marginal="violin",
                                        labels={'count': 'Daily Inspections'})
                fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            # 3. Monthly Trend (Bar)
            st.subheader("3. Monthly Seasonality")
            st.caption("Inspection volume aggregated by month. Useful for planning resource allocation based on seasonal demand.")
            
            if 'inspection_mon' in df_history.columns:
                # Sort by month number
                mon_counts = df_history.groupby(['inspection_month_num', 'inspection_mon']).size().reset_index(name='count')
                mon_counts = mon_counts.sort_values('inspection_month_num')
                
                fig_mon = px.bar(mon_counts, x='inspection_mon', y='count',
                                 labels={'inspection_mon': 'Month', 'count': 'Total Inspections'},
                                 color='count', color_continuous_scale='Blues')
                fig_mon.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_mon, use_container_width=True)

            # 4. Time Series with Rolling Average
            st.subheader("4. Inspection Trends Over Time")
            st.caption("Daily inspection volume over the entire dataset timeline, with a **7-day rolling average** (orange line) to smooth out daily volatility.")
            
            if 'inspection_date' in df_history.columns:
                ts_data = df_history.groupby('inspection_date').size().reset_index(name='Original')
                ts_data['Rolling Mean'] = ts_data['Original'].rolling(window=7).mean()
                
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=ts_data['inspection_date'], y=ts_data['Original'], mode='lines', name='Daily Count', line=dict(color='lightblue', width=1)))
                fig_ts.add_trace(go.Scatter(x=ts_data['inspection_date'], y=ts_data['Rolling Mean'], mode='lines', name='7-Day Rolling Avg', line=dict(color='orange', width=2)))
                
                fig_ts.update_layout(xaxis_title="Date", yaxis_title="Number of Inspections", height=400, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown("---")
            st.header("🔬 Deep Dive: Statistical Analysis")
            st.caption("Advanced statistical distributions and correlations requested for deeper data understanding.")

            # 5. Odometer Boxplot
            st.subheader("5. Odometer Reading Distribution")
            st.caption("Visualizes the spread of odometer readings and identifies outliers (dots beyond the whiskers).")
            if 'odometer_reading' in df_history.columns:
                fig_odo_box = px.box(df_history, y="odometer_reading", points="outliers", 
                                     title="Odometer Reading Boxplot")
                fig_odo_box.update_layout(height=400)
                st.plotly_chart(fig_odo_box, use_container_width=True)

            # 6. Correlation Heatmap
            st.subheader("6. Feature Correlation Heatmap")
            st.caption("Shows how features are related. **1.0** = perfect positive correlation, **-1.0** = perfect negative correlation.")
            
            # Select columns for correlation roughly matching the user's image
            corr_cols = ['year', 'inspection_month_num', 'odometer_reading', 'rating_engineTransmission']
            # Filter to those that exist
            existing_corr_cols = [c for c in corr_cols if c in df_history.columns]
            
            if len(existing_corr_cols) > 1:
                corr_matrix = df_history[existing_corr_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                     color_continuous_scale='RdBu_r', origin='lower')
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)

            # 7, 8, 9. Rating Distributions by Time
            st.subheader("7. Rating Variability by Time Factors")
            st.caption("Boxplots showing how the Engine Rating varies across different time periods. Useful for spotting operational inconsistencies.")
            
            if 'rating_engineTransmission' in df_history.columns:
                tab_h, tab_m, tab_d = st.tabs(["By Hour", "By Month", "By Day of Week"])
                
                with tab_h:
                    if 'inspection_hour' in df_history.columns:
                        fig_box_h = px.box(df_history, x='inspection_hour', y='rating_engineTransmission',
                                           labels={'inspection_hour': 'Hour of Day', 'rating_engineTransmission': 'Engine Rating'})
                        st.plotly_chart(fig_box_h, use_container_width=True)
                
                with tab_m:
                    if 'inspection_mon' in df_history.columns:
                        # Order months correctly? 'inspection_month_num' helps sorting but 'inspection_mon' is name
                        # We can sort dataframe by num
                        df_sorted = df_history.sort_values('inspection_month_num')
                        fig_box_m = px.box(df_sorted, x='inspection_mon', y='rating_engineTransmission',
                                           labels={'inspection_mon': 'Month', 'rating_engineTransmission': 'Engine Rating'})
                        st.plotly_chart(fig_box_m, use_container_width=True)

                with tab_d:
                    if 'inspection_dow' in df_history.columns:
                        # Order might be random, but usually fine for exploration
                        fig_box_d = px.box(df_history, x='inspection_dow', y='rating_engineTransmission',
                                           labels={'inspection_dow': 'Day of Week', 'rating_engineTransmission': 'Engine Rating'})
                        st.plotly_chart(fig_box_d, use_container_width=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()

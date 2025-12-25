import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def render_analytics_tabs(df_clean):
    """
    Renders the 'Deep Dive Analytics' section with multiple tabs.
    
    Args:
        df_clean: The cleaned DataFrame to visualize.
    """
    st.subheader("Deep Dive Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlations", "Trends", "Diagnostics"])
    
    if df_clean is not None:
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
            # Heuristic: 'No' or 'Weak' or other values might indicate failure vs 'Yes' (if yes=good)
            
            snd_col = next((c for c in df_clean.columns if 'engineSound_value' in c), None)
            if snd_col:
                st.write(f"**Engine Sound Distribution**")
                st.bar_chart(df_clean[snd_col].value_counts())

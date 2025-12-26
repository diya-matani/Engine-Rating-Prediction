# 🚗 Engine Rating Prediction Dashboard

An advanced Machine Learning application designed to assess, predict, and analyze the condition of vehicle engines. This dashboard is built with **Streamlit** and features a premium **Honeywell-style** dark aesthetic, providing real-time insights for inspection engineers and market analysts.

## 🌟 Key Features

### 1. 🔍 Real-time Quality Monitor
- **Instant Engine Rating**: Predicts a quality score (0.0 - 5.0) based on vehicle parameters.
- **Health Diagnostics**: Visualizes system health (Battery, Oil, Sound, etc.) using interactive Radar and Donut charts.
- **Predictive Degradation**: AI-driven projection of how the engine rating will decay over the next +50,000 km.
- **Explainability**: "Decision Factors" chart highlights which features (e.g., Year, Odometer) influenced the AI's score the most.

### 2. 📈 Market Intelligence Hub
- **Linear Storytelling Layout**: A scrollable narrative uncovering market trends.
- **Historical Analytics**:
  - **Volume Trends**: Daily inspection patterns with 7-day moving averages.
  - **Seasonality**: Monthly breakdowns to spot peak activity times.
  - **Asset Vintage**: Distribution of vehicles by registration year.
  - **Correlation Heatmaps**: Deep-dive into relationships between Age, Mileage, and Ratings.

### 3. 📄 Inspection Records Database
- **Searchable Data**: A full history of past inspections.
- **Smart Filters**: "Show Weak Engines Only" (Rating ≤ 2) for focused triage.
- **Sorting**: Native column sorting by Rating, Date, etc.
- **Downloadable Reports**: Export filtered datasets to CSV.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YourUsername/Engine-Rating-Prediction.git
    cd Engine-Rating-Prediction
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

- **`app.py`**: The main Streamlit application containing the dashboard logic, UI layout, and visualization code.
- **`run_project.py`**: The training pipeline script. Reruns feature engineering, trains the LightGBM model, and saves artifacts.
- **`final_model_lgbm.pickle`**: The pre-trained LightGBM model used for predictions.
- **`model_columns.json`**: Metadata defining the exact feature columns expected by the model.
- **`Car_Features.csv` / `data.xlsx`**: Historical datasets used for the "Market Intelligence" and "Records" tabs.

## 🤖 Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly Express & Graph Objects
- **Machine Learning**: LightGBM, Scikit-Learn
- **Data Processing**: Pandas, NumPy

---

> **Note**: This project uses a custom "Honeywell" styling helper to ensure consistent dark-mode visuals across all charts.

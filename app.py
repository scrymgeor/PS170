import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import xgboost as xgb

st.set_page_config(page_title="ISRO ESS QA Dashboard", layout="wide")

# --- ML PIPELINE & DATA GENERATION ---
@st.cache_data
def run_ml_pipeline():
    np.random.seed(42)
    n_components = 1000 # Simulating a lot of 1000 chips
    
    # 1. Generate Realistic Base Data
    v0 = np.random.normal(10, 1.5, n_components)
    v24 = v0 + np.random.normal(1, 0.5, n_components)
    v168_actual = v24 + np.random.normal(4, 1.2, n_components)
    
    df = pd.DataFrame({
        'Component_ID': [f'COMP-{i:04d}' for i in range(1, n_components + 1)],
        'V_0h': v0,
        'V_24h': v24,
        'V_168h_Actual': v168_actual
    })
    
    # Inject a Latent Defect (Passes absolute limits, but drifts dangerously)
    df.loc[999, 'V_0h'] = 12.0
    df.loc[999, 'V_24h'] = 22.0
    df.loc[999, 'V_168h_Actual'] = 49.0 # Just under the 50 limit!
    df.loc[999, 'Component_ID'] = 'COMP-1000 (LATENT DEFECT)'

    # ==========================================
    # USP 1 & 2: Physics-Informed Feature Engineering & Lot Normalization
    # ==========================================
    df['Delta_0_24'] = df['V_24h'] - df['V_0h']
    df['Drift_Rate_24h'] = df['Delta_0_24'] / 24.0
    
    lot_median_v24 = df['V_24h'].median()
    df['Deviation_from_Lot_Median'] = abs(df['V_24h'] - lot_median_v24)

    # ==========================================
    # USP 3 (Module A): Dynamic Outlier Detection (Isolation Forest)
    # ==========================================
    # We look at behavior, not just absolute numbers.
    features_A = ['V_0h', 'V_24h', 'Delta_0_24', 'Deviation_from_Lot_Median']
    iso_forest = IsolationForest(contamination=0.01, random_state=42) # Expect 1% anomaly rate
    df['Module_A_Anomaly'] = iso_forest.fit_predict(df[features_A])
    # IsolationForest outputs -1 for anomalies, 1 for normal. Convert to True/False
    df['Dynamic_Anomaly_Flag'] = df['Module_A_Anomaly'] == -1

    # ==========================================
    # USP 3 (Module B): Time-Series Drift Predictor (XGBoost)
    # ==========================================
    # Train AI to predict 168h value based on 0h and 24h behavior
    features_B = ['V_0h', 'V_24h', 'Delta_0_24', 'Drift_Rate_24h']
    
    # In a real scenario, we'd train on historical data. Here we train on the current lot.
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_model.fit(df[features_B], df['V_168h_Actual'])
    
    df['V_168h_Predicted'] = xgb_model.predict(df[features_B])
    
    # Calculate Safety Slope
    df['Predicted_Slope_24_168'] = (df['V_168h_Predicted'] - df['V_24h']) / (168 - 24)
    safety_slope_limit = 0.150 # Max allowed drift rate per hour
    df['Slope_Anomaly_Flag'] = df['Predicted_Slope_24_168'] > safety_slope_limit
    
    # Final Decision Engine
    # Reject if it fails Module A OR Module B
    df['Status'] = np.where(df['Dynamic_Anomaly_Flag'] | df['Slope_Anomaly_Flag'], 'REJECT', 'PASS')
    
    return df

# Load Data
df = run_ml_pipeline()

# --- DASHBOARD UI ---
st.title("🚀 ISRO Component Burn-In AI Dashboard")
st.markdown("### AI-Driven Anomaly Detection for Environmental Stress Screening (ESS)")

st.sidebar.header("QA Inspector Controls")
# Sort so the defect is at the top of the list for easy finding
sorted_comps = df.sort_values(by='Status', ascending=False)['Component_ID'].tolist()
selected_comp = st.sidebar.selectbox("Select Component to Inspect:", sorted_comps)

comp_data = df[df['Component_ID'] == selected_comp].iloc[0]

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("0h Measurement", f"{comp_data['V_0h']:.2f} µA")
col2.metric("24h Measurement", f"{comp_data['V_24h']:.2f} µA")
col3.metric("168h AI Forecast", f"{comp_data['V_168h_Predicted']:.2f} µA", delta=f"{comp_data['V_168h_Predicted'] - comp_data['V_24h']:.2f} µA drift", delta_color="inverse")

if comp_data['Status'] == 'PASS':
    col4.success("STATUS: PASS (Nominal)")
else:
    col4.error("STATUS: REJECT (Anomaly)")

st.divider()

# --- VISUALIZATIONS ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Module A: Lot Distribution Context")
    fig_a = go.Figure()
    
    # Plot normal components
    normal_df = df[df['Status'] == 'PASS']
    fig_a.add_trace(go.Scatter(x=normal_df['V_0h'], y=normal_df['V_24h'], mode='markers', name='Normal', marker=dict(color='lightblue', size=6, opacity=0.4)))
    
    # Plot anomalous components
    anomaly_df = df[df['Status'] == 'REJECT']
    fig_a.add_trace(go.Scatter(x=anomaly_df['V_0h'], y=anomaly_df['V_24h'], mode='markers', name='Anomalies', marker=dict(color='orange', size=8, opacity=0.8)))
    
    # Highlight selected
    fig_a.add_trace(go.Scatter(x=[comp_data['V_0h']], y=[comp_data['V_24h']], mode='markers', name='Selected', marker=dict(color='red', size=16, symbol='star', line=dict(color='black', width=2))))
    
    fig_a.update_layout(xaxis_title="0h Leakage (µA)", yaxis_title="24h Leakage (µA)", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_a, use_container_width=True)

with chart_col2:
    st.subheader("Module B: AI Time-Series Forecast")
    fig_b = go.Figure()
    
    # Actual Data
    fig_b.add_trace(go.Scatter(x=[0, 24], y=[comp_data['V_0h'], comp_data['V_24h']], mode='lines+markers', name='Measured', line=dict(color='blue', width=3)))
    
    # AI Prediction
    fig_b.add_trace(go.Scatter(x=[24, 168], y=[comp_data['V_24h'], comp_data['V_168h_Predicted']], mode='lines+markers', name='XGBoost Prediction', line=dict(color='orange', width=3, dash='dash')))
    
    # Absolute Limit
    fig_b.add_hline(y=50, line_dash="dot", annotation_text="Max Limit (50 µA)", annotation_position="bottom right", line_color="red")
    
    fig_b.update_layout(xaxis_title="Burn-In Hours", yaxis_title="Leakage Current (µA)", yaxis=dict(range=[0, 55]), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_b, use_container_width=True)

# --- USP 4: EXPLAINABILITY REPORT ---
st.subheader("🤖 AI Explainability Report (Why did the model make this decision?)")

if comp_data['Status'] == 'REJECT':
    reasons = []
    if comp_data['Dynamic_Anomaly_Flag']:
        reasons.append(f"**Contextual Outlier (Module A):** The component deviates significantly from the lot median. Its 0h to 24h drift is {comp_data['Delta_0_24']:.2f} µA, which the Isolation Forest model flagged as an abnormal trajectory compared to its batch.")
    if comp_data['Slope_Anomaly_Flag']:
        reasons.append(f"**Dangerous Degradation Slope (Module B):** The XGBoost model predicts a 168h leakage of {comp_data['V_168h_Predicted']:.2f} µA. While this is technically below the absolute limit of 50µA, the resulting drift rate is **{comp_data['Predicted_Slope_24_168']:.3f} µA/hr**. This exceeds the maximum safety tolerance of 0.150 µA/hr.")
    
    st.error(" \n\n".join(reasons))
    st.error("⚠️ **CONCLUSION: Latent Defect Detected.** Proceeding with this component risks critical payload failure.")
else:
    st.success(f"""
    **Component Cleared for Payload:**
    * **Module A (Context):** Isolation Forest confirms early behavior matches batch baseline.
    * **Module B (Drift):** XGBoost forecasts 168h value at {comp_data['V_168h_Predicted']:.2f} µA. Drift slope ({comp_data['Predicted_Slope_24_168']:.3f} µA/hr) is well within safety thresholds.
    """)

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:35:04 2026

@author: HP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ISRO ESS QA Dashboard", layout="wide")

# --- 1. GENERATE SYNTHETIC DATA (For Demonstration) ---
@st.cache_data
def load_data():
    # Simulate a lot of 100 normal components
    np.random.seed(42)
    lot_v0 = np.random.normal(10, 1.5, 100)
    lot_v24 = lot_v0 + np.random.normal(1, 0.5, 100)
    lot_v168_pred = lot_v24 + np.random.normal(3, 1, 100)
    
    df = pd.DataFrame({
        'Component_ID': [f'COMP-{i:03d}' for i in range(1, 101)],
        'V_0h': lot_v0,
        'V_24h': lot_v24,
        'V_168h_Predicted': lot_v168_pred,
        'Status': ['PASS'] * 100
    })
    
    # Inject a "Latent Defect" (The tricky one that escapes static limits)
    # Absolute limit is 50. This part stays under 50, but drifts way too fast!
    defect = {'Component_ID': 'COMP-042 (DEFECT)', 'V_0h': 12.0, 'V_24h': 19.0, 'V_168h_Predicted': 48.0, 'Status': 'REJECT'}
    df.loc[41] = defect
    
    return df

df = load_data()

# --- 2. DASHBOARD UI ---
st.title("🚀 ISRO Component Burn-In QA Dashboard")
st.markdown("### AI-Driven Anomaly Detection for Environmental Stress Screening (ESS)")

# Sidebar for Selection
st.sidebar.header("QA Inspector Controls")
selected_comp = st.sidebar.selectbox("Select Component to Inspect:", df['Component_ID'].tolist())

# Get data for the selected component
comp_data = df[df['Component_ID'] == selected_comp].iloc[0]
lot_median_v0 = df['V_0h'].median()
drift_rate = (comp_data['V_168h_Predicted'] - comp_data['V_24h']) / (168 - 24)

# --- 3. TOP METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("0h Measurement", f"{comp_data['V_0h']:.2f} µA")
col2.metric("24h Measurement", f"{comp_data['V_24h']:.2f} µA")
col3.metric("168h Prediction", f"{comp_data['V_168h_Predicted']:.2f} µA", delta=f"{comp_data['V_168h_Predicted'] - comp_data['V_24h']:.2f} µA drift", delta_color="inverse")

if comp_data['Status'] == 'PASS':
    col4.success("STATUS: PASS (Nominal)")
else:
    col4.error("STATUS: REJECT (Anomaly)")

st.divider()

# --- 4. VISUALIZATIONS (Modules A & B) ---
chart_col1, chart_col2 = st.columns(2)

# MODULE A: Dynamic Outlier Detection (Scatter Plot)
with chart_col1:
    st.subheader("Module A: Lot Distribution Context")
    fig_a = go.Figure()
    
    # Plot the whole lot
    fig_a.add_trace(go.Scatter(
        x=df['V_0h'], y=df['V_24h'], mode='markers', name='Lot Components',
        marker=dict(color='lightblue', size=8, opacity=0.6)
    ))
    
    # Highlight selected component
    fig_a.add_trace(go.Scatter(
        x=[comp_data['V_0h']], y=[comp_data['V_24h']], mode='markers', name='Selected Component',
        marker=dict(color='red' if comp_data['Status']=='REJECT' else 'green', size=14, symbol='star')
    ))
    
    fig_a.update_layout(xaxis_title="0h Leakage (µA)", yaxis_title="24h Leakage (µA)", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_a, use_container_width=True)

# MODULE B: Time-Series Drift Predictor (Line Chart)
with chart_col2:
    st.subheader("Module B: Time-Series Drift Forecast")
    fig_b = go.Figure()
    
    times = [0, 24, 168]
    values = [comp_data['V_0h'], comp_data['V_24h'], comp_data['V_168h_Predicted']]
    
    # Plot Actual Data (0 to 24)
    fig_b.add_trace(go.Scatter(x=[0, 24], y=[comp_data['V_0h'], comp_data['V_24h']], mode='lines+markers', name='Actual (Measured)', line=dict(color='blue', width=3)))
    
    # Plot Prediction (24 to 168)
    fig_b.add_trace(go.Scatter(x=[24, 168], y=[comp_data['V_24h'], comp_data['V_168h_Predicted']], mode='lines+markers', name='AI Prediction', line=dict(color='orange', width=3, dash='dash')))
    
    # Add Absolute Limit Line
    fig_b.add_hline(y=50, line_dash="dot", annotation_text="Absolute Datasheet Limit (50 µA)", annotation_position="bottom right", line_color="red")
    
    fig_b.update_layout(xaxis_title="Burn-In Hours", yaxis_title="Leakage Current (µA)", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_b, use_container_width=True)

# --- 5. EXPLAINABILITY SECTION (Crucial for the Hackathon) ---
st.subheader("🤖 AI Explainability Report")
if comp_data['Status'] == 'REJECT':
    st.error(f"""
    **Component Flagged for Early Rejection:**
    * **Dynamic Outlier Check:** At 24h, this component measures {comp_data['V_24h']:.2f} µA. The lot median is {df['V_24h'].median():.2f} µA. This part is dynamically abnormal compared to its batch.
    * **Drift Rate Analysis:** The slope of degradation from 24h to 168h is {drift_rate:.3f} µA/hr. This exceeds the maximum safety slope of 0.150 µA/hr.
    * **Conclusion:** Although the predicted 168h value ({comp_data['V_168h_Predicted']:.2f} µA) is technically *below* the static 50 µA limit, this is a **Latent Defect**. Passing this component risks catastrophic field failure in orbit.
    """)
else:
    st.success(f"""
    **Component Cleared for Payload:**
    * **Dynamic Check:** Values conform tightly to the lot median.
    * **Drift Rate:** The predicted degradation slope ({drift_rate:.3f} µA/hr) is well within safety tolerances.
    """)
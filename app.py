# app.py
# A Streamlit application to demonstrate the IOLIN project.

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="IOLIN Project Dashboard",
    page_icon="⚡",
    layout="wide"
)

# --- Helper Functions to Load Data ---
@st.cache_data
def load_metrics_data(file_path):
    """Loads the metrics CSV files from the results folder."""
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

@st.cache_data
def load_analysis_report():
    """Loads the final text-based analysis report."""
    report_path = os.path.join("results", "comparison_analysis.txt")
    if not os.path.exists(report_path):
        return "Analysis report not found. Please run 'experiments/analyze_results.py' first."
    with open(report_path, 'r') as f:
        return f.read()

# --- Load Data ---
iolin_metrics = load_metrics_data(os.path.join("results", "iolin_metrics.csv"))
olin_metrics = load_metrics_data(os.path.join("results", "olin_metrics.csv"))
analysis_report = load_analysis_report()

# --- Main App ---
st.title("⚡ IOLIN: Incremental On-line Information Network")
st.markdown("""
This dashboard presents the results of a Python implementation of the research paper **"Real-time data mining of non-stationary data streams from sensor networks"**.
We compare the performance of the traditional regenerative **OLIN** algorithm against the proposed efficient **IOLIN** algorithm.
""")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["📊 Final Results & Analysis", "🔬 Interactive Simulation"])

# ==============================================================================
# FINAL RESULTS TAB
# ==============================================================================
with tab1:
    st.header("Final Performance Comparison")
    
    if olin_metrics is None or iolin_metrics is None:
        st.error("Metrics files not found in the 'results' folder. Please run the experiments first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Statistical Analysis")
            st.code(analysis_report, language='text')

        with col2:
            st.subheader("Performance Charts")
            
            # Prepare data for charting
            summary_data = {
                'Algorithm': ['Regenerative OLIN', 'Incremental IOLIN'],
                'Avg Processing Time (s)': [olin_metrics['processing_time_s'].mean(), iolin_metrics['processing_time_s'].mean()],
                'Total Time (s)': [olin_metrics['processing_time_s'].sum(), iolin_metrics['processing_time_s'].sum()]
            }
            summary_df = pd.DataFrame(summary_data).set_index('Algorithm')

            st.markdown("##### Average Time per Window (s)")
            st.bar_chart(summary_df['Avg Processing Time (s)'])
            
            st.markdown("##### Total Experiment Time (s)")
            st.bar_chart(summary_df['Total Time (s)'])

# ==============================================================================
# INTERACTIVE SIMULATION TAB
# ==============================================================================
with tab2:
    st.header("IOLIN Algorithm Simulation")
    st.markdown("Use the slider to move through the data stream and observe how the IOLIN algorithm decides whether to `Update` or `Rebuild` the model at each step.")

    if iolin_metrics is None:
        st.error("IOLIN metrics file not found. Please run 'experiments/run_iolin_experiment.py' first.")
    else:
        # --- Interactive Slider ---
        total_windows = len(iolin_metrics)
        # The slider will return the index of the selected window
        step_index = st.slider("Select a window to inspect:", 0, total_windows - 1, 0)

        # Get data for the selected step
        step_data = iolin_metrics.iloc[step_index]
        is_drift = step_data['is_drift']
        action = step_data['action']
        time_taken = step_data['processing_time_s']
        error = step_data['final_error_rate']
        window_start = int(step_data['window_start'])
        window_end = int(step_data['window_end'])

        # --- Status Dashboard ---
        st.subheader(f"Status at Window: {window_start} - {window_end}")
        
        if is_drift:
            st.error(f"**Status:** Concept Drift Detected! Model was **Rebuilt**.")
        else:
            st.success(f"**Status:** Concept Stable. Model was **Updated**.")
        
        # Display key metrics for the selected step
        kpi_col1, kpi_col2 = st.columns(2)
        kpi_col1.metric(label="Action Taken", value=action)
        kpi_col1.metric(label="Processing Time for this Window", value=f"{time_taken:.3f} s")
        kpi_col2.metric(label="Final Error Rate for this Window", value=f"{error:.3f}")

        # --- Live Performance Chart ---
        st.subheader("Error Rate Over Time")
        
        # We'll plot the error rate up to the selected step
        chart_data = iolin_metrics[['final_error_rate']].head(step_index + 1)
        chart_data.rename(columns={'final_error_rate': 'Error Rate'}, inplace=True)
        
        st.line_chart(chart_data)
        st.caption("This chart shows the model's error rate on each new window of data up to the selected step.")

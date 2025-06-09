import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("IOLIN: Incremental On-line Information Network")
st.markdown("An interactive dashboard to demonstrate and evaluate the IOLIN algorithm against the regenerative OLIN.")

# --- Tab Layout ---
tab1, tab2 = st.tabs(["Interactive Simulation", "Final Results"])

with tab1:
    st.header("Real-time Algorithm Simulation")
    st.markdown("This tab will demonstrate the IOLIN algorithm processing a data stream window by window.")

    # Placeholder for the data stream slider
    total_windows = 100 # Example value
    current_step = st.slider("Move through the data stream:", 1, total_windows, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drift Detection Status")
        # Example of dynamic status display
        if current_step % 15 == 0: # Simulate a drift
            st.error(f"🚨 Step {current_step}: Concept Drift Detected! Rebuilding model from scratch.")
        else:
            st.success(f"✅ Step {current_step}: Concept Stable. Updating existing model.")

        st.metric(label="Training Error (E_tr)", value=f"{np.random.uniform(0.1, 0.15):.3f}")
        st.metric(label="Validation Error (E_val)", value=f"{np.random.uniform(0.12, 0.18):.3f}")


    with col2:
        st.subheader("Live Performance Chart")
        # Create some dummy data for the chart that changes with the slider
        chart_data = pd.DataFrame(
            np.random.randn(current_step, 2) / 10 + 0.15,
            columns=['Error Rate', 'Drift Threshold']
        )
        st.line_chart(chart_data)

with tab2:
    st.header("Final Evaluation Results")
    st.markdown("A side-by-side comparison of the final performance metrics for OLIN and IOLIN after processing the entire dataset.")

    # Placeholder for final results
    st.subheader("Performance Metrics")
    results_data = {
        'Algorithm': ['Regenerative OLIN', 'Incremental IOLIN'],
        'Average Run Time (s)': [27.75, 15.62],
        'Average Error Rate': [0.178, 0.178],
        'Classification Rate (records/sec)': [334, 532]
    }
    results_df = pd.DataFrame(results_data)
    st.table(results_df.set_index('Algorithm'))

    st.subheader("Statistical Analysis")
    st.info("Run Time Comparison: The reduction in run time with IOLIN was found to be statistically significant (p < 0.05).")
    st.info("Accuracy Comparison: The difference in error rates between the two algorithms was not statistically significant.")


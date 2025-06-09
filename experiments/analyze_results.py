# experiments/analyze_results.py
# This script will load the metrics from both the OLIN and IOLIN experiments.
# It will perform the paired samples T-test to compare them,
# print the analysis, and save a summary to /results/comparison_analysis.txt.

import pandas as pd
import os
from scipy.stats import ttest_rel

def analyze():
    """Loads and analyzes the results from the OLIN and IOLIN experiments."""
    print("--- Starting Results Analysis ---")

    try:
        olin_path = os.path.join("results", "olin_metrics.csv")
        iolin_path = os.path.join("results", "iolin_metrics.csv")
        olin_df = pd.read_csv(olin_path)
        iolin_df = pd.read_csv(iolin_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find result files. {e}")
        print("Please run both 'run_olin_experiment.py' and 'run_iolin_experiment.py' first.")
        return

    # Ensure dataframes have the same length for comparison
    min_len = min(len(olin_df), len(iolin_df))
    olin_df = olin_df.head(min_len)
    iolin_df = iolin_df.head(min_len)

    # --- Performance Summary ---
    summary = {
        'OLIN': {
            'Avg Error Rate': olin_df['error_rate'].mean(),
            'Avg Processing Time (s)': olin_df['processing_time_s'].mean(),
            'Total Time (s)': olin_df['processing_time_s'].sum()
        },
        'IOLIN': {
            'Avg Error Rate': iolin_df['final_error_rate'].mean(),
            'Avg Processing Time (s)': iolin_df['processing_time_s'].mean(),
            'Total Time (s)': iolin_df['processing_time_s'].sum()
        }
    }
    summary_df = pd.DataFrame(summary)

    # --- Statistical T-Test (Paired) ---
    # Compare processing times
    time_ttest = ttest_rel(olin_df['processing_time_s'], iolin_df['processing_time_s'])
    
    # Compare error rates
    error_ttest = ttest_rel(olin_df['error_rate'], iolin_df['final_error_rate'])

    # --- Generate Report ---
    report = []
    report.append("=========================================")
    report.append("   IOLIN vs. OLIN Performance Analysis   ")
    report.append("=========================================\n")
    report.append("--- Overall Performance Metrics ---\n")
    report.append(summary_df.to_string())
    report.append("\n\n--- Statistical Significance (Paired T-Test) ---\n")
    report.append(f"Comparing Processing Times:")
    report.append(f"  T-statistic: {time_ttest.statistic:.4f}")
    report.append(f"  P-value: {time_ttest.pvalue:.4f}")
    if time_ttest.pvalue < 0.05:
        report.append("  Conclusion: The difference in processing time is statistically significant.\n")
    else:
        report.append("  Conclusion: The difference in processing time is NOT statistically significant.\n")

    report.append(f"Comparing Error Rates:")
    report.append(f"  T-statistic: {error_ttest.statistic:.4f}")
    report.append(f"  P-value: {error_ttest.pvalue:.4f}")
    if error_ttest.pvalue < 0.05:
        report.append("  Conclusion: The difference in error rate is statistically significant.\n")
    else:
        report.append("  Conclusion: The difference in error rate is NOT statistically significant.\n")
    
    report_str = "\n".join(report)
    print(report_str)

    # Save report to file
    report_path = os.path.join("results", "comparison_analysis.txt")
    with open(report_path, 'w') as f:
        f.write(report_str)
    print(f"\nAnalysis report saved to {report_path}")


if __name__ == '__main__':
    analyze()

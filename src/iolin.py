# src/iolin.py
# This module will implement the main IOLIN algorithm.
# Key components:
# - Drift detection logic (using Eq. 4 and 5 from the paper).
# - `Update_Current_Network` function containing the three update strategies:
#   1. Check_Split_Validity
#   2. Replace_Last_Layer
#   3. New_Split_Process

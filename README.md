# IOLIN: Incremental On-line Information Network

This project is an implementation of the research paper "Real-time data mining of non-stationary data streams from sensor networks" by Cohen et al.

## Project Structure
- **/data**: Contains raw and processed datasets.
- **/src**: Core source code for IN, OLIN, and IOLIN algorithms.
- **/notebooks**: Jupyter notebooks for data exploration and result visualization.
- **/experiments**: Scripts to run the full experiments and analysis.
- **/results**: Output files from experiments (metrics, logs, etc.).
- **app.py**: A Streamlit application to demonstrate the project.

## Setup
1. Create a virtual environment:
   `python -m venv venv`
   `source venv/bin/activate`  # On Windows use `venv\Scripts\activate`
2. Install dependencies:
   `pip install -r requirements.txt`

## How to Run
1. Run the experiments:
   `python experiments/run_olin_experiment.py`
   `python experiments/run_iolin_experiment.py`
2. Analyze the results:
   `python experiments/analyze_results.py`
3. Launch the Streamlit GUI:
   `streamlit run app.py`

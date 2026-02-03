# Source code (`src/`)

Core implementation of the FlowSense-IDS pipeline.

## Modules

- `signal.py`  
  Reads a Wireshark-exported CSV (column names mapped via `columns.json`), builds 1-second time-window traffic signals, and saves the result as `network_signals.csv`.

- `ML - model.py`  
  Loads `network_signals.csv`, performs feature engineering, runs a hybrid anomaly detection pipeline (unsupervised models + simple expert rules + majority voting), and saves `ml_results.csv` along with generated plots.

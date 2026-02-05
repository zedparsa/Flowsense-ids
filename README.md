<!-- ===== Project Header ===== -->

<div align="center">

# FlowSense-IDS

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Press+Start+2P&size=14&pause=1000000&color=60A3BC&center=true&vCenter=true&width=1100&lines=Hybrid+network+traffic+anomaly+detection+from+time-window+signals.&repeat=false&cursor=false" />
</p>



<!-- Quick Links -->
<p align="center">
  <a href="#overview"><img src="https://img.shields.io/badge/Overview-111827?style=for-the-badge" /></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick%20start-111827?style=for-the-badge" /></a>
  <a href="#signal-extraction"><img src="https://img.shields.io/badge/Signal%20extraction-111827?style=for-the-badge" /></a>
  <a href="#machine-learning"><img src="https://img.shields.io/badge/Machine%20learning-111827?style=for-the-badge" /></a>
  <a href="#outputs"><img src="https://img.shields.io/badge/Outputs-111827?style=for-the-badge" /></a>
  <a href="#limitations"><img src="https://img.shields.io/badge/Limitations-111827?style=for-the-badge" /></a>
</p>

</div>

---
## Overview

**FlowSense-IDS** is a small, course-friendly pipeline that converts packet logs into discrete-time signals (fixed windows) and flags suspicious windows using multiple unsupervised methods plus simple expert rules. The focus is clarity, explainability, and presentation-ready outputs rather than production deployment.

### Highlights

| Item | Description |
|---|---|
| Goal | Flag unusual traffic windows using signals + hybrid detection |
| Input | Wireshark-exported CSV mapped with columns.json |
| Output | ml_results.csv + plots (timeline, model comparison, correlations) |
| Approach | Unsupervised models + expert rules + majority voting |

### Pipeline

```mermaid
    flowchart LR
      A[CSV] --> B[Windowing]
      B --> C[Signal features]
      C --> D[Feature engineering]
      D --> E[Unsupervised models]
      D --> F[Expert rules]
      E --> G[Majority voting]
      F --> G
      G --> H[ml_results.csv + plots]
```
**Flow**:  
> Wireshark CSV → windowing (1 second) → signal features → feature engineering → unsupervised models + expert rules → majority voting → results (CSV + plots)

**Why signals + hybrid approach**:  
> Aggregating traffic into fixed windows makes intensity, diversity, and timing changes visible as simple signal statistics; combining multiple detectors improves robustness and produces more defensible outputs (votes, scores, and rule-based labels)

### Signals used

| Signal | What it captures | Why it helps |
|---|---|---|
| Packet count | Traffic intensity / bursts | Detect spikes and surges |
| Traffic volume | Bytes per window | Separates many-small vs few-large packets |
| Unique source IPs | Source diversity | Multi-source patterns |
| Source IP entropy | Concentration vs spread | Identifies skewed vs diverse sources |
| Time-interval variance | Timing irregularity | Bursty vs stable behavior |

### ML models used

| Model | Type | Purpose |
|---|---|---|
| Isolation Forest | Unsupervised | Detects outliers based on isolation |
| One-Class SVM | Unsupervised | Learns a boundary of normal behavior and flags deviations |
| KMeans | Clustering | Distance-based anomaly detection |
| Expert rules | Rule-based | Threshold-based labeling for interpretability |
| Ensemble voting | Fusion | Combines all detectors (majority vote) |

---

## Quick start

### 1) Install

**Ubuntu/Linux**:
```bash
pip3 install -r requirements.txt
```
**Windows/macOS**:
```bash
pip install -r requirements.txt
```

### 2) Build signals
```bash
python src/signal.py
```

### 3) Run detection
```bash
python src/ml_model.py
```

---

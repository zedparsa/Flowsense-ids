<!-- ===== Project Header ===== -->

<div align="center">

# FlowSense-IDS

Hybrid network traffic anomaly detection from time-window signals.

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-informational)
![Status](https://img.shields.io/badge/Status-Student%20Project-2ea44f)

<!-- Quick Links -->
<a href="#quick-start">Quick start</a> •
<a href="#pipeline">Pipeline</a> •
<a href="#signals-used">Signals</a> •
<a href="#outputs">Outputs</a> •
<a href="#limitations">Limitations</a>

</div>

---

## Overview

FlowSense-IDS is a small, course-friendly pipeline that converts packet logs into discrete-time signals (fixed windows) and flags suspicious windows using multiple unsupervised methods plus simple expert rules. The focus is on clarity, explainability, and presentation-ready outputs rather than production deployment.

## Highlights

| Item | Description |
|---|---|
| Goal | Flag unusual traffic windows using signals + hybrid detection |
| Input | Wireshark-exported CSV mapped with `columns.json` |
| Output | `ml_results.csv` + plots (timeline, model comparison, correlations) |
| Approach | Unsupervised models + expert rules + majority voting |

---

## Pipeline

```mermaid
flowchart LR
  A[Wireshark CSV] --> B[Windowing 1s]
  B --> C[Signal features]
  C --> D[Feature engineering]
  D --> E[Unsupervised models]
  D --> F[Expert rules]
  E --> G[Majority voting]
  F --> G
  G --> H[ml_results.csv + plots]
```

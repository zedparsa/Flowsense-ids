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

## Signal extraction
> (signal.py) — block by block

#### 1) Config: load mapping 
This part solves a practical issue: Wireshark CSV exports don’t always have the same column names (they vary by version, language, and export settings). Instead of hard-coding fragile column names inside the program, you load a JSON “translation layer” that maps whatever the CSV contains to your own stable schema.


```python
    MAPPING_FILE = "columns.json"
    CSV_FILE = "test.wireshark.csv"
    OUT_FILE = "network_signals.csv"

    try:
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            column_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file '{MAPPING_FILE}' not found!")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: '{MAPPING_FILE}' is not valid JSON: {e}")
        raise
```
Conceptually, this separates “format compatibility” from “signal logic.” If the CSV headers change later, you update only `columns.json`, not the feature-engineering code. The explicit error handling is a correctness guarantee: if the mapping file is missing or invalid, the pipeline stops early rather than producing a clean-looking but wrong dataset.  

Key implementation notes:
> - The `try/except` ensures “fail fast”: if the mapping file is missing or malformed, the pipeline stops early instead of producing a wrong dataset silently.  
> - `encoding="utf-8"` avoids issues if the JSON file ever contains non-ASCII characters

---

#### 2) Read CSV
This block is the ingestion gate that converts the exported file into a structured table (a DataFrame). Everything downstream assumes the table exists and contains rows (packets) and columns (fields), so this stage validates that assumption first.


```python
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: File '{CSV_FILE}' not found!")
        raise
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty!")
        raise
```
Conceptually, it’s about making failure explicit and early. If the file is missing or empty, stopping is the correct behavior because any computed signals would be meaningless yet still look “valid” (for example, a file full of zeros). Keeping this stage simple also makes the pipeline easier to explain: “load data once, then process it.”

Key implementation notes:
> - `pd.read_csv` is the single source of truth loader here; everything downstream depends on having a valid table.
> - You handle the two most common failure cases explicitly: missing file and empty file.

---

#### 3) Rename columns to a standard schema
This part creates a stable internal contract for your data. After renaming, the rest of your code can assume the same column names every time (`time`, `source`, `length`, etc.), no matter what the original CSV header labels were.


```python
    df = df.rename(columns=column_map)

    REQUIRED = {"no", "time", "source", "length"}
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns after renaming: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Check your '{MAPPING_FILE}'."
        )

    df = df[list(REQUIRED)]
```
Conceptually, you are building a clean boundary between the messy outside world (CSV exports) and the clean inside world (your analysis pipeline). The required-column check is critical: it prevents confusing downstream errors by validating the schema upfront and giving a clear, actionable message if something is wrong.

Key implementation notes:
> - The “missing required columns” error is presentation-friendly because it prints both what is missing and what is available
> - Using a set for `REQUIRED` is fine for validation, but `df = df[list(REQUIRED)]` can reorder columns unpredictably; for stable ordering, use a list like `["no","time","source","length"]`.

---

#### 4) Cleaning
Cleaning turns “CSV text” into reliable numeric and categorical values that you can safely aggregate. Real CSV exports often contain missing values, non-numeric strings, or inconsistent formatting, and those issues can break computations or distort results.


```python
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0)
    df["source"] = df["source"].fillna("UNKNOWN")

    df = df.dropna(subset=["time"])

    if len(df) == 0:
        raise ValueError("No valid rows after cleaning (time is all invalid).")
```
Conceptually, you are making the dataset machine-usable. Time must be numeric because it defines your entire timeline; length must be numeric because you sum bytes; source should not be null because grouping and diversity metrics depend on consistent categories. Dropping rows with invalid time is principled: if a packet cannot be placed on the timeline, it cannot contribute to time-window signals.


Key implementation notes:
> - `errors="coerce"` is ideal for messy CSV exports: bad values become NaN, then you drop only rows where time is invalid.
> - Lengths that fail conversion become 0, which prevents crashes and keeps aggregation consistent.
> - Filling `source` with `"UNKNOWN"` preserves row count and makes “missing source” explicit instead of silently dropping packets.

---

#### 5) Build shared time windows
This is the key transformation that turns packet logs into a time-series dataset. Packets arrive at irregular continuous timestamps, but signal extraction needs discrete, comparable bins so every second (or window) becomes one observation.

```python
    times = df["time"]
    n = pd.Series(np.floor(times).astype(int), index=times.index)

    min_n = int(n.min())
    max_n = int(n.max())
    global_index = range(min_n, max_n + 1)

    source_ip = df["source"]
    size = df["length"]
```

Conceptually, you are defining the sampling grid for your signals. By flooring time to integer seconds, every packet is assigned to a 1-second window, turning “events in time” into “counts and statistics per window.” Building a continuous global index from the first to last window matters because gaps become meaningful zeros (no traffic) rather than missing timestamps that can confuse plots and models.

Key implementation notes:
> - `np.floor(time)` defines the 1-second window ID; it is simple, explainable, and consistent.
> - `global_index` guarantees a continuous timeline; gaps become explicit zeros instead of missing rows, which is crucial for plotting and ML.
> - Keeping `source_ip` and `size` as separate Series makes later groupby operations clearer.

---

#### 6) Core helper: basic_signal
This helper enforces a strong invariant: every signal must align to the same global time axis and provide a value for every time window. Without that, features can end up with different lengths or missing windows, making merges error-prone and creating subtle bugs.

```python
    def basic_signal(compute_function):
        signal = compute_function(n)
        return signal.reindex(global_index, fill_value=0)
```

Conceptually, this is output standardization. Each signal function focuses only on its definition (count, sum, entropy, timing variability), and the helper handles alignment and missing-window behavior. Filling missing windows with zero is also a semantic choice: “no packets happened” becomes a real value, not “unknown.”

Key implementation notes:
> - `reindex(..., fill_value=0)` is the main reason your signals are comparable and stackable in a single output table.
> - This also makes downstream ML simpler because it never has to deal with missing time windows.

---

#### 7) Signal: packet_count
This feature measures traffic intensity: how many packets arrive in each second. It is simple, intuitive, and extremely useful for detecting sudden spikes, bursts, or drops in activity.


```python
    def packet_count():
        return basic_signal(lambda n: n.value_counts().sort_index())
```
Conceptually, packet_count is “event frequency” in time-series terms. Many network anomalies show up as changes in frequency, such as abrupt surges (flood-like behavior) or sudden silence (outage). It also provides context for other signals, like distinguishing “many small packets” from “few large packets” when combined with volume.

Key implementation notes:
> - `value_counts()` on the window index is an efficient way to count packets per second.
> - `sort_index()` keeps windows in chronological order before reindexing.

---

#### 8) Signal: traffic_volume
This feature measures total bytes per second. It answers a different question than packet_count: not “how many packets,” but “how much data moved.”

```python
    def traffic_volume():
        return basic_signal(lambda n: size.groupby(n).sum())
```

Conceptually, traffic_volume is a coarse throughput measure. Two windows can have the same packet_count but very different byte totals, which often indicates very different behavior (control chatter versus bulk transfer). When you present it, it pairs naturally with packet_count to explain traffic shape and load.

Key implementation notes:
> - `size.groupby(n).sum()` directly implements “bytes per window”.
> - Together with packet_count, it enables derived features like average packet size later.

---

#### 9) Signal: source_entropy
This feature quantifies how concentrated or diverse the source distribution is within each time window using Shannon entropy, \(H = -\sum p_i \log_2(p_i)\). It goes beyond simply counting unique sources by considering whether traffic is dominated by one source or spread evenly among many.


```python
    def source_entropy():
        def calculate_entropy(ip_list):
            ip_list = ip_list.dropna()
            if len(ip_list) == 0:
                return 0.0
            counts = ip_list.value_counts()
            p = counts / counts.sum()
            p = p[p > 0]
            return float(-(p * np.log2(p)).sum())

        return basic_signal(lambda n: source_ip.groupby(n).apply(calculate_entropy))
```
Conceptually, entropy answers: “Is the traffic coming mainly from one sender, or from many senders with similar contribution?” Low entropy means concentration; high entropy means diversity and balance. This is powerful in analysis because it captures distribution shape, not just the number of categories.

Key implementation notes:
> - Conceptually, you compute Shannon entropy `H = -sum(p log2 p)`, which is easy to justify in a report
> - The empty-window guard returns 0.0 so the signal stays numeric and safe for ML
> - `p = p[p > 0]` is defensive; it avoids log issues if any zero probabilities appear

---

#### 10) Signal: unique_source_ip
This feature counts how many distinct sources appear per second. It is an easy-to-explain diversity indicator and provides an immediate sense of how many different senders were active.


```python
    def unique_source_ip():
        return basic_signal(lambda n: source_ip.groupby(n).nunique())
```
Conceptually, it answers: “How many unique talkers were present in this window?” It complements entropy: unique_source_ip measures diversity by presence, while entropy measures diversity by balance. Two windows can have the same number of unique sources but different entropy if one source dominates packet share.

Key implementation notes:
> - `nunique()` is the cleanest definition of “source diversity” at window level.
> - This pairs well with entropy: unique counts “how many”, entropy captures “how evenly distributed”.

---

#### 11) Signal: time_interval_variance
This feature measures timing irregularity inside each second by computing the variance of inter-arrival times. It looks at how packets are spaced within the window, not just how many there are.


```python
    def time_interval_variance():
        def calculate_variance(time_list):
            if len(time_list) < 2:
                return 0.0
            sorted_time = time_list.sort_values()
            dt = sorted_time.diff().dropna()
            if len(dt) == 0:
                return 0.0
            return float(dt.var())

        return basic_signal(lambda n: times.groupby(n).apply(calculate_variance))
```
Conceptually, it distinguishes regular patterns from bursty patterns even when packet_count is the same. For example, 20 packets evenly spaced and 20 packets arriving in clumps are different behaviors; this feature helps separate them. Returning \(0\) when there are fewer than two packets is principled because inter-arrival variability cannot be defined without at least two timestamps.

Key implementation notes:
> - Sorting timestamps then taking `diff()` is the correct way to compute inter-arrival gaps
> - The `< 2` guard is important because variance is undefined with fewer than 2 samples
> - Returning 0.0 on sparse windows keeps the feature stable and avoids NaNs

---

#### 12) Compute & save
This block assembles all computed signals into a single aligned feature table and exports it for downstream use. The key concept is alignment: every row corresponds to one time window, and every column is a signal value for that same window.


```python
    print("Calculating signals...")

    pcount = packet_count()
    tvolume = traffic_volume()
    sentropy = source_entropy()
    usip = unique_source_ip()
    tivariance = time_interval_variance()

    all_signals = pd.DataFrame({
        "time_window": list(global_index),
        "packet_count": pcount.values,
        "traffic_volume": tvolume.values,
        "source_ip_entropy": sentropy.values,
        "unique_source_ip": usip.values,
        "time_variance": tivariance.values
    }).fillna(0)

    all_signals.to_csv(OUT_FILE, index=False)

    print("Done.")
    print(f"Saved: {OUT_FILE} | rows={len(all_signals)}")
```
Conceptually, this is the point where you move from raw events (packets) to an ML-ready dataset (a time-series feature matrix). Exporting to CSV finalizes a clean pipeline boundary: capture and mapping produce inputs, signal extraction produces features, and later stages (visualization or ML) consume the output reliably.

Key implementation notes:
> - Building one DataFrame with a shared `time_window` axis makes the output easy to plot, debug, and feed to ML.
> - `.fillna(0)` is a final safety net, although basic_signal already prevents missing windows in most cases.
> - Printing the saved path and row count helps during demos and debugging.

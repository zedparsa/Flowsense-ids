import json
import pandas as pd
import numpy as np


# ==================== Config: load mapping ====================

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


# ==================== Read CSV ====================

try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: File '{CSV_FILE}' not found!")
    raise
except pd.errors.EmptyDataError:
    print("Error: CSV file is empty!")
    raise


# ==================== Rename columns to standard schema ====================

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


# ==================== Cleaning ====================

df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0)
df["source"] = df["source"].fillna("UNKNOWN")

df = df.dropna(subset=["time"])

if len(df) == 0:
    raise ValueError("No valid rows after cleaning (time is all invalid).")


# ==================== Build shared time windows ====================

times = df["time"]
n = pd.Series(np.floor(times).astype(int), index=times.index)

min_n = int(n.min())
max_n = int(n.max())
global_index = range(min_n, max_n + 1)

source_ip = df["source"]
size = df["length"]


# ==================== Core helper ====================

def basic_signal(compute_function):
    signal = compute_function(n)
    return signal.reindex(global_index, fill_value=0)


# ==================== Signals ====================

def packet_count():
    return basic_signal(lambda n: n.value_counts().sort_index())

def traffic_volume():
    return basic_signal(lambda n: size.groupby(n).sum())

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

def unique_source_ip():
    return basic_signal(lambda n: source_ip.groupby(n).nunique())

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


# ==================== Compute & Save ====================

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

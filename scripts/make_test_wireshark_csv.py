import numpy as np
import pandas as pd
from dataclasses import dataclass

# ==================== Config ====================


OUT_WIFIRESHARK = "test.wireshark.csv"
OUT_GT_WINDOWS = "ground_truth_windows.csv"

SEED = 42

NORMAL_SECONDS = 1200          # 20 min
DDOS_SECONDS = 120             # 2 min
NORMAL2_SECONDS = 180          # 3 min
PORTSCAN_SECONDS = 120         # 2 min
NORMAL3_SECONDS = 180          # 3 min
BOTNET_SECONDS = 240           # 4 min

BASE_TIME = 0.0

# Rates are packets per second (mean of Poisson)
RATES = {
    "Normal": 12,
    "DDoS": 180,
    "PortScan": 140,
    "Botnet": 45
}

# Source pool sizes
SOURCES = {
    "Normal": 25,
    "DDoS": 450,
    "PortScan": 2,
    "Botnet": 3
}

# Packet length distributions (bytes)
LENGTH = {
    "Normal": ("normal", 700, 250, 60, 1514),    # mean, std, min, max
    "DDoS": ("normal", 220, 70, 60, 600),
    "PortScan": ("normal", 90, 25, 60, 180),
    "Botnet": ("normal", 160, 40, 60, 300),
}

# Botnet adds periodic bursts to help sustained MA rule
BOTNET_BURST_EVERY = 15   # seconds
BOTNET_BURST_ADD = 60     # extra packets on burst seconds

# ==================== Helpers ====================

@dataclass
class Segment:
    name: str
    seconds: int

def ip_pool(n, base="10.0.0."):
    if n <= 0:
        return ["UNKNOWN"]
    n = min(n, 250)
    return [f"{base}{i+1}" for i in range(n)]

def sample_lengths(dist_cfg, count, rng):
    kind, mu, sigma, vmin, vmax = dist_cfg
    if count <= 0:
        return np.array([], dtype=int)

    if kind == "normal":
        x = rng.normal(mu, sigma, size=count)
    else:
        x = rng.uniform(vmin, vmax, size=count)

    x = np.clip(x, vmin, vmax)
    return x.astype(int)

def generate_segment(seg: Segment, t0: float, rng: np.random.Generator):
    rows = []
    gt = []

    srcs = ip_pool(SOURCES[seg.name], base="10.1.0." if seg.name != "DDoS" else "172.16.0.")
    rate = RATES[seg.name]
    len_cfg = LENGTH[seg.name]

    for s in range(seg.seconds):
        lam = rate

        if seg.name == "Botnet" and (s % BOTNET_BURST_EVERY == 0) and s > 0:
            lam = rate + BOTNET_BURST_ADD

        k = int(rng.poisson(lam=lam))
        if k == 0:
            gt.append((int(np.floor(t0 + s)), seg.name))
            continue

        times = (t0 + s) + rng.random(size=k)
        sources = rng.choice(srcs, size=k, replace=True)
        lengths = sample_lengths(len_cfg, k, rng)

        for i in range(k):
            rows.append((float(times[i]), str(sources[i]), int(lengths[i])))

        gt.append((int(np.floor(t0 + s)), seg.name))

    return rows, gt, t0 + seg.seconds

# ==================== Build Dataset ====================

def main():
    rng = np.random.default_rng(SEED)

    plan = [
        Segment("Normal", NORMAL_SECONDS),
        Segment("DDoS", DDOS_SECONDS),
        Segment("Normal", NORMAL2_SECONDS),
        Segment("PortScan", PORTSCAN_SECONDS),
        Segment("Normal", NORMAL3_SECONDS),
        Segment("Botnet", BOTNET_SECONDS),
    ]

    print("=" * 60)
    print("SYNTHETIC WIRESHARK CSV GENERATOR")
    print("=" * 60)
    print("\nGenerating segments...")

    all_rows = []
    all_gt = []
    t = BASE_TIME

    for seg in plan:
        print(f"  Segment: {seg.name:8s} | seconds={seg.seconds}")
        rows, gt, t = generate_segment(seg, t, rng)
        all_rows.extend(rows)
        all_gt.extend(gt)

    if len(all_rows) == 0:
        raise ValueError("Generated 0 packets. Check rates/seconds.")

    df = pd.DataFrame(all_rows, columns=["Time", "Source", "Length"])
    df = df.sort_values("Time").reset_index(drop=True)
    df.insert(0, "No.", np.arange(1, len(df) + 1))

    df.to_csv(OUT_WIFIRESHARK, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT_WIFIRESHARK} | packets={len(df)}")

    gt_df = pd.DataFrame(all_gt, columns=["time_window", "segment_label"])
    gt_df.to_csv(OUT_GT_WINDOWS, index=False, encoding="utf-8")
    print(f"Saved: {OUT_GT_WINDOWS} | windows={len(gt_df)}")

    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        raise

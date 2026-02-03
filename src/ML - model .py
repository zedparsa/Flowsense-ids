import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



# ==================== Config ====================


CSV_FILE = "network_signals.csv"
OUT_FILE = "ml_results.csv"

TRAIN_RATIO = 0.7
VOTE_THRESHOLD = 2
RANDOM_STATE = 42



# ==================== Load Data ====================


print("="*60)
print("NETWORK ANOMALY DETECTION - ML PIPELINE")
print("="*60)

print("\nLoading data...")

try:
    signals = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Error: File '{CSV_FILE}' not found!")
    raise

if len(signals) == 0:
    raise ValueError("Error: Empty dataset!")



# ==================== Feature Engineering ====================


print("\nFeature engineering...")

features_basic = ['packet_count', 'traffic_volume', 'source_ip_entropy', 
                  'unique_source_ip', 'time_variance']

# Ratio features
signals['avg_packet_size'] = signals['traffic_volume'] / (signals['packet_count'] + 1)
signals['entropy_per_ip'] = signals['source_ip_entropy'] / (signals['unique_source_ip'] + 1)

# Rolling features (temporal patterns)
for w in [5, 10, 30]:
    signals[f'packet_ma{w}'] = signals['packet_count'].rolling(w).mean().fillna(0)
    signals[f'packet_std{w}'] = signals['packet_count'].rolling(w).std().fillna(0)

# Rate of change
signals['packet_change'] = signals['packet_count'].diff().fillna(0)

features_all = features_basic + [
    'avg_packet_size', 'entropy_per_ip',
    'packet_ma5', 'packet_std5', 'packet_ma10', 'packet_std10', 'packet_ma30', 'packet_std30',
    'packet_change'
]

print(f"  Created {len(features_all)} features")



# ==================== Train/Test Split (Time-based) ====================


print("\nSplitting data (time-based)...")

train_size = int(TRAIN_RATIO * len(signals))
train_indices = range(0, train_size)
test_indices = range(train_size, len(signals))

print(f"  Train: {train_size} windows ({TRAIN_RATIO*100:.0f}%)")
print(f"  Test:  {len(signals)-train_size} windows ({(1-TRAIN_RATIO)*100:.0f}%)")

# Normalization (fit on train only!)
scaler = StandardScaler()
X_train = scaler.fit_transform(signals.loc[train_indices, features_all])
X_test = scaler.transform(signals.loc[test_indices, features_all])
X_all = scaler.transform(signals[features_all])



# ==================== Layer 1: ML Models ====================


print("\n" + "="*60)
print("TRAINING ML MODELS")
print("="*60)

models = {
    'IsolationForest': IsolationForest(contamination=0.1, random_state=RANDOM_STATE, n_estimators=100),
    'OneClassSVM': OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
}

predictions_train = {}
predictions_test = {}
predictions_all = {}

for name, model in models.items():
    print(f"\n{name}...")
    model.fit(X_train)
    
    predictions_train[name] = model.predict(X_train)
    predictions_test[name] = model.predict(X_test)
    predictions_all[name] = model.predict(X_all)
    
    train_anomalies = (predictions_train[name] == -1).sum()
    test_anomalies = (predictions_test[name] == -1).sum()
    print(f"  Train: {train_anomalies} anomalies ({train_anomalies/len(train_indices)*100:.1f}%)")
    print(f"  Test:  {test_anomalies} anomalies ({test_anomalies/len(test_indices)*100:.1f}%)")

# Get IF decision scores for anomaly scoring
if_model = models['IsolationForest']
if_scores_all = if_model.decision_function(X_all)



# ==================== Layer 2: Clustering ====================


print("\n" + "="*60)
print("CLUSTERING ANALYSIS")
print("="*60)

print("\nKMeans clustering...")

kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
kmeans.fit(X_train)

# Distance-based anomaly detection
distances_all = np.min(kmeans.transform(X_all), axis=1)
threshold_kmeans = np.percentile(distances_all[train_indices], 90)

print(f"  Distance threshold: {threshold_kmeans:.3f}")

predictions_train['KMeans'] = np.where(distances_all[train_indices] > threshold_kmeans, -1, 1)
predictions_test['KMeans'] = np.where(distances_all[test_indices] > threshold_kmeans, -1, 1)
predictions_all['KMeans'] = np.where(distances_all > threshold_kmeans, -1, 1)

print(f"  Train: {(predictions_train['KMeans']==-1).sum()} anomalies")
print(f"  Test:  {(predictions_test['KMeans']==-1).sum()} anomalies")



# ==================== Layer 3: Expert System ====================


print("\n" + "="*60)
print("EXPERT SYSTEM (Rule-Based AI)")
print("="*60)

# Learn thresholds from training data
q95_packet = signals.loc[train_indices, 'packet_count'].quantile(0.95)
q95_ip = signals.loc[train_indices, 'unique_source_ip'].quantile(0.95)
q25_size = signals.loc[train_indices, 'avg_packet_size'].quantile(0.25)
q05_entropy = signals.loc[train_indices, 'source_ip_entropy'].quantile(0.05)
q95_volume = signals.loc[train_indices, 'traffic_volume'].quantile(0.95)
q75_ma = signals.loc[train_indices, 'packet_ma30'].quantile(0.75)

print(f"\nLearned thresholds from training data:")
print(f"  High packet count: > {q95_packet:.1f}")
print(f"  High unique IPs: > {q95_ip:.1f}")
print(f"  Low packet size: < {q25_size:.1f}")
print(f"  Low entropy: < {q05_entropy:.2f}")


def expert_system(row):
    """
    Rule-based AI with data-driven thresholds.
    Returns (score, attack_type).
    """
    score = 0
    attack_type = "Normal"
    
    # Rule 1: DDoS (high packets + many IPs)
    if row['packet_count'] > q95_packet and row['unique_source_ip'] > q95_ip:
        score += 5
        attack_type = "DDoS"
    
    # Rule 2: Port Scan (many small packets, few IPs)
    elif row['packet_count'] > q95_packet and row['avg_packet_size'] < q25_size:
        score += 4
        attack_type = "Port_Scan"
    
    # Rule 3: Botnet (low entropy + sustained traffic)
    elif row['source_ip_entropy'] < q05_entropy and row['packet_ma30'] > q75_ma:
        score += 3
        attack_type = "Botnet"
    
    # Rule 4: Data Exfiltration (high volume, few packets)
    elif row['traffic_volume'] > q95_volume and row['packet_count'] < q95_packet/2:
        score += 4
        attack_type = "Data_Exfiltration"
    
    # Rule 5: Sudden Spike
    elif row['packet_count'] > row['packet_ma30'] + 2 * row['packet_std30'] and row['packet_std30'] > 0:
        score += 2
        attack_type = "Sudden_Spike"
    
    return score, attack_type


# Apply expert system
signals[['expert_score', 'attack_type']] = signals.apply(
    lambda row: pd.Series(expert_system(row)), axis=1
)

predictions_train['ExpertSystem'] = np.where(signals.loc[train_indices, 'expert_score'] >= 3, -1, 1)
predictions_test['ExpertSystem'] = np.where(signals.loc[test_indices, 'expert_score'] >= 3, -1, 1)
predictions_all['ExpertSystem'] = np.where(signals['expert_score'] >= 3, -1, 1)

print(f"\n  Train: {(predictions_train['ExpertSystem']==-1).sum()} anomalies")
print(f"  Test:  {(predictions_test['ExpertSystem']==-1).sum()} anomalies")



# ==================== Layer 4: Ensemble (Majority Voting) ====================


print("\n" + "="*60)
print("ENSEMBLE FUSION (Majority Voting)")
print("="*60)

# Count how many models voted for anomaly
votes_train = np.array(list(predictions_train.values()))
votes_test = np.array(list(predictions_test.values()))
votes_all = np.array(list(predictions_all.values()))

anomaly_votes_train = (votes_train == -1).sum(axis=0)
anomaly_votes_test = (votes_test == -1).sum(axis=0)
anomaly_votes_all = (votes_all == -1).sum(axis=0)

# Final decision: if VOTE_THRESHOLD or more models agree → anomaly
final_prediction_train = np.where(anomaly_votes_train >= VOTE_THRESHOLD, -1, 1)
final_prediction_test = np.where(anomaly_votes_test >= VOTE_THRESHOLD, -1, 1)
final_prediction_all = np.where(anomaly_votes_all >= VOTE_THRESHOLD, -1, 1)

print(f"\nVoting rule: {VOTE_THRESHOLD}+ models agree → Anomaly")
print(f"  Train: {(final_prediction_train == -1).sum()} anomalies")
print(f"  Test:  {(final_prediction_test == -1).sum()} anomalies")

# Store results
signals['prediction'] = final_prediction_all
signals['votes'] = anomaly_votes_all

# Anomaly score: combination of IF decision + expert + distance
if_score_norm = (if_scores_all - if_scores_all.min()) / (if_scores_all.max() - if_scores_all.min() + 1e-9)
expert_score_norm = signals['expert_score'] / 5.0
distance_norm = (distances_all - distances_all.min()) / (distances_all.max() - distances_all.min() + 1e-9)

signals['anomaly_score'] = (
    (1 - if_score_norm) * 60 +
    expert_score_norm * 30 +
    distance_norm * 10
)



# ==================== Evaluation ====================


print("\n" + "="*60)
print("EVALUATION")
print("="*60)

anomalies_train = signals.loc[train_indices][signals.loc[train_indices, 'prediction'] == -1]
anomalies_test = signals.loc[test_indices][signals.loc[test_indices, 'prediction'] == -1]

print(f"\nTRAIN SET:")
print(f"  Total: {len(train_indices)} windows")
print(f"  Detected: {len(anomalies_train)} anomalies ({len(anomalies_train)/len(train_indices)*100:.1f}%)")

print(f"\nTEST SET:")
print(f"  Total: {len(test_indices)} windows")
print(f"  Detected: {len(anomalies_test)} anomalies ({len(anomalies_test)/len(test_indices)*100:.1f}%)")

if len(anomalies_test) > 0:
    print(f"  Avg anomaly score: {anomalies_test['anomaly_score'].mean():.1f}/100")
    print(f"  Max anomaly score: {anomalies_test['anomaly_score'].max():.1f}/100")
    print(f"  High confidence (4 votes): {(anomalies_test['votes'] == 4).sum()} cases")
    
    # Case study: strongest anomaly
    print(f"\n  📌 CASE STUDY - Strongest Anomaly:")
    top_anomaly = anomalies_test.loc[anomalies_test['anomaly_score'].idxmax()]
    print(f"     Time window: {int(top_anomaly['time_window'])} seconds")
    print(f"     Packets: {int(top_anomaly['packet_count'])}")
    print(f"     Unique IPs: {int(top_anomaly['unique_source_ip'])}")
    print(f"     Traffic: {int(top_anomaly['traffic_volume'])} bytes")
    print(f"     Attack type: {top_anomaly['attack_type']}")
    print(f"     Anomaly score: {top_anomaly['anomaly_score']:.1f}/100")
    print(f"     Model votes: {int(top_anomaly['votes'])}/4")

print("\n--- Per Model Detection (Test Set) ---")
for name in ['IsolationForest', 'OneClassSVM', 'KMeans', 'ExpertSystem']:
    count = (predictions_test[name] == -1).sum()
    print(f"  {name:20s}: {count:3d} anomalies")
print(f"  {'Ensemble (2+ votes)':20s}: {(final_prediction_test == -1).sum():3d} anomalies")

print("\n--- Model Agreement (Test Set) ---")
all_agree = (anomaly_votes_test == 4).sum()
majority = (anomaly_votes_test >= 2).sum()
print(f"  All 4 models agree: {all_agree} windows")
print(f"  2+ models agree: {majority} windows")

print("\n--- Attack Types Detected (All Data) ---")
attack_counts = signals[signals['prediction'] == -1]['attack_type'].value_counts()
for attack, count in attack_counts.items():
    print(f"  {attack}: {count}")



# ==================== Save Results ====================


signals.to_csv(OUT_FILE, index=False)
print(f"\n✓ Results saved: {OUT_FILE}")



# ==================== Plot Theme (Dark / Professional) ====================


plt.style.use("dark_background")

# Color palette (smoky dark theme)
BG = "#0d1117"          # background
PANEL = "#161b22"       # plot background
GRID = "#30363d"        # grid lines
TEXT = "#c9d1d9"        # primary text
MUTED = "#8b949e"       # secondary text
SPLIT = "#6e7681"       # train/test split line

NORMAL = "#58a6ff"      # soft blue (normal traffic)
ANOMALY = "#f85149"     # soft red (anomaly)
SCORE = "#ffa657"       # soft orange (score)
VOLUME = "#56d364"      # soft green (volume)

# Font settings (DejaVu Sans - professional & readable)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.alpha": 0.3,
    "axes.grid": True,
    
    "legend.frameon": True,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
})



# ==================== Visualizations ====================


print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

anomalies_all = signals[signals['prediction'] == -1]

# Plot 1: Main timeline
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Packet count
axes[0].plot(
    signals['time_window'], signals['packet_count'],
    alpha=0.85, color=NORMAL, linewidth=1.8, label='Traffic'
)
axes[0].axvline(
    x=signals.loc[train_size-1, 'time_window'],
    color=SPLIT, linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split'
)
axes[0].scatter(
    anomalies_all['time_window'], anomalies_all['packet_count'],
    c=anomalies_all['anomaly_score'], cmap='Reds', s=100,
    edgecolors=PANEL, linewidths=1, label='Anomaly', zorder=5
)
axes[0].set_ylabel('Packet Count', fontsize=12, color=TEXT)
axes[0].legend(loc='upper right', framealpha=0.95)
axes[0].set_title('Network Anomaly Detection - Hybrid AI System', fontsize=14, fontweight='bold', color=TEXT, pad=15)

# Anomaly score
axes[1].fill_between(
    signals['time_window'], 0, signals['anomaly_score'],
    alpha=0.4, color=SCORE
)
axes[1].axvline(
    x=signals.loc[train_size-1, 'time_window'],
    color=SPLIT, linestyle='--', linewidth=2, alpha=0.7
)
axes[1].set_ylabel('Anomaly Score', fontsize=12, color=TEXT)

# Traffic volume
axes[2].plot(
    signals['time_window'], signals['traffic_volume'],
    alpha=0.85, color=VOLUME, linewidth=1.6
)
axes[2].axvline(
    x=signals.loc[train_size-1, 'time_window'],
    color=SPLIT, linestyle='--', linewidth=2, alpha=0.7
)
axes[2].scatter(
    anomalies_all['time_window'], anomalies_all['traffic_volume'],
    color=ANOMALY, s=80, edgecolors=PANEL, linewidths=1, zorder=5
)
axes[2].set_xlabel('Time Window (seconds)', fontsize=12, color=TEXT)
axes[2].set_ylabel('Traffic Volume (bytes)', fontsize=12, color=TEXT)

plt.tight_layout()
plt.savefig('anomaly_detection_complete.png', dpi=300, bbox_inches='tight', facecolor=BG)
print("  ✓ Saved: anomaly_detection_complete.png")

# Plot 2: Model comparison
fig, ax = plt.subplots(figsize=(10, 6))
model_counts = {}
for name in ['IsolationForest', 'OneClassSVM', 'KMeans', 'ExpertSystem']:
    model_counts[name] = (predictions_test[name] == -1).sum()
model_counts['Ensemble'] = (final_prediction_test == -1).sum()

colors_bars = [NORMAL, '#9ecbff', '#7ee787', '#d2a8ff', SCORE]
bars = ax.bar(
    model_counts.keys(), model_counts.values(),
    color=colors_bars, edgecolor=GRID, linewidth=1.5
)
ax.set_ylabel('Anomalies Detected (Test Set)', fontsize=12, color=TEXT)
ax.set_title('Model Comparison', fontsize=14, fontweight='bold', color=TEXT, pad=15)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2., height + 0.8,
        f'{int(height)}', ha='center', va='bottom',
        fontweight='bold', fontsize=11, color=TEXT
    )

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor=BG)
print("  ✓ Saved: model_comparison.png")

# Plot 3: Feature correlation
fig, ax = plt.subplots(figsize=(10, 8))
important_features = ['packet_count', 'traffic_volume', 'source_ip_entropy', 
                      'unique_source_ip', 'avg_packet_size', 'anomaly_score']
corr = signals[important_features].corr()
sns.heatmap(
    corr, annot=True, fmt='.2f',
    cmap='rocket_r', center=0,
    square=True, linewidths=1.2, linecolor=GRID,
    cbar_kws={"shrink": 0.8},
    annot_kws={"fontsize": 10, "color": "#e6edf3"}
)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', color=TEXT, pad=15)
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight', facecolor=BG)
print("  ✓ Saved: feature_correlation.png")


plt.show()


# ==================== Summary ====================


print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n✓ {len(features_all)} features engineered")
print(f"✓ 4 models trained (IF, SVM, KMeans, Expert)")
print(f"✓ Majority voting ensemble ({VOTE_THRESHOLD}+ votes)")
print(f"✓ {len(anomalies_all)} anomalies detected")
print(f"✓ 3 visualizations generated (dark theme)\n")

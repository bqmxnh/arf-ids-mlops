from pathlib import Path
import pandas as pd

# ============================================================
# Paths
# ============================================================
base = Path("dataset/balanced_syn_cleaned.csv")
stream = Path("dataset/stream_data.csv")      # dữ liệu online có label thật
unlabeled = Path("dataset/unlabeled_log.csv") # dữ liệu hậu kiểm (có thể đã gán label)
merged = Path("dataset/merged_for_retrain.csv")

print("[MERGE] Starting dataset merge for retraining...")

# ============================================================
# 1. Load base dataset
# ============================================================
if not base.exists():
    raise FileNotFoundError("Base dataset not found!")

base_df = pd.read_csv(base)
merged_df = base_df.copy()
print(f"Loaded base dataset with {len(base_df)} samples.")

# ============================================================
# 2. Append stream_data.csv (nếu có và có label)
# ============================================================
if stream.exists():
    stream_df = pd.read_csv(stream)
    stream_df = stream_df.dropna(subset=["Label"])
    if len(stream_df) > 0:
        merged_df = pd.concat([merged_df, stream_df], ignore_index=True)
        print(f"Added {len(stream_df)} labeled streaming samples.")
    else:
        print("stream_data.csv exists but no labeled samples found.")
else:
    print("stream_data.csv not found, skipping.")

# ============================================================
# 3. Append unlabeled_log.csv (nếu admin đã gán label)
# ============================================================
if unlabeled.exists():
    unlabeled_df = pd.read_csv(unlabeled)
    unlabeled_df = unlabeled_df.dropna(subset=["Label"])
    if len(unlabeled_df) > 0:
        merged_df = pd.concat([merged_df, unlabeled_df], ignore_index=True)
        print(f"Added {len(unlabeled_df)} verified unlabeled_log samples.")
    else:
        print("unlabeled_log.csv has no labeled rows (admin chưa gán).")
else:
    print("unlabeled_log.csv not found, skipping.")

# ============================================================
# 4. Save final merged dataset
# ============================================================
merged_df.to_csv(merged, index=False)
print(f"Saved merged dataset to {merged} ({len(merged_df)} total samples)")

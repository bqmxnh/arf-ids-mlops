import pandas as pd
from pathlib import Path

base = Path("dataset/balanced_syn_cleaned.csv")
stream = Path("dataset/stream_data.csv")
merged = Path("dataset/merged_for_retrain.csv")

print("🧩 Merging datasets...")

base_df = pd.read_csv(base)
if stream.exists():
    stream_df = pd.read_csv(stream)
    df = pd.concat([base_df, stream_df], ignore_index=True)
    print(f"✅ Added {len(stream_df)} streaming rows.")
else:
    df = base_df
    print("⚠️ No streaming data found.")

df.to_csv(merged, index=False)
print(f"📦 Saved merged dataset to {merged}")

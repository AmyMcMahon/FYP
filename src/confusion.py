import pandas as pd
import os

# Paths to your two result CSVs
manual_path = "../datasets/results/bert-hyper-test.csv"
hyper_path  = "../datasets/results/bert-manual-test.csv"


# Load into DataFrames
manual_df = pd.read_csv(manual_path)
hyper_df  = pd.read_csv(hyper_path)

# Ensure we’re comparing on the same key (the comment text)
manual_bodies = set(manual_df["body"])
hyper_bodies  = set(hyper_df["body"])

# Compute differences
only_in_manual = manual_bodies - hyper_bodies
only_in_hyper  = hyper_bodies  - manual_bodies

# Extract the rows
manual_only_df = manual_df[manual_df["body"].isin(only_in_manual)]
hyper_only_df  = hyper_df[hyper_df["body"].isin(only_in_hyper)]

# Prepare output directory
out_dir = "datasets/results"
os.makedirs(out_dir, exist_ok=True)

# Write to CSV
manual_only_path = os.path.join(out_dir, "manual_only_false_negatives.csv")
hyper_only_path  = os.path.join(out_dir, "hyper_only_false_negatives.csv")

manual_only_df.to_csv(manual_only_path, index=False)
hyper_only_df.to_csv(hyper_only_path, index=False)

print(f"Saved {len(manual_only_df)} rows to {manual_only_path}")
print(f"Saved {len(hyper_only_df)} rows to {hyper_only_path}")

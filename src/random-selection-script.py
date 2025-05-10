import pandas as pd
import json
import random

# === CONFIG ===
input_path = "datasets/test/womenEngineers_submission_filtered.json"
output_path = "sampled_submissions.xlsx"
sample_size = 300

# === LOAD DATA ===
# Each line is a separate JSON object (JSONL format)
with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# === COMBINE TEXT FIELDS PROPERLY ===
if "title" in df.columns and "selftext" in df.columns:
    df["body"] = df["title"].fillna('') + "\n\n" + df["selftext"].fillna('')
elif "title" in df.columns:
    df["body"] = df["title"].fillna('')
elif "selftext" in df.columns:
    df["body"] = df["selftext"].fillna('')
else:
    raise ValueError("No 'title' or 'selftext' fields found in submission data.")


# Drop empty 'body' values
df = df.dropna(subset=["body"])
df = df[df["body"].str.strip() != ""]

# === SAMPLE ===
sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

# Add empty label column
sampled_df["label"] = ""

# === EXPORT TO EXCEL ===
sampled_df.to_excel(output_path, index=False)

print(f"✅ Exported {len(sampled_df)} sampled submissions to '{output_path}'")

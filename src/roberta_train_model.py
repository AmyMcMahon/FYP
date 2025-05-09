import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Progress bar
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def df_diagnostics(name, df):
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("Missing values per column:")
    print(df.isna().sum())
    print("\n.info():")
    df.info()
    print("\n.head():")
    print(df.head())
    print("\n" + "="*40)


# Load environment variables
load_dotenv()
huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Load RoBERTa model and tokenizer
# model_name = "cardiffnlp/twitter-roberta-base-sentiment"
model_name = "bert-ear-manual"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize sentiment analysis pipeline with truncation enabled
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True, max_length=512)

df = pd.read_csv("../datasets/train/final_labels.csv")
df = df[['body', 'level_1']].dropna()

new_df = pd.read_csv("../datasets/train/SD_dataset_FINAL.csv")
comments_df = pd.read_excel("../datasets/train/sampled_comments.xlsx")
submissions_df = pd.read_excel("../datasets/train/sampled_submissions.xlsx")

combined_new = pd.concat([comments_df, submissions_df], ignore_index=True)
combined_new['level_1'] = combined_new['label'].map({
    'Neutral': 'Nonmisogynistic',
    'Misogynistic': 'Misogynistic',
    'Mentions Misogyny': 'Misogynistic'
})
combined_new = combined_new[['body', 'level_1']].dropna()

new_df['level_1'] = new_df['level_1'].map({1: 'Misogynistic', 0: 'Nonmisogynistic'})
df = pd.concat([df, new_df], ignore_index=True)
df = pd.concat([df, combined_new], ignore_index=True)

df = df[~df["body"].isin(["[removed]", "[deleted]"])]

# 2) Drop exact duplicate comments
df = df.drop_duplicates(subset="body").reset_index(drop=True)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['level_1'])

dataset = Dataset.from_pandas(df[['body', 'label']])

# dataset = dataset.train_test_split(test_size=0.2)
test_dataset = dataset

# Dictionary to count occurrences of each level
level_counts = {"misogynist": 0, "non-misogynist": 0}

# List to store labeled data
labeled_data = []

# Process the dataset with a progress bar
for line in tqdm(test_dataset, desc="Classifying comments", unit="comment"):
    input_text = line.get("body", "").strip()

    # Run sentiment classification
    result = classifier(input_text)
    sentiment_label = result[0]['label']  
    # Update level counts
    level_counts[sentiment_label] += 1

    # Save the labeled data
    labeled_data.append({"body": input_text, "sentiment": sentiment_label})

# Save labeled data to CSV
labelled_df = pd.DataFrame(labeled_data)
label_map = {
    "non-misogynist": 0,
    "misogynist":     1
}

# 2) Apply to your predictions
labelled_df["predicted_label"] = labelled_df["sentiment"].map(label_map)

test_df = pd.DataFrame(test_dataset)
test_df = test_df.rename(columns={"label": "true_label"})

print("=== test_df head ===")
print(test_df.head())

# Show first 5 rows of labelled_df
print("\n=== labelled_df head ===")
print(labelled_df.head())

merged = pd.merge(
    test_df[["body", "true_label"]],
    labelled_df[["body", "predicted_label"]],
    on="body",
    how="inner"
)

false_negatives = merged[
    (merged["true_label"] == 0) &
    (merged["predicted_label"] == 1)
]
false_negatives.to_csv('../datasets/results/bert-manual-test.csv', index=False)


# # 4) Compute confusion matrix
# y_true = merged["true_label"]
# y_pred = merged["predicted_label"]
# cm = confusion_matrix(y_true, y_pred)

# # 5) Plot with labels
# fig, ax = plt.subplots(figsize=(6,6))
# cax = ax.matshow(cm, cmap="Blues")
# fig.colorbar(cax)

# # Set tick labels
# classes = ["Misogynistic", "Non-misogynistic"]
# ax.set_xticks([0,1])
# ax.set_yticks([0,1])
# ax.set_xticklabels(classes, rotation=0)      # parallel labels
# ax.set_yticklabels(classes)

# # Ensure x-axis labels are on the bottom
# ax.xaxis.set_label_position('bottom')
# ax.xaxis.tick_bottom()

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix: bert-ear-hyper")

# # Annotate each cell with white text if the cell is "dark"
# threshold = cm.max() / 2.0
# for (i, j), val in np.ndenumerate(cm):
#     color = "white" if val > threshold else "black"
#     ax.text(j, i, val, ha="center", va="center", color=color)

# plt.tight_layout()
# plt.savefig("../images/bert-hyper-clean-total-confusion.png")
# plt.show()
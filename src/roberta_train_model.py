import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm  # Progress bar
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load environment variables
load_dotenv()
huggingface_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Load RoBERTa model and tokenizer
# model_name = "cardiffnlp/twitter-roberta-base-sentiment"
# model_name = "MilaNLProc/bert-base-uncased-ear-misogyny"

model_name = "EleutherAI/pythia-6.9b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Initialize sentiment analysis pipeline with truncation enabled
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True, max_length=512)

# Path to input dataset
file_path = '../datasets/test/womenEngineers_submission_filtered.json'

# Path to output CSV
output_csv_path = '../datasets/results/pythia-sexism-submission.csv'

# Dictionary to count occurrences of each level
level_counts = {"misogynist": 0, "non-misogynist": 0}

# List to store labeled data
labeled_data = []

# Read all lines from the file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Process the dataset with a progress bar
for line in tqdm(lines, desc="Classifying comments", unit="comment"):
    data = json.loads(line)
    input_text = data.get("body", "").strip()

    # Run sentiment classification
    result = classifier(input_text)
    sentiment_label = result[0]['label']  # "LABEL_0", "LABEL_1", "LABEL_2"

    # Update level counts
    level_counts[sentiment_label] += 1

    # Save the labeled data
    labeled_data.append({"body": input_text, "sentiment": sentiment_label})

# Save labeled data to CSV
df = pd.DataFrame(labeled_data)
df.to_csv(output_csv_path, index=False)
print(f"Saved labeled data to {output_csv_path}")

# Plot the distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=list(level_counts.keys()), 
            y=list(level_counts.values()), 
            hue=list(level_counts.keys()), 
            palette="Blues", 
            legend=False)

plt.xlabel("Sentiment Level")
plt.ylabel("Number of Comments")
plt.title("Distribution of Submissions Across Sentiment Levels (pythia-70m-deduped)")
plt.savefig("../images/Pythia-sexism-submission.png")
plt.show()

import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['level_1'])  

dataset = Dataset.from_pandas(df[['body', 'label']])

dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
test_dataset = dataset['test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "MilaNLProc/bert-base-uncased-ear-misogyny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#model = AutoModelForSequenceClassification.from_pretrained("misogyny-classifier")
#tokenizer = AutoTokenizer.from_pretrained("misogyny-classifier")

def tokenize_function(example):
    return tokenizer(example["body"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model_name,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_result = trainer.evaluate()
print("Evaluation Results:", eval_result)

file_path = '../datasets/test/womenEngineers_comment_filtered.json'
with open(file_path, 'r') as f:
    lines = f.readlines()

texts = []
for line in lines:
    data = json.loads(line)
    body = data.get("body", "").strip()  
    texts.append(body)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
inputs_dataset = Dataset.from_dict({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
})
predictions = trainer.predict(inputs_dataset).predictions  
pred_labels = predictions.argmax(axis=-1) 

level_counts = {"misogynist": sum(pred_labels == 1), "non-misogynist": sum(pred_labels == 0)}

plt.figure(figsize=(8, 6))
sns.barplot(x=list(level_counts.keys()), 
            y=list(level_counts.values()), 
            hue=list(level_counts.keys()), 
            palette="Blues", 
            legend=False)

plt.xlabel("Sentiment Level")
plt.ylabel("Number of Comments")
plt.title("Distribution of Comments Across Sentiment Levels (Bert EAR manual)")
plt.savefig("../images/hyperparams-bert-comments.png")
plt.show()

trainer.save_model("misogyny-classifier")
tokenizer.save_pretrained("misogyny-classifier")

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1. Load your dataset
df = pd.read_csv("../datasets/train/final_labels.csv")
df = df[['body', 'level_1']].dropna()

# 2. Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['level_1'])  # e.g., Nonmisogynistic = 1, Misogynistic = 0

# 3. Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[['body', 'label']])

# 4. Train/test split
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
test_dataset = dataset['test']

# 5. Load tokenizer & model
model_name = "MilaNLProc/bert-base-uncased-ear-misogyny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Tokenize the data
def tokenize_function(example):
    return tokenizer(example["body"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 7. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }

# 8. Training arguments
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

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 10. Train
trainer.train()

# 11. Evaluate
eval_result = trainer.evaluate()
print("Evaluation Results:", eval_result)

# (Optional) Save model
trainer.save_model("misogyny-classifier")
tokenizer.save_pretrained("misogyny-classifier")

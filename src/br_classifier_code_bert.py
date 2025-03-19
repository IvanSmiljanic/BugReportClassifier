# This program trains a bug report classifier to identify whether a bug report is performance-based or not

import pandas as pd
import numpy as np
import torch
import re
import glob
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Preprocessing functions (found in lab1)
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Setting up the device to use the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Combining CSV files into a unified dataset
csv_files = glob.glob('datasets/*.csv')
df_list = [pd.read_csv(csv) for csv in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Applying preprocessing
df['text'] = df.apply(
    lambda row: f"{row['Title']} {row['Body']}" if pd.notna(row['Body']) else row['Title'], axis=1
)
df['text'] = df['text'].apply(lambda text: remove_emoji(remove_html(text)))

texts = df['text'].to_list()
labels = df['class'].to_list()

# Splitting train and test data
# Using 70-30 split as recommended in lab1.pdf
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# Tokenization
model_name = 'microsoft/codebert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Creating class for the unified dataset
class BugReportDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
# Create datasets
train_dataset = BugReportDataset(train_encodings, train_labels)
test_dataset = BugReportDataset(test_encodings, test_labels)

# Model initialisation
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Function returning performance metrics of a model
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# This function will be fed to Optuna for hyperparameter optimisation
def objective(trial):
    # Training arguments for a CodeBERT model
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        save_strategy='no',
        logging_steps=50,
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
        num_train_epochs=trial.suggest_int('num_train_epochs', 2, 5),
        per_device_train_batch_size=trial.suggest_categorical('batch_size', [8, 16]),
        weight_decay=trial.suggest_float('weight_decay', 0.0, 0.3),
        load_best_model_at_end=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Instantiating a CodeBERT model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Training a model for a given set of hyperparameters
    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result['eval_f1']

# Finding the optimal hyperparameters for the model
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Output the best hyperparameters found
print("Best Hyperparameters:", study.best_params)

# Train and evaluate a model using best hyperparameters
best_args = TrainingArguments(
    output_dir='./best_model',
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=50,
    learning_rate=study.best_params['learning_rate'],
    num_train_epochs=study.best_params['num_train_epochs'],
    per_device_train_batch_size=study.best_params['batch_size'],
    weight_decay=study.best_params['weight_decay'],
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device),
    args=best_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()

# Output evaluation results
print("\n=== Best Transformer (CodeBERT) Model Results ===")
for key, value in eval_results.items():
    print(f"{key.replace('eval_', '').capitalize()}: {value:.4f}")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import time


# Configuration and Hyperparameters
MODEL_NAME = "distilbert-base-uncased"  # You can replace this with another pre-trained model
NUM_LABELS = 4  # AG News has 4 categories: World, Sports, Business, Sci/Tech
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)

# Load the AG News dataset from Hugging Face
dataset = load_dataset("ag_news")

# We'll split the training set into train/validation (e.g., 90/10 split)
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Define a collate function to tokenize batch examples
def collate_fn(batch):
    texts = [example["text"] for example in batch]
    labels = [example["label"] for example in batch]
    # Tokenize and pad sequences
    encoding = tokenizer(texts, truncation=True, padding="longest", max_length=MAX_LEN, return_tensors="pt")
    encoding["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoding

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in dataloader:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += input_ids.size(0)

    avg_loss = epoch_loss / total_samples
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy.item()

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            epoch_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += input_ids.size(0)

    avg_loss = epoch_loss / total_samples
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy.item()

# Main training loop
best_val_accuracy = 0
for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    elapsed = time.time() - start_time

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Time: {elapsed:.2f} sec\n")

    # Save the best model based on validation accuracy
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        model.save_pretrained("./trained_news_model")
        tokenizer.save_pretrained("./trained_news_model")
        print("Saved best model.")

print("Training complete!")

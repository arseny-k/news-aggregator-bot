import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Convert text into a fixed-size vector (naive character-based approach)
def text_to_vector(text, vector_size=100):
    vector = torch.zeros(vector_size)
    text = text or ""
    for i, char in enumerate(text):
        vector[i % vector_size] += ord(char)
    return vector / (len(text) if len(text) > 0 else 1)

# Define a simple classifier model
class NewsClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Custom Dataset using Hugging Face's datasets library
class AGNewsDataset(Dataset):
    def __init__(self, split="train", vector_size=100, max_samples=None):
        # Load the AG News dataset
        self.dataset = load_dataset("ag_news", split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))
        self.vector_size = vector_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample["text"]
        label = sample["label"]  # Labels are already 0-indexed in this dataset
        vector = text_to_vector(text, self.vector_size)
        return vector, label

# Training loop function
def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.float()
            labels = labels.long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete.")

# Parameters for our model and training
INPUT_SIZE = 100
HIDDEN_SIZE = 50
NUM_CLASSES = 4  # AG News has 4 classes: e.g., World, Sports, Business, Sci/Tech

# Instantiate the model
model = NewsClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# Create the dataset and dataloader (limiting samples for quicker training; remove max_samples for full training)
dataset = AGNewsDataset(split="train", vector_size=INPUT_SIZE, max_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
train_model(model, dataloader, epochs=10, lr=0.001)

# Save the trained model weights
torch.save(model.state_dict(), "ag_news_classifier.pth")
print("Model saved to ag_news_classifier.pth")

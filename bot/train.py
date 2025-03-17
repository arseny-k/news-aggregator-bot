import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import AG_NEWS

# Test Commit

# A simple function to convert text into a fixed-size vector
def text_to_vector(text, vector_size=100):
    vector = torch.zeros(vector_size)
    text = text or ""
    for i, char in enumerate(text):
        vector[i % vector_size] += ord(char)
    return vector / (len(text) if len(text) > 0 else 1)

# Define a simple PyTorch model for classification
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

# Create a custom Dataset for the AG_NEWS dataset
class AGNewsDataset(Dataset):
    def __init__(self, split='train', vector_size=100, max_samples=None):
        # Load AG_NEWS; each item is a tuple: (label, text)
        self.data = list(AG_NEWS(split=split))
        if max_samples is not None:
            self.data = self.data[:max_samples]
        self.vector_size = vector_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data[idx]
        # AG_NEWS labels are 1-indexed, so subtract 1 for 0-indexing
        label = label - 1
        vector = text_to_vector(text, self.vector_size)
        return vector, label

# Training loop
def train_model(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.float()  # Ensure inputs are floats
            labels = labels.long()   # Ensure labels are longs
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete.")

# Define training parameters
INPUT_SIZE = 100
HIDDEN_SIZE = 50
NUM_CLASSES = 4  # AG_NEWS dataset has 4 categories: typically "World", "Sports", "Business", "Sci/Tech"

# Define category names for reference
categories = ["World", "Sports", "Business", "Sci/Tech"]

# Instantiate the model
model = NewsClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# Create the dataset and dataloader (limiting samples for quicker training; remove max_samples for full training)
dataset = AGNewsDataset(split='train', vector_size=INPUT_SIZE, max_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
train_model(model, dataloader, epochs=10, lr=0.001)

# Save the trained model
torch.save(model.state_dict(), "ag_news_classifier.pth")
print("Model saved to ag_news_classifier.pth")

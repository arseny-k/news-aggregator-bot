import logging
import os
import requests
import torch
import torch.nn as nn
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define your PyTorch model and utility functions
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

def text_to_vector(text, vector_size=100):
    vector = torch.zeros(vector_size)
    text = text or ""
    for i, char in enumerate(text):
        vector[i % vector_size] += ord(char)
    return vector / (len(text) if len(text) > 0 else 1)

categories = ["World", "Sports", "Technology", "Entertainment", "Politics"]
INPUT_SIZE = 100
HIDDEN_SIZE = 50
NUM_CLASSES = len(categories)
model = NewsClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
model.eval()


# Instantiate the model.
model = NewsClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# Try to load trained weights.
try:
    model.load_state_dict(torch.load("ag_news_classifier.pth"))
    model.eval()
    print("Loaded trained model weights.")
except FileNotFoundError:
    print("Trained model weights not found, using untrained model.")

def get_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        return articles
    except requests.RequestException as e:
        logger.error("Error fetching news: %s", e)
        return []

# Asynchronous command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to the News Aggregator Bot!\nUse /news to fetch the latest headlines."
    )

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    articles = get_news()
    if not articles:
        await update.message.reply_text("Failed to fetch news. Please try again later.")
        return

    messages = []
    for article in articles[:5]:
        title = article.get('title', 'No Title')
        description = article.get('description', 'No Description')
        vector = text_to_vector(title)
        with torch.no_grad():
            output = model(vector.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            category = categories[predicted.item()]
        message = f"*{title}*\nCategory: {category}\n{description}\n"
        messages.append(message)
    final_message = "\n".join(messages)
    await update.message.reply_text(final_message, parse_mode="Markdown")

# Updated main function using Application builder
def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("news", news))
    logger.info("Bot started. Listening for commands...")
    application.run_polling()

if __name__ == '__main__':
    main()

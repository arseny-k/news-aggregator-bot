import logging
import os
import requests
import torch
import torch.nn as nn
from telegram.ext import Updater, CommandHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API keys from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define a simple PyTorch model for classifying news headlines
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

# Dummy text-to-vector converter: converts text into a fixed-size vector based on character values.
def text_to_vector(text, vector_size=100):
    vector = torch.zeros(vector_size)
    text = text or ""
    for i, char in enumerate(text):
        vector[i % vector_size] += ord(char)
    return vector / (len(text) if len(text) > 0 else 1)

# Define categories for the dummy classifier.
categories = ["World", "Sports", "Technology", "Entertainment", "Politics"]

# Instantiate the model with dummy parameters.
INPUT_SIZE = 100
HIDDEN_SIZE = 50
NUM_CLASSES = len(categories)
model = NewsClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
model.eval()  # Set the model to evaluation mode

# Function to fetch news from NewsAPI.org
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

# Telegram bot command handler for /start
def start(update, context):
    update.message.reply_text(
        "Welcome to the News Aggregator Bot!\nUse /news to fetch the latest headlines."
    )

# Telegram bot command handler for /news
def news(update, context):
    articles = get_news()
    if not articles:
        update.message.reply_text("Failed to fetch news. Please try again later.")
        return

    messages = []
    # Process a limited number of articles for brevity.
    for article in articles[:5]:
        title = article.get('title', 'No Title')
        description = article.get('description', 'No Description')

        # Convert the headline to a vector and classify it with our dummy model.
        vector = text_to_vector(title)
        with torch.no_grad():
            output = model(vector.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            category = categories[predicted.item()]

        message = f"*{title}*\nCategory: {category}\n{description}\n"
        messages.append(message)

    final_message = "\n".join(messages)
    update.message.reply_text(final_message, parse_mode="Markdown")

def main():
    # Create the Updater and pass in your bot's token.
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Register command handlers.
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("news", news))

    # Start the Bot.
    updater.start_polling()
    logger.info("Bot started. Listening for commands...")
    updater.idle()

if __name__ == '__main__':
    main()

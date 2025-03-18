import logging
import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("trained_news_model")
model = AutoModelForSequenceClassification.from_pretrained("trained_news_model")
model.eval()  # Ensure model is in evaluation mode

# AG News categories mapping
categories = ["World", "Sports", "Business", "Sci/Tech"]

# Function to fetch news headlines using NewsAPI.org
def get_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return articles
    except requests.RequestException as e:
        logger.error("Error fetching news: %s", e)
        return []

# Async command handler for /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to the News Aggregator Bot!\nUse /news to fetch the latest headlines."
    )

# Async command handler for /news
async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    articles = get_news()
    if not articles:
        await update.message.reply_text("Failed to fetch news. Please try again later.")
        return

    messages = []
    # Process only the first 5 articles for brevity
    for article in articles[:5]:
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")

        # Tokenize and classify the headline using the fine-tuned model
        inputs = tokenizer(
            title,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        category = categories[predicted_class]
        message = f"*{title}*\nCategory: {category}\n{description}\n"
        messages.append(message)

    final_message = "\n".join(messages)
    await update.message.reply_text(final_message, parse_mode="Markdown")

# Main function to set up and run the bot
def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("news", news))
    logger.info("Bot started. Listening for commands...")
    application.run_polling()

if __name__ == "__main__":
    main()

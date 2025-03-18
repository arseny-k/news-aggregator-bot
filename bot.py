import logging
import os
import requests
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue
from dotenv import load_dotenv
from datetime import time as dt_time

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
model.eval()

# AG News categories mapping
categories = ["World", "Sports", "Business", "Sci/Tech"]

# User preferences and shown news tracking
user_preferences = {}
shown_news = set()

# Function to fetch news headlines using NewsAPI.org
def get_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except requests.RequestException as e:
        logger.error("Error fetching news: %s", e)
        return []

# Function to classify news articles
def classify_news(title):
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
    return categories[predicted_class]

# Async command handler for /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    user_preferences[user_id] = set()
    await update.message.reply_text(
        "Welcome to the News Aggregator Bot!\nUse /news to fetch the latest headlines.\nUse /hide <category> to hide topics you don't like.\nUse /random to get a single random news article."
    )

# Async command handler for /hide
async def hide(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id

    # Ensure user entry exists
    if user_id not in user_preferences:
        user_preferences[user_id] = set()

    if len(context.args) == 0:
        await update.message.reply_text("Usage: /hide <category>. Available categories: World, Sports, Business, Sci/Tech")
        return

    category = " ".join(context.args).capitalize()
    if category not in categories:
        await update.message.reply_text("Invalid category. Available categories: World, Sports, Business, Sci/Tech")
        return

    user_preferences[user_id].add(category)
    await update.message.reply_text(f"Category '{category}' has been hidden.")


# Async command handler for /news
async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    articles = get_news()
    if not articles:
        await update.message.reply_text("Failed to fetch news. Please try again later.")
        return
    
    messages = []
    for article in articles:
        title = article.get("title", "No Title")
        if title in shown_news:
            continue
        
        category = classify_news(title)
        if user_id in user_preferences and category in user_preferences[user_id]:
            continue
        
        description = article.get("description", "No Description")
        message = f"*{title}*\nCategory: {category}\n{description}\n"
        messages.append(message)
        shown_news.add(title)
        
        if len(messages) >= 5:
            break
    
    final_message = "\n".join(messages) if messages else "No new articles available."
    await update.message.reply_text(final_message, parse_mode="Markdown")

# Async command handler for /random
async def random_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    articles = get_news()
    if not articles:
        await update.message.reply_text("Failed to fetch news. Please try again later.")
        return
    
    random.shuffle(articles)
    for article in articles:
        title = article.get("title", "No Title")
        if title in shown_news:
            continue
        
        category = classify_news(title)
        if user_id in user_preferences and category in user_preferences[user_id]:
            continue
        
        description = article.get("description", "No Description")
        message = f"*{title}*\nCategory: {category}\n{description}\n"
        shown_news.add(title)
        await update.message.reply_text(message, parse_mode="Markdown")
        return
    
    await update.message.reply_text("No new articles available.")

# Function to send scheduled news
async def scheduled_news(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id = job.chat_id
    await news(Update(chat_id=chat_id, message=None), context)

# Async command handler for /schedule
async def schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    job_queue = context.job_queue
    if job_queue is None:
        await update.message.reply_text("Error: Job queue is not available.")
        return

    chat_id = update.message.chat_id
    # Schedule at 8:00 AM
    job_queue.run_daily(scheduled_news, time=dt_time(hour=8, minute=0), chat_id=chat_id)
    await update.message.reply_text("Daily news updates scheduled at 8 AM!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Here are the available commands:\n\n"
        "/start - Start the bot and initialize your preferences.\n"
        "/hide <category> - Hide news topics you don't want to see. Available categories: World, Sports, Business, Sci/Tech.\n"
        "/news - Fetch the latest news headlines (ensures no repetition).\n"
        "/random - Get a single random news article.\n"
        "/schedule - Schedule daily news updates at 8:00 AM.\n"
        "/help - Display this help message."
    )
    await update.message.reply_text(help_text)


# Main function to set up and run the bot
def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    job_queue = application.job_queue
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("hide", hide))
    application.add_handler(CommandHandler("news", news))
    application.add_handler(CommandHandler("random", random_news))
    application.add_handler(CommandHandler("schedule", schedule))
    application.add_handler(CommandHandler("help", help_command))
    
    logger.info("Bot started. Listening for commands...")
    application.run_polling()

if __name__ == "__main__":
    main()

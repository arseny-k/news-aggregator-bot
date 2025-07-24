# Telegram News Bot

Telegram News Bot is a Telegram-based news aggregator that fetches the latest headlines from [NewsAPI](https://newsapi.org) and classifies them using a fine-tuned transformer model (based on the AG News dataset). The bot offers features such as user personalization (hiding unwanted topics), random news selection, avoidance of repeated articles, and scheduled daily news updates.

## Features

- **News Aggregation:** Retrieves top headlines from the US using NewsAPI.
- **News Classification:** Uses a fine-tuned transformer model (e.g., DistilBERT) to classify news.
- **User Personalization:** Users can hide specific categories using the `/hide` command.
- **Random News:** Get a single random news article with the `/random` command.
- **Avoid Repetition:** Tracks shown news articles to prevent repeats.
- **Scheduled Updates:** Schedule daily news updates (e.g., at 8:00 AM) via the `/schedule` command.
- **Help Command:** Provides usage instructions with `/help`.


## Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose:** Ensure Docker and Docker Compose are installed on your system.
- **Telegram Bot Token:** Obtain one from [BotFather](https://t.me/BotFather) on Telegram.
- **NewsAPI Key:** Sign up at [NewsAPI.org](https://newsapi.org) for an API key.
- **Pre-trained Model:** The training process (using `train.py`) will produce a `trained_news_model` directory that the bot uses.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/arseny-k/news-aggregator-bot.git
cd news-aggregator-bot
```

### 2. Environment Setup

Create a .env file in the project root and add your credentials:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
NEWS_API_KEY=your_news_api_key
```

### 3. Training the Model

The train.py script fine-tunes a transformer model (e.g., distilbert-base-uncased) on the AG News dataset.

To train the model:

```
python train.py
```

After training, the best model is saved in the `./trained_news_model` directory. Make sure this directory is available for the bot.

### 4. Running with Docker

The project includes a Dockerfile and docker-compose.yml for containerized deployment.
Build and Run Containers

Build the Docker images and start containers:

```
docker-compose up --build
```
The bot container will start, and you can interact with the Telegram bot as usual.

## Bot Commands

- **`/start`**  
  Initializes the bot and provides a welcome message.

- **`/news`**  
  Fetches the latest news headlines while avoiding repetition.

- **`/random`**  
  Retrieves a single random news article.

- **`/hide <category>`**  
  Allows users to hide unwanted news topics.  
  **Available categories:** World, Sports, Business, Sci/Tech.  
  **Example:** `/hide Sports`

- **`/schedule`**  
  Schedules daily news updates at **8:00 AM** by default.

- **`/help`**  
  Displays a list of available commands and their usage.

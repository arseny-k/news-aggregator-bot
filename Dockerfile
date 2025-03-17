FROM rocm/pytorch:latest

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot code (assuming bot.py is in the bot folder)
COPY bot/ bot/

# Set the working directory to the bot folder
WORKDIR /app/bot

# Run the bot
CMD ["python", "bot.py"]

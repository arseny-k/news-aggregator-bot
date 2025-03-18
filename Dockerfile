FROM rocm/pytorch:latest

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot code (assuming bot.py is in the bot folder)
COPY . .


# Run the bot
CMD ["python", "bot.py"]

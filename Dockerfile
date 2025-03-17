# Use an official lightweight Python image.
FROM python:3.9-slim

# Prevent Python from writing pyc files to disk and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container.
WORKDIR /app

# Copy requirements.txt into the container.
COPY requirements.txt /app/

# Upgrade pip and install dependencies.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code into the container.
COPY bot/ /app/bot/

# Specify the command to run your bot.
CMD ["python", "bot/bot.py"]

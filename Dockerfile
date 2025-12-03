# Use official Python 3.11 image

FROM python:3.11-slim

# Set time zone to match your local machine

ENV TZ=America/New_York
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

# Set working directory

WORKDIR /app

# Copy bot code and requirements

COPY . /app

# Install Python packages (add any other dependencies your bot uses)

RUN pip install --no-cache-dir alpaca-trade-api pandas beautifulsoup4 lxml

# Run your bot

CMD ["python", "-u", "alpaca_bot_current.py"]

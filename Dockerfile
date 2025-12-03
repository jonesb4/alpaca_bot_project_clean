# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages (requires your requirements.txt file)
RUN pip install --no-cache-dir -r requirements.txt

# Use the Python interpreter as the direct command entry point
CMD ["python3", "alpaca_bot_current.py"]
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# (You must still create a requirements.txt file)
RUN pip install --no-cache-dir -r requirements.txt

# Make the start script executable
RUN chmod +x start.sh

# Tell the container which command to run when it starts
CMD ["./start.sh"]
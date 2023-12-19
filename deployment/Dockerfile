# Use the official lightweight Python image.
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the contents of the local deployment directory to the container at /app
COPY . ./

# Define environment variable
ENV FLASK_APP=app.py

# Run the web service on container startup using gunicorn.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

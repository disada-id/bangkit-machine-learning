# Gunakan base image Python
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# copy the content of the local src directory to the working directory
COPY deployment/ .

# Run app.py when the container launches
CMD ["python", "./app.py" ]

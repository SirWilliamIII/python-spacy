# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM python:3.10.14

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# This Dockerfile exposes port 5001 to allow external access to the application running inside the container.
EXPOSE 5001

# Install dependencies and download the English language model for spaCy
# Download the English language model for spaCy
RUN pip3 install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm

# Run app.py when the container launches
CMD ["python3", "app.py"]
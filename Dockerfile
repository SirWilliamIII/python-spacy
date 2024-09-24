# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM python:3.10.15

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app


# Install any needed packages specified in requirements.txt
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
# Download the English language model for spaCy
RUN python -m spacy download en_core_web_sm

COPY . .


# Run app.py when the container launches
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
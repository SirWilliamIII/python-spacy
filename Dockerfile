# syntax=docker/dockerfile:1
FROM python:3.10.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt 

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 5001

CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
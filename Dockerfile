# syntax=docker/dockerfile:1
FROM python:3.10.14

WORKDIR /app

COPY . .

EXPOSE 5000

RUN pip3 install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm

CMD ["python3", "app.py"]




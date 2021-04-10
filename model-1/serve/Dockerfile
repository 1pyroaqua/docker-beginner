FROM python:3.6-slim

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY code/ /app/serve/

RUN mkdir -p /app/model

RUN chmod -R +x /app/*

ENV ENVIRONMENT serve

ENTRYPOINT serve/serve.py


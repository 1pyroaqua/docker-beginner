FROM python:3.6-slim

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY code/ /app/train/

RUN mkdir -p /app/model

RUN chmod -R +x /app/*

ENV ENVIRONMENT train

ENTRYPOINT train/train.py


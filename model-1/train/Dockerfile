FROM python:3.6-slim

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

COPY code/train /app/train

RUN chmod +x /app/train/*

ENV ENVIRONMENT train

ENTRYPOINT train/train.py


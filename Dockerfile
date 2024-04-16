# syntax=docker/dockerfile:1

# TODO: Setup better development environment
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:24.02-py3
WORKDIR /home/code
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

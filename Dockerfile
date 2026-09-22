# syntax=docker/dockerfile:1

# TODO: Setup better development environment
# FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:24.02-py3
FROM python:3.12-slim
WORKDIR /home/code
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["chainlit", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"]
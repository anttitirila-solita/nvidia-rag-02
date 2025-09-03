FROM python:3.12-slim

WORKDIR /app

COPY reset.py .

RUN pip install requests pymilvus

CMD ["python", "reset.py"]

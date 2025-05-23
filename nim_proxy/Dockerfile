FROM python:3.11-slim

WORKDIR /app

# Copy the proxy file from the relative src/ path
COPY ./nim_proxy.py /app/nim_proxy.py

# Install dependencies
RUN pip install fastapi uvicorn httpx

EXPOSE 8002

CMD ["uvicorn", "nim_proxy:app", "--host", "0.0.0.0", "--port", "8002"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG AZURE_ML_ENDPOINT
ARG AZURE_ML_API_KEY

ENV AZURE_ML_ENDPOINT=${AZURE_ML_ENDPOINT}
ENV AZURE_ML_API_KEY=${AZURE_ML_API_KEY}

ENV HOST=0.0.0.0 
ENV PORT=8000 
ENV REQUEST_TIMEOUT=30

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
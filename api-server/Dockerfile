FROM python:3.9-alpine

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python train_model.py \
    && useradd --create-home --uid 10001 securepulse \
    && chown -R securepulse:securepulse /app

USER securepulse

CMD ["sh", "-c", "uvicorn modules.api:app --host 127.0.0.1 --port 8000 & exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]

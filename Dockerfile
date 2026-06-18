FROM node:22-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

ENV HOST=0.0.0.0
ENV PORT=7860
ENV OPEN_BROWSER=0
ENV DATABASE_BACKEND=sqlite
ENV SQLITE_DB_PATH=/app/data/chatbot_app.sqlite3
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py README.md ./
COPY backend ./backend
COPY data ./data
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 7860

CMD ["python", "-B", "app.py"]

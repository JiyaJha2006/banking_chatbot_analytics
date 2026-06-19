---
title: Banking Chatbot Analytics
sdk: docker
app_port: 7860
license: mit
---

# Banking Chatbot

React frontend + Python backend API layout:

```text
banking_chatbot/
  app.py                 # Python API server for React
  frontend/              # React UI
    src/
    package.json
  backend/               # Banking/model/database logic
    chat_service.py
    db.py
    ingest.py
    load_mysql_data.py
    mysql_test.py
    sync_embeddings.py
  data/
    BankFAQs.csv
    vector_db/
  requirements.txt
```

## Run the backend API

Copy the environment template once for local development:

```powershell
copy .env.example .env
```

Then edit `.env` if your MySQL username, password, host, or database name is different.

```powershell
cd C:\Users\drjha\banking_chatbot
python app.py
```

The backend runs at `http://127.0.0.1:8000`.

## Run the React frontend

Open a second terminal:

```powershell
cd C:\Users\drjha\banking_chatbot\frontend
npm install
npm run dev
```

Then open the Vite URL, usually `http://127.0.0.1:5173`.

## Rebuild the vector database

```powershell
python backend/ingest.py
```

## MySQL utilities

```powershell
python backend/mysql_test.py
python backend/load_mysql_data.py
python backend/sync_embeddings.py
```

## Deploy on Hugging Face Spaces

This project includes a `Dockerfile` for Hugging Face Spaces. The deployed version uses SQLite for users, tokens, saved questions, and analytics so it does not need paid MySQL hosting.

1. Create a new Space on Hugging Face.
2. Choose **Docker** as the Space SDK.
3. Make the Space public if you want it to count as open source.
4. Upload or push this repository.
5. Hugging Face will build the Docker image and run `app.py` on port `7860`.

Deployment environment variables are already set in the Dockerfile:

```text
HOST=0.0.0.0
PORT=7860
OPEN_BROWSER=0
DATABASE_BACKEND=sqlite
SQLITE_DB_PATH=/app/data/chatbot_app.sqlite3
```

Local development still uses MySQL by default. To test the deployment database locally:

```powershell
$env:DATABASE_BACKEND="sqlite"
$env:SQLITE_DB_PATH="C:\Users\drjha\banking_chatbot\data\chatbot_app.sqlite3"
$env:OPEN_BROWSER="0"
python app.py
```

## Environment variables and secrets

Do not commit real `.env` files. Commit only `.env.example`.

For GitHub Secrets or hosting provider secrets, add either `MYSQL_URL`:

```text
MYSQL_URL=mysql://USERNAME:PASSWORD@HOST:3306/DATABASE_NAME
```

or add the separate values:

```text
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root123
MYSQL_DATABASE=banking_chatbot_db
```

Other supported variables:

```text
HOST=127.0.0.1
PORT=8000
OPEN_BROWSER=1
DATABASE_BACKEND=mysql
SQLITE_DB_PATH=data/chatbot_app.sqlite3
```

---
title: Banking Chatbot Analytics
sdk: docker
app_port: 7860
license: mit
---

# Banking Chatbot Analytics

Banking Chatbot Analytics is an AI-powered banking support web application. It combines a chatbot, authentication system, account-specific chat history, smart banking recommendations, product comparison, multilingual support, voice features, and an FAQ analytics dashboard.

The project is designed as a full-stack AI application with a React frontend, Python backend, vector search, token-based authorization, and database-backed analytics.

## Live Demo

Hugging Face Space:

```text
https://jiya1234567-banking-chatbot-analytics.hf.space
```

Repository:

```text
https://github.com/JiyaJha2006/banking_chatbot_analytics
```

## Features

- User sign up and sign in
- 24-hour authorization token after login
- Account-specific chat history
- User-specific stored questions
- English and Hindi language toggle
- Voice input using browser speech recognition
- Voice output using browser speech synthesis
- Context-aware follow-up handling
- Multi-step banking conversations
- Suggested related questions after answers
- Smart account recommendation agent
- Product comparison tables, such as FD vs RD
- Banking form assistant for documents and application steps
- FAQ analytics dashboard
- MySQL support for local development
- SQLite support for free Hugging Face deployment
- Docker deployment through Hugging Face Spaces
- GitHub Actions CI pipeline for build checks

## Main Modules

```text
banking_chatbot_analytics/
  app.py                         Python backend server and API routing
  Dockerfile                     Hugging Face Docker deployment
  requirements.txt               Python dependencies
  .env.example                   Safe environment variable template
  backend/
    auth_service.py              Login, signup, token auth, profile, analytics
    chat_service.py              Chatbot logic, RAG, recommendations, comparisons
    db.py                        Environment-based MySQL configuration
    ingest.py                    Builds Chroma vector database
    load_mysql_data.py           Loads official markdown KB data into MySQL
    official_kb.py               Parses official markdown knowledge-base files
    sync_embeddings.py           Syncs MySQL knowledge data into Chroma
  frontend/
    src/main.jsx                 React application
    src/styles.css               UI styling
    package.json                 Frontend dependencies and scripts
  data/
    official_kb/                 Official markdown knowledge-base files
    vector_db/                   Generated ChromaDB vector database
  .github/workflows/
    ci.yml                       GitHub Actions CI workflow
```

## Tech Stack

Frontend:

- React
- Vite
- CSS
- Lucide React icons
- Browser Speech Recognition
- Browser Speech Synthesis

Backend:

- Python
- `http.server` / `ThreadingHTTPServer`
- MySQL for local database mode
- SQLite for deployment database mode
- Token-based authorization

AI and Search:

- Sentence Transformers
- Cross Encoder reranking
- Hugging Face Transformers
- FLAN-T5
- ChromaDB vector database
- Official markdown banking knowledge base

Deployment and DevOps:

- Docker
- Hugging Face Spaces
- GitHub
- GitHub Actions CI

## How It Works

1. The user signs up or signs in.
2. The backend creates a 24-hour authorization token.
3. The frontend stores the token and sends it with protected API requests.
4. The user asks a banking question.
5. The backend detects intent and context.
6. The chatbot either:
   - answers from the official banking knowledge base,
   - asks a follow-up question,
   - recommends an account,
   - compares banking products,
   - or gives a document/application checklist.
7. The question, topic, and response time are stored for analytics.
8. The analytics dashboard displays usage statistics and charts.

## Key AI Features

### Context-Aware Conversations

The chatbot remembers the current topic during a session. For example:

```text
User: What is a savings account?
Bot: A savings account is...

User: How do I open it?
Bot: To open a savings account...
```

### Multi-Step Conversations

For questions where more information is needed, the bot asks a follow-up.

Example:

```text
User: Home loan
Bot: Before I list the exact steps for home loan, are you salaried or self-employed?
```

### Smart Account Recommendation Agent

The recommendation agent extracts profile signals from the user message.

Example:

```text
User: I am a college student and mostly use UPI.
Bot: Based on your profile, a Student Savings Account would be suitable.
```

It also returns:

- Recommended account
- Why recommended
- Benefits
- Detected profile signals

### Product Comparison

Example:

```text
User: Compare FD and RD
```

The bot returns a structured comparison table:

| Feature | Fixed Deposit | Recurring Deposit |
| --- | --- | --- |
| Deposit | One-time lump sum | Monthly fixed instalment |
| Interest | Fixed for the chosen tenure | Fixed for the chosen tenure |
| Best for | Parking a fixed amount safely | Building savings every month |

### Banking Form Assistant

For application-style questions, the bot gives practical checklists.

Example:

```text
User: How do I apply for a home loan?
```

The bot returns required documents and basic steps.

## Analytics Dashboard

The analytics dashboard tracks:

- Most asked question
- Most searched topic
- Number of users
- Total questions
- Average response time
- Top questions chart
- Top topics chart
- Daily questions chart

Analytics are calculated from stored user questions and response timings.

## API Endpoints

Authentication:

```text
POST /api/register
POST /api/login
GET  /api/token
POST /api/logout
```

Chat:

```text
POST /api/chat
```

Profile and Analytics:

```text
GET /api/profile?user_id={user_id}
GET /api/analytics
```

Navigation/API discovery:

```text
GET /api/health
GET /api/pages
GET /api/buttons
```

## Environment Variables

Do not commit real `.env` files. Commit only `.env.example`.

Local `.env` example:

```text
HOST=127.0.0.1
PORT=8000
OPEN_BROWSER=1
DATABASE_BACKEND=mysql

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root123
MYSQL_DATABASE=banking_chatbot_db

SQLITE_DB_PATH=data/chatbot_app.sqlite3
```

You can also use one MySQL URL instead of separate values:

```text
MYSQL_URL=mysql://USERNAME:PASSWORD@HOST:3306/DATABASE_NAME
```

For Hugging Face deployment, the Dockerfile uses:

```text
HOST=0.0.0.0
PORT=7860
OPEN_BROWSER=0
DATABASE_BACKEND=sqlite
SQLITE_DB_PATH=/app/data/chatbot_app.sqlite3
```

## Local Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/JiyaJha2006/banking_chatbot_analytics.git
cd banking_chatbot_analytics
```

### 2. Create Environment File

```powershell
copy .env.example .env
```

Edit `.env` if your MySQL username, password, host, or database name is different.

### 3. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### 5. Build the Frontend

```powershell
cd frontend
npm run build
cd ..
```

### 6. Run the App

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

## Development Mode

Run backend:

```powershell
python app.py
```

Run frontend separately:

```powershell
cd frontend
npm run dev
```

The Vite development server usually runs at:

```text
http://127.0.0.1:5173
```

## Database Modes

### MySQL Mode

Used for local development.

```text
DATABASE_BACKEND=mysql
```

### SQLite Mode

Used for free Hugging Face deployment.

```text
DATABASE_BACKEND=sqlite
SQLITE_DB_PATH=data/chatbot_app.sqlite3
```

SQLite stores:

- users
- login tokens
- questions
- analytics data

## Rebuild Vector Database

If the official markdown knowledge base changes, rebuild ChromaDB:

```powershell
python backend/ingest.py
```

## MySQL Utilities

Test MySQL connection:

```powershell
python backend/mysql_test.py
```

Load official knowledge-base data into MySQL:

```powershell
python backend/load_mysql_data.py
```

Sync embeddings:

```powershell
python backend/sync_embeddings.py
```

## Deployment

This project is deployed on Hugging Face Spaces using Docker.

Hugging Face deployment steps:

1. Create a new Hugging Face Space.
2. Choose Docker as the SDK.
3. Make the Space public.
4. Push this repository to the Space.
5. Hugging Face builds the Docker image and runs the app on port `7860`.

The `Dockerfile`:

- builds the React frontend,
- installs Python dependencies,
- copies backend and data files,
- runs `python -B app.py`.

## CI Pipeline

The project includes a GitHub Actions CI workflow.

The CI pipeline:

- checks Python syntax,
- installs frontend dependencies,
- builds the React frontend.

CI file:

```text
.github/workflows/ci.yml
```

Deployment is handled separately by Hugging Face Spaces. When changes are pushed to the Space repository, Hugging Face automatically rebuilds the app.

## Security Notes

- Real `.env` files are ignored by Git.
- Only `.env.example` should be committed.
- Database passwords should be stored in GitHub Secrets or hosting provider secrets.
- Authorization tokens expire after 24 hours.
- Protected API routes require a valid Bearer token.

## License

This project is licensed under the MIT License.

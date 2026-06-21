# 🧠 Mini RAG — AI-Powered E-Learning Platform

An AI-powered multi-tenant educational assistant that uses **Retrieval-Augmented Generation (RAG)** to help students learn from uploaded course materials.

Teachers upload PDF course documents, and students interact with an AI tutor that can answer questions, summarize content, generate quizzes, explain topics simply, create flashcards, and translate between Arabic and English — all grounded strictly in the uploaded materials.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Upload** | Drag & drop PDF upload with validation (20MB limit) |
| 💬 **Q&A Chat** | Ask questions answered from course materials only |
| 📝 **Summarize** | Get summaries of specific pages or topics |
| ❓ **Quiz Generation** | Auto-generate MCQ quizzes for self-assessment |
| 💡 **Explain Simply** | Get simple explanations of complex concepts |
| 📇 **Flashcards** | Create study flashcards from course content |
| 🌍 **Translate** | Translate between Arabic ↔ English |
| 🔒 **Multi-Tenant** | Teacher-isolated data with separate vector indexes |
| 🎯 **Document Targeting**| Filter queries by specific document IDs or page ranges |
| 🛡️ **Hallucination Control**| Relevance threshold filtering and confidence scoring |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (HTML/CSS/JS)             │
│         Dashboard │ Upload │ Chat │ AI Tools         │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP/REST
┌─────────────────────▼───────────────────────────────┐
│                  FastAPI Backend                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Upload   │  │ Prompt   │  │ AI Actions       │   │
│  │ Pipeline │  │ Router   │  │ chat/summarize/  │   │
│  │          │  │          │  │ quiz/explain/    │   │
│  │ PDF→     │  │ Action→  │  │ flashcards/     │   │
│  │ Chunk→   │  │ Template │  │ translate        │   │
│  │ Embed→   │  │          │  │                  │   │
│  │ Store    │  │          │  │                  │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│  ┌───────────────────┐  ┌────────────────────────┐   │
│  │ FAISS Vector Store│  │ OpenAI API             │   │
│  │ (per-teacher)     │  │ Embeddings + Chat      │   │
│  └───────────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### 1. Clone & Setup

```bash
git clone <repo-url>
cd Mini_RAG_For_Graduation

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the Dashboard

Navigate to **http://localhost:8000** in your browser.

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a PDF document |
| `POST` | `/api/chat` | Ask questions (RAG Q&A) |
| `POST` | `/api/summarize` | Summarize content |
| `POST` | `/api/quiz` | Generate quiz |
| `POST` | `/api/explain` | Simple explanation |
| `POST` | `/api/flashcards` | Create flashcards |
| `POST` | `/api/translate` | Translate AR↔EN |
| `GET`  | `/api/documents` | List teacher's documents |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `POST` | `/api/reindex/{id}` | Re-index an existing document |
| `GET`  | `/api/actions` | List supported AI actions |
| `GET`  | `/api/stats` | Global platform metrics |
| `GET`  | `/health` | Health check |

### Example: Upload

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@course.pdf" \
  -F "teacher_id=teacher_1"
```

### Example: Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "teacher_id": "teacher_1",
    "message": "What is machine learning?",
    "from_page": 1,
    "to_page": 10
  }'
```

### Interactive API Docs

Visit **http://localhost:8000/docs** for Swagger UI.

---

## 📁 Project Structure

```
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py             # Environment configuration
│   ├── api/routes.py         # API endpoints
│   ├── ingestion/            # PDF processing pipeline
│   ├── processing/           # Text chunking
│   ├── embeddings/           # OpenAI embedding service
│   ├── vector_store/         # FAISS index management
│   ├── retrieval/            # Semantic search
│   ├── prompts/              # Prompt templates
│   ├── router/               # Action → prompt routing
│   ├── llm/                  # OpenAI chat completion
│   └── utils/                # Database, Arabic utils, logging
├── frontend/
│   ├── index.html            # Dashboard UI
│   ├── css/styles.css        # Design system
│   └── js/app.js             # Frontend logic
├── requirements.txt
├── Dockerfile
├── docker-compose.yml        # Orchestrates FastAPI & Nginx
├── nginx.conf                # Production reverse proxy config
├── .dockerignore             # Optimizes image builds
└── .env.example
```

---

## 🌐 Multilingual Support

- **Arabic + English** fully supported
- Multilingual embeddings (`text-embedding-3-small`)
- Arabic text normalization (Alef, diacritics, Taa Marbuta)
- Cross-language queries (Arabic questions on English content and vice versa)
- AI responds in the same language as the user's question

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `LLM_MODEL` | `gemini-2.5-flash` | Chat completion model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |
| `SIMILARITY_THRESHOLD`| `0.25` | Min cosine similarity to accept a chunk |
| `TEMPERATURE` | `0.3` | LLM generation temperature |
| `MAX_TOKENS` | `2000` | Max LLM token generation |
| `MAX_RETRIES` | `3` | Max retries for external API calls |
| `LOG_LEVEL` | `INFO` | Application log level |
| `MAX_FILE_SIZE_MB` | `20` | Max upload size |

---

## 📄 License

This project is part of a graduation project.

# Fashion AI App

A wardrobe management and outfit recommendation app powered by custom ML models and an LLM-based chatbot.

## Features

- 📸 Upload and catalog wardrobe items with automatic tagging
- 🎨 Get outfit recommendations based on occasion, weather, preferences
- 💬 Chat interface for natural language outfit requests
- 🧠 Learn from user feedback over time

## Project Structure

```
fashion-app/
├── backend/          # FastAPI backend (API, services, database)
├── ml/               # ML pipeline package (ingestion, recommendation, chatbot)
├── frontend/          # Next.js frontend
├── model_artifacts/  # Trained model weights (binary files)
├── data/             # Runtime data and training datasets
├── storage/           # Persistent storage (ChromaDB)
├── scripts/          # Training and data prep scripts
└── docs/             # Documentation
```

See `docs/ARCHITECTURE.md` for detailed folder structure and rationale.

## Tech Stack

- **Frontend:** Next.js + Tailwind CSS
- **Backend:** Python + FastAPI
- **Database:** PostgreSQL + ChromaDB (vectors)
- **ML:** PyTorch, YOLOv8, CLIP, EfficientNet
- **LLM:** OpenAI GPT-4o
- **Queue:** Redis + RQ

## Getting Started

### Prerequisites

- Python 3.10+
- Poetry
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Setup

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd fashion-app
   ```

2. **Install dependencies**
   ```bash
   # Python (backend + ML)
   poetry install
   
   # Frontend
   cd frontend && npm install
   ```

3. **Start infrastructure**
   ```bash
   docker-compose -f infrastructure/docker/docker-compose.yml up -d postgres redis
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run migrations**
   ```bash
   cd backend
   poetry run alembic upgrade head
   ```

6. **Start services**
   ```bash
   # Backend API
   poetry run uvicorn backend.app.main:app --reload --port 8000
   
   # Background worker (in another terminal)
   poetry run rq worker
   
   # Frontend (in another terminal)
   cd frontend && npm run dev
   ```

## Documentation

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) - Project phases and task breakdown
- [`docs/DATABASE.md`](docs/DATABASE.md) - Database schema and architecture
- [`docs/ML_PIPELINE.md`](docs/ML_PIPELINE.md) - ML pipeline modules and implementation
- [`docs/CHATBOT.md`](docs/CHATBOT.md) - Chatbot architecture and flow
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Folder structure rationale

## Development

### Running Tests

```bash
# All tests
poetry run pytest

# Backend tests only
poetry run pytest backend/tests

# ML tests only
poetry run pytest ml/tests
```

### Training Models

```bash
# Download datasets first
poetry run python scripts/data/download_deepfashion2.py

# Train YOLOv8
poetry run python scripts/training/train_yolo.py

# Train classifier
poetry run python scripts/training/train_classifier.py
```

## License

MIT


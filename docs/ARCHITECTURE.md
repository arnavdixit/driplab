# Architecture & Folder Structure Justification

This document explains the rationale behind the folder structure and architectural decisions.

---

## Table of Contents

1. [Monorepo Structure](#monorepo-structure)
2. [Naming Conventions](#naming-conventions)
3. [Separation of Concerns](#separation-of-concerns)
4. [Package Organization](#package-organization)
5. [Data Organization](#data-organization)
6. [Testing Strategy](#testing-strategy)
7. [Deployment Considerations](#deployment-considerations)

---

## Monorepo Structure

### Why Monorepo?

**Decision:** Single repository with `backend/`, `ml/`, and `frontend/` as sibling packages.

**Rationale:**
- **Single source of truth** - All code in one place, easier to coordinate changes
- **Shared dependencies** - Backend and ML share libraries (torch, openai, etc.)
- **Atomic commits** - Can update API and ML logic together
- **Simpler CI/CD** - One pipeline, one deployment story
- **Easier local development** - `poetry install` sets up everything

**Alternative considered:** Separate repos
- ❌ Harder to keep API/ML contracts in sync
- ❌ More complex dependency management
- ❌ Multiple CI/CD pipelines to maintain

**Implementation:**
- Root `pyproject.toml` with Poetry packages for `backend` and `ml`
- Single `poetry install` installs all Python dependencies
- Clean imports: `from ml.ingestion import pipeline`, `from backend.app.models import Garment`

---

## Naming Conventions

### `model_artifacts/` vs `backend/app/models/`

**Problem:** Initial structure had `models/` at root (for trained weights) and `backend/app/models/` (for SQLAlchemy ORM models). This creates ambiguity.

**Solution:** Renamed root to `model_artifacts/`

**Rationale:**
- ✅ **Zero ambiguity** - Clear these are binary files, not Python code
- ✅ **Standard terminology** - "artifacts" is common in ML (model artifacts, training artifacts)
- ✅ **Tooling friendly** - No import conflicts (`import models` could mean either)
- ✅ **Self-documenting** - Name makes purpose obvious

**Alternative considered:** `ml_models/`, `artifacts/models/`
- `ml_models/` - Too generic, doesn't convey "binary files"
- `artifacts/models/` - More nested, but clearer if you have other artifact types

**Chosen:** `model_artifacts/` - Short, clear, standard

---

## Separation of Concerns

### Backend vs ML Package Boundaries

**Decision:** `backend/` handles HTTP, DB, orchestration. `ml/` handles pure ML/LLM logic.

**Rationale:**

| Concern | Location | Why |
|---------|---------|-----|
| HTTP endpoints | `backend/app/api/` | Web framework (FastAPI) lives here |
| Database models | `backend/app/models/` | SQLAlchemy ORM, DB-specific |
| Business logic | `backend/app/services/` | Orchestrates DB + ML calls |
| ML pipelines | `ml/ingestion/` | Pure ML, no DB/HTTP |
| LLM logic | `ml/chatbot/` | Pure LLM, no DB/HTTP |
| Recommendation | `ml/recommendation/` | Pure ML algorithms |

**Key Principle:** `ml/` is **importable library code**, not a service. It takes Python objects in, returns Python objects out.

**Example:**
```python
# backend/app/services/chat_orchestrator.py
from ml.chatbot.constraint_extractor import ConstraintExtractor
from backend.app.models import Conversation

class ChatOrchestrator:
    def process(self, user_id, message):
        # 1. Load from DB (backend concern)
        conv = self.db.get_conversation(user_id)
        
        # 2. Call ML (pure function, no DB)
        constraints = self.extractor.extract(message, context_dict)
        
        # 3. Save to DB (backend concern)
        self.db.save_message(conv.id, message)
```

**Benefits:**
- ✅ **Testable** - ML code can be unit tested with mock data
- ✅ **Reusable** - ML code can be used in scripts, notebooks, other services
- ✅ **Clear contracts** - Easy to see what each package does
- ✅ **No circular dependencies** - ML doesn't import backend

---

## Package Organization

### Backend Structure (Clean Architecture)

```
backend/app/
├── api/          # HTTP layer (thin, delegates to services)
├── services/     # Business logic (orchestrates DB + ML)
├── models/       # SQLAlchemy ORM models
├── schemas/      # Pydantic request/response validation
├── db/           # Database session/config
└── workers/      # Background job workers
```

**Rationale:**
- **Layered architecture** - API → Services → Models
- **Dependency direction** - API depends on Services, Services depend on Models
- **Testability** - Can mock services when testing API, mock DB when testing services

**Why not flat structure?**
- ❌ Harder to see dependencies
- ❌ Easier to create circular imports
- ❌ Less clear what each file's responsibility is

### ML Structure (Domain-Driven)

```
ml/
├── ingestion/        # Image → metadata pipeline
├── recommendation/   # Outfit generation/scoring
├── chatbot/          # LLM/chatbot components
├── vector_store/     # ChromaDB integration
├── utils/            # Shared utilities
└── evaluation/       # Model evaluation
```

**Rationale:**
- **Domain-driven** - Each top-level module is a distinct ML capability
- **Self-contained** - Each module can be understood independently
- **Reusable** - Modules can be imported independently

**Why not organize by "type" (models/, training/, inference/)?**
- ❌ Harder to find all code related to one feature (e.g., chatbot)
- ❌ Mixes concerns (training and inference are different but related)

---

## Data Organization

### `data/runtime/` vs `data/datasets/`

**Problem:** Mixing user-generated runtime data with training datasets creates confusion.

**Solution:** Split into `data/runtime/` and `data/datasets/`

**Rationale:**

| Directory | Purpose | Lifecycle | Backup Strategy |
|-----------|---------|-----------|-----------------|
| `data/runtime/` | User uploads, processed images | Generated at runtime | Must backup (user data) |
| `data/datasets/` | Training datasets | Downloaded once | Can re-download |

**Benefits:**
- ✅ **Clear separation** - Obvious what's user data vs training data
- ✅ **Different storage strategies** - Runtime can go to S3, datasets can stay local
- ✅ **Easier migration** - Can move `runtime/` to cloud without touching datasets
- ✅ **Gitignore clarity** - Both ignored, but for different reasons

**Alternative considered:** `ml/datasets/`
- ✅ Also good - Keeps training data with ML code
- ❌ But datasets are large, might want to symlink to external drive
- **Decision:** Keep at root `data/datasets/` for flexibility

---

## Testing Strategy

### Separate Test Directories

**Structure:**
```
backend/tests/     # Backend tests (integration, API)
ml/tests/          # ML tests (unit, no DB)
```

**Rationale:**
- ✅ **Different test types** - Backend tests need DB, ML tests don't
- ✅ **Different runtimes** - Can run ML tests without starting Postgres
- ✅ **Clear separation** - Easy to see what's tested where
- ✅ **Parallel execution** - Can run `pytest backend/tests` and `pytest ml/tests` separately

**Test Organization:**
```
backend/tests/
├── api/           # Test HTTP endpoints
├── services/      # Test business logic (with DB mocks)
└── conftest.py    # Backend fixtures (DB session, etc.)

ml/tests/
├── test_ingestion_pipeline.py
├── test_chatbot_constraints.py
└── conftest.py    # ML fixtures (mock embeddings, etc.)
```

**Why not `tests/` at root?**
- ❌ Harder to see which tests belong to which package
- ❌ Fixtures become harder to organize
- ❌ Less clear what dependencies each test needs

---

## Deployment Considerations

### Docker Structure

```
infrastructure/docker/
├── Dockerfile.backend    # FastAPI API server
├── Dockerfile.worker    # RQ worker (runs ML processing)
├── Dockerfile.ml-api    # Optional: separate ML inference API
└── Dockerfile.frontend  # Next.js
```

**Rationale:**
- ✅ **Service boundaries** - Each Dockerfile represents a deployable service
- ✅ **Different requirements** - Worker needs GPU, backend doesn't
- ✅ **Independent scaling** - Can scale workers without scaling API
- ✅ **Clear dependencies** - Each service's dependencies are explicit

**Why separate worker from backend?**
- Worker needs ML models loaded (GPU memory)
- Worker needs different resource limits
- Can scale workers independently based on queue depth
- Backend API can be lightweight, stateless

### Storage Structure

```
storage/
└── chroma_db/    # ChromaDB vector store
```

**Rationale:**
- ✅ **Grouped** - All persistent storage in one place
- ✅ **Clear purpose** - Obvious this is for vector database
- ✅ **Extensible** - Can add `storage/redis_dump/`, `storage/cache/` later

**Why not `chroma_db/` at root?**
- ❌ Looks "random" - What is this directory?
- ❌ Harder to see it's storage-related
- ❌ Less organized

---

## Scripts as Thin Wrappers

**Structure:**
```
scripts/
├── data/          # Data download/prep scripts
├── training/      # Training scripts
└── utils/         # Utility scripts
```

**Principle:** Scripts are thin wrappers that call functions from `ml/`

**Example:**
```python
# scripts/training/train_yolo.py
from ml.ingestion.detection import train_yolo_detector

if __name__ == "__main__":
    train_yolo_detector(parse_args())
```

**Rationale:**
- ✅ **Single source of truth** - Training logic lives in `ml/`, not scripts
- ✅ **Reusable** - Can call training from notebooks, tests, other scripts
- ✅ **Testable** - Can unit test training logic without running scripts
- ✅ **No duplication** - Training and inference use same code paths

**Why not put training logic in scripts?**
- ❌ Harder to reuse (can't import from scripts easily)
- ❌ Duplication risk (training vs inference might diverge)
- ❌ Harder to test (scripts are harder to unit test)

---

## Key Architectural Principles

### 1. **Clear Boundaries**
- Backend handles HTTP, DB, orchestration
- ML handles pure algorithms, no side effects
- Frontend handles UI, calls backend API

### 2. **Dependency Direction**
```
Frontend → Backend API → Backend Services → ML Package
                                    ↓
                              Backend Models (DB)
```

- ML never imports backend
- Services orchestrate, don't contain ML logic
- API is thin, delegates to services

### 3. **Testability**
- ML code is pure functions (easy to unit test)
- Backend services can be tested with DB mocks
- API can be tested with service mocks

### 4. **Deployability**
- Each Dockerfile is a deployable service
- Services can scale independently
- Clear resource requirements per service

### 5. **Maintainability**
- Clear naming (no ambiguity)
- Logical grouping (related code together)
- Documented boundaries (docstrings explain contracts)

---

## Evolution Path

This structure supports growth:

### Phase 1: MVP
- Single backend service
- ML code as library
- Simple deployment

### Phase 2: Scale
- Separate worker service (GPU)
- ML code still library, but worker loads models
- Can scale workers independently

### Phase 3: Advanced
- Optional ML API service (for external ML inference)
- ML code remains library, used by multiple services
- Microservices architecture if needed

**Key:** ML code stays as library, not service. Services use it, but it's not deployed separately (unless you want ML API).

---

## Summary

This folder structure prioritizes:

1. ✅ **Clarity** - Clear naming, obvious purpose
2. ✅ **Separation** - Backend vs ML vs Frontend
3. ✅ **Testability** - Easy to test each layer independently
4. ✅ **Scalability** - Can scale services independently
5. ✅ **Maintainability** - Easy to find and modify code

Trade-offs made:
- Monorepo (simpler coordination) vs separate repos (more isolation)
- ML as library (reusable) vs ML as service (more isolation)
- Grouped data (clearer) vs scattered (more flexible)

Overall, this structure balances simplicity for a solo developer with scalability for future growth.


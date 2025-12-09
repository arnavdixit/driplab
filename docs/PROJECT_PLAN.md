# Fashion AI - Project Plan

## Overview

A wardrobe management and outfit recommendation app powered by custom ML models and an LLM-based chatbot. Users upload photos of their clothes, the system analyzes and tags them, and provides personalized outfit recommendations through natural language conversation.

**Project Structure:**
- `backend/` - FastAPI backend (API, services, database models)
- `ml/` - ML pipeline package (ingestion, recommendation, chatbot)
- `frontend/` - Next.js frontend (Cursor builds)
- `model_artifacts/` - Trained model weights (binary files)
- `data/` - Runtime data (`runtime/`) and training datasets (`datasets/`)
- `storage/` - Persistent storage (ChromaDB)
- `scripts/` - Training and data prep scripts
- `infrastructure/` - Docker and deployment configs

See `docs/ARCHITECTURE.md` for detailed folder structure rationale.

**Detailed Task Specs:** See `docs/tasks/` for implementation details:
- `docs/tasks/BACKEND.md` - BE-001 to BE-022
- `docs/tasks/ML.md` - ML-001 to ML-021
- `docs/tasks/FRONTEND.md` - FE-001 to FE-014
- `docs/tasks/INFRA.md` - INF-001 to INF-006

### Key Features
- Upload and catalog wardrobe items with automatic tagging
- Get outfit recommendations based on occasion, weather, preferences
- Chat interface for natural language outfit requests
- Learn from user feedback over time

### Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js + Tailwind CSS (Cursor builds) |
| Backend | Python + FastAPI |
| Database | PostgreSQL + ChromaDB (vectors) |
| ML | PyTorch, YOLOv8, CLIP, EfficientNet |
| LLM | OpenAI GPT-4o |
| Queue | Redis + RQ |
| Storage | Local filesystem (S3 later) |

---

## Project Phases

### Phase 1: Foundation (Week 1-2)
**Goal:** Basic infrastructure, upload flow, and database

| ID | Task | Owner | Priority | Dependencies |
|----|------|-------|----------|--------------|
| BE-001 | FastAPI project setup with Poetry | You | P0 | - |
| BE-002 | PostgreSQL schema + SQLAlchemy models | You | P0 | BE-001 |
| BE-003 | Alembic migrations setup | You | P0 | BE-002 |
| BE-004 | Image upload endpoint | You | P0 | BE-002 |
| BE-005 | Local image storage service | You | P0 | BE-004 |
| BE-006 | Wardrobe CRUD endpoints | You | P0 | BE-002 |
| INF-001 | Docker Compose (Postgres, Redis) | You | P0 | - |
| INF-002 | Environment config (.env) | You | P0 | BE-001 |
| FE-001 | Next.js project setup | Cursor | P0 | - |
| FE-002 | Upload interface (drag-drop) | Cursor | P0 | BE-004 |
| FE-003 | Wardrobe gallery view | Cursor | P0 | BE-006 |

**Milestone:** Can upload garment photos and view them in gallery

---

### Phase 2: ML Pipeline (Week 2-4)
**Goal:** Process uploaded images, extract metadata, generate embeddings

| ID | Task | Owner | Priority | Dependencies |
|----|------|-------|----------|--------------|
| ML-001 | Image preprocessing pipeline | You | P0 | BE-004 |
| ML-002 | Image quality checker | You | P1 | ML-001 |
| ML-003 | YOLOv8 detection integration | You | P0 | ML-001 |
| ML-004 | Download DeepFashion2 dataset | You | P0 | - |
| ML-005 | Fine-tune YOLOv8 on fashion | You | P1 | ML-003, ML-004 |
| ML-006 | EfficientNet classifier setup | You | P0 | ML-003 |
| ML-007 | Train garment classifier | You | P1 | ML-006, ML-004 |
| ML-008 | Color extraction (rule-based) | You | P0 | ML-003 |
| ML-009 | Attribute tagger (pattern, fit) | You | P1 | ML-006 |
| ML-010 | CLIP embedding pipeline | You | P0 | ML-003 |
| ML-011 | ChromaDB integration | You | P0 | ML-010 |
| BE-007 | Background job system (RQ) | You | P0 | INF-001 |
| BE-008 | ML processing worker | You | P0 | BE-007, ML-001 |
| FE-004 | Garment detail view | Cursor | P1 | BE-006 |
| FE-005 | Processing status indicator | Cursor | P1 | BE-008 |

**Milestone:** Uploaded photos auto-processed with category, color, attributes, embeddings

---

### Phase 3: Recommendations (Week 4-5)
**Goal:** Generate and score outfit combinations

| ID | Task | Owner | Priority | Dependencies |
|----|------|-------|----------|--------------|
| ML-012 | Outfit slot logic (top/bottom/etc) | You | P0 | ML-006 |
| ML-013 | Candidate generator | You | P0 | ML-012 |
| ML-014 | Rule-based compatibility scorer | You | P0 | ML-013 |
| ML-015 | Color harmony rules | You | P0 | ML-014 |
| ML-016 | Formality matching logic | You | P0 | ML-014 |
| ML-017 | Download Polyvore dataset | You | P1 | - |
| ML-018 | Train learned compatibility model | You | P2 | ML-017, ML-010 |
| BE-009 | Recommendation endpoint | You | P0 | ML-014 |
| BE-010 | Constraint parsing | You | P0 | BE-009 |
| BE-011 | Feedback endpoint (like/dislike) | You | P1 | BE-009 |
| FE-006 | Outfit recommendation cards | Cursor | P0 | BE-009 |
| FE-007 | Like/dislike buttons | Cursor | P1 | BE-011 |
| FE-008 | Outfit detail modal | Cursor | P1 | FE-006 |

**Milestone:** Can request outfits for occasion, see scored recommendations

---

### Phase 4: Chat Integration (Week 5-6)
**Goal:** Natural language interface for outfit requests

| ID | Task | Owner | Priority | Dependencies |
|----|------|-------|----------|--------------|
| BE-012 | OpenAI client setup | You | P0 | INF-002 |
| BE-013 | Constraint extraction function | You | P0 | BE-012 |
| BE-014 | Context builder | You | P0 | BE-006 |
| BE-015 | Constraint merger | You | P0 | BE-013 |
| BE-016 | Chat orchestrator | You | P0 | BE-013-015, BE-009 |
| BE-017 | Response generator | You | P0 | BE-016 |
| BE-018 | Chat endpoint (streaming) | You | P0 | BE-016 |
| BE-019 | Conversation persistence | You | P1 | BE-018 |
| BE-020 | Session state management | You | P0 | BE-019 |
| FE-009 | Chat interface | Cursor | P0 | BE-018 |
| FE-010 | Streaming response display | Cursor | P0 | FE-009 |
| FE-011 | Outfit cards in chat | Cursor | P0 | FE-009, FE-006 |

**Milestone:** Can chat naturally, get outfit recommendations, refine with follow-ups

---

### Phase 5: Polish & Personalization (Week 6+)
**Goal:** Improve models, add personalization, deploy

| ID | Task | Owner | Priority | Dependencies |
|----|------|-------|----------|--------------|
| ML-019 | Feedback-based re-ranking | You | P1 | BE-011 |
| ML-020 | User preference learning | You | P2 | ML-019 |
| ML-021 | Fine-tune CLIP on fashion | You | P2 | ML-010 |
| BE-021 | User preferences endpoint | You | P1 | BE-002 |
| BE-022 | Weather API integration | You | P2 | BE-016 |
| FE-012 | Preferences/settings page | Cursor | P1 | BE-021 |
| FE-013 | Outfit history view | Cursor | P2 | BE-019 |
| FE-014 | Mobile responsive design | Cursor | P1 | All FE |
| INF-003 | Production deployment | You | P1 | All |
| INF-004 | Model artifact storage | You | P1 | All ML |

**Milestone:** Polished, personalized app ready for real use

---

## Task Quick Reference

### Backend (BE)
```
BE-001  FastAPI setup                    Phase 1  P0
BE-002  PostgreSQL + SQLAlchemy          Phase 1  P0
BE-003  Alembic migrations               Phase 1  P0
BE-004  Image upload endpoint            Phase 1  P0
BE-005  Image storage service            Phase 1  P0
BE-006  Wardrobe CRUD                    Phase 1  P0
BE-007  Background jobs (RQ)             Phase 2  P0
BE-008  ML processing worker             Phase 2  P0
BE-009  Recommendation endpoint          Phase 3  P0
BE-010  Constraint parsing               Phase 3  P0
BE-011  Feedback endpoint                Phase 3  P1
BE-012  OpenAI client                    Phase 4  P0
BE-013  Constraint extraction            Phase 4  P0
BE-014  Context builder                  Phase 4  P0
BE-015  Constraint merger                Phase 4  P0
BE-016  Chat orchestrator                Phase 4  P0
BE-017  Response generator               Phase 4  P0
BE-018  Chat endpoint                    Phase 4  P0
BE-019  Conversation persistence         Phase 4  P1
BE-020  Session state                    Phase 4  P0
BE-021  User preferences endpoint        Phase 5  P1
BE-022  Weather API                      Phase 5  P2
```

### ML Pipeline (ML)
```
ML-001  Image preprocessing              Phase 2  P0
ML-002  Quality checker                  Phase 2  P1
ML-003  YOLOv8 integration               Phase 2  P0
ML-004  DeepFashion2 dataset             Phase 2  P0
ML-005  Fine-tune YOLOv8                 Phase 2  P1
ML-006  EfficientNet setup               Phase 2  P0
ML-007  Train classifier                 Phase 2  P1
ML-008  Color extraction                 Phase 2  P0
ML-009  Attribute tagger                 Phase 2  P1
ML-010  CLIP embeddings                  Phase 2  P0
ML-011  ChromaDB integration             Phase 2  P0
ML-012  Outfit slot logic                Phase 3  P0
ML-013  Candidate generator              Phase 3  P0
ML-014  Rule-based scorer                Phase 3  P0
ML-015  Color harmony rules              Phase 3  P0
ML-016  Formality matching               Phase 3  P0
ML-017  Polyvore dataset                 Phase 3  P1
ML-018  Learned compatibility            Phase 3  P2
ML-019  Feedback re-ranking              Phase 5  P1
ML-020  User preference learning         Phase 5  P2
ML-021  Fine-tune CLIP                   Phase 5  P2
```

### Frontend (FE) - Cursor Builds
```
FE-001  Next.js setup                    Phase 1  P0
FE-002  Upload interface                 Phase 1  P0
FE-003  Wardrobe gallery                 Phase 1  P0
FE-004  Garment detail view              Phase 2  P1
FE-005  Processing status                Phase 2  P1
FE-006  Outfit cards                     Phase 3  P0
FE-007  Like/dislike buttons             Phase 3  P1
FE-008  Outfit detail modal              Phase 3  P1
FE-009  Chat interface                   Phase 4  P0
FE-010  Streaming responses              Phase 4  P0
FE-011  Outfit cards in chat             Phase 4  P0
FE-012  Settings page                    Phase 5  P1
FE-013  Outfit history                   Phase 5  P2
FE-014  Mobile responsive                Phase 5  P1
```

### Infrastructure (INF)
```
INF-001  Docker Compose                  Phase 1  P0
INF-002  Environment config              Phase 1  P0
INF-003  Production deployment           Phase 5  P1
INF-004  Model storage                   Phase 5  P1
```

---

## Success Criteria

### MVP (End of Week 6)
- [ ] Upload 20+ garment photos successfully
- [ ] Auto-categorization accuracy > 75%
- [ ] Generate outfit recommendations for 5+ occasions
- [ ] Chat understands basic constraints (occasion, weather, exclusions)
- [ ] Like/dislike feedback captured
- [ ] Recommendations feel non-random to user

### V1 (Month 2-3)
- [ ] Classification accuracy > 85%
- [ ] Learned compatibility model trained
- [ ] Personalization shows improvement after 20+ feedback signals
- [ ] 3-turn conversations handled correctly
- [ ] Mobile-responsive UI

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| YOLOv8 fine-tuning takes too long | Medium | Medium | Start with pretrained, fine-tune in parallel |
| Classification accuracy too low | Medium | High | Use more training data, ensemble models |
| OpenAI API costs high | Low | Medium | Cache responses, use smaller models for simple tasks |
| RTX 3050 VRAM insufficient | Medium | Medium | Use cloud GPU for training, optimize batch sizes |
| Compatibility rules feel arbitrary | High | Medium | Test extensively, iterate based on feedback |

---

## Getting Started

**Week 1, Day 1:**
1. Run `INF-001`: Set up Docker Compose
2. Run `BE-001`: Initialize FastAPI project
3. Run `FE-001`: Initialize Next.js project (Cursor)

**Week 1, Day 2-3:**
1. Run `BE-002`, `BE-003`: Database schema
2. Run `BE-004`, `BE-005`: Upload endpoint
3. Run `FE-002`: Upload UI (Cursor)

**Week 1, Day 4-5:**
1. Run `BE-006`: Wardrobe CRUD
2. Run `FE-003`: Gallery view (Cursor)
3. Start `ML-004`: Download datasets

**End of Week 1 Goal:** Upload photos, see them in gallery

---

## Recommended Task Order

Execute tasks in this order to respect dependencies and maintain momentum.

### Phase 1: Foundation (Tasks 1-11)

| # | Task | Description | Notes |
|---|------|-------------|-------|
| 1 | INF-001 | Docker Compose setup | Start DB/Redis first |
| 2 | BE-001 | FastAPI project setup | Backend foundation |
| 3 | INF-002 | Environment config | Configure .env |
| 4 | BE-002 | PostgreSQL schema + models | Database models |
| 5 | BE-003 | Alembic migrations | Apply schema |
| 6 | FE-001 | Next.js project setup | Can parallel with 4-5 |
| 7 | BE-004 | Image upload endpoint | Upload API |
| 8 | BE-005 | Local storage service | File handling |
| 9 | FE-002 | Upload interface | Upload UI |
| 10 | BE-006 | Wardrobe CRUD endpoints | Gallery API |
| 11 | FE-003 | Wardrobe gallery view | Gallery UI |

**Checkpoint:** Can upload photos and view in gallery

### Phase 2: ML Pipeline (Tasks 12-26)

| # | Task | Description | Notes |
|---|------|-------------|-------|
| 12 | ML-001 | Image preprocessing | ML foundation |
| 13 | ML-004 | Download DeepFashion2 | Start download early |
| 14 | ML-003 | YOLOv8 detection | Detection pipeline |
| 15 | ML-006 | EfficientNet classifier | Classification setup |
| 16 | ML-008 | Color extraction | Rule-based colors |
| 17 | ML-010 | CLIP embedding pipeline | Embeddings |
| 18 | ML-011 | ChromaDB integration | Vector storage |
| 19 | BE-007 | Background job system | Redis + RQ |
| 20 | BE-008 | ML processing worker | Connect ML to backend |
| 21 | ML-002 | Image quality checker | Optional validation |
| 22 | FE-004 | Garment detail view | Detail UI |
| 23 | FE-005 | Processing status | Status indicators |
| 24 | ML-005 | Fine-tune YOLOv8 | P1 - Can defer |
| 25 | ML-007 | Train classifier | P1 - Can defer |
| 26 | ML-009 | Attribute tagger | P1 - Can defer |

**Checkpoint:** Photos auto-processed with category, color, embeddings

### Phase 3: Recommendations (Tasks 27-39)

| # | Task | Description | Notes |
|---|------|-------------|-------|
| 27 | ML-012 | Outfit slot logic | Define outfit structure |
| 28 | ML-013 | Candidate generator | Generate outfit combos |
| 29 | ML-014 | Rule-based scorer | Compatibility scoring |
| 30 | ML-015 | Color harmony rules | Color matching |
| 31 | ML-016 | Formality matching | Style matching |
| 32 | BE-009 | Recommendation endpoint | Recommendation API |
| 33 | BE-010 | Constraint parsing | Parse filters |
| 34 | FE-006 | Outfit recommendation cards | Outfit UI |
| 35 | BE-011 | Feedback endpoint | Like/dislike API |
| 36 | FE-007 | Like/dislike buttons | Feedback UI |
| 37 | FE-008 | Outfit detail modal | Outfit details |
| 38 | ML-017 | Download Polyvore | P1 - For learned model |
| 39 | ML-018 | Train compatibility model | P2 - Can defer |

**Checkpoint:** Can request and see outfit recommendations

### Phase 4: Chat Integration (Tasks 40-51)

| # | Task | Description | Notes |
|---|------|-------------|-------|
| 40 | BE-012 | OpenAI client setup | LLM foundation |
| 41 | BE-014 | Context builder | Build chat context |
| 42 | BE-013 | Constraint extraction | LLM extracts constraints |
| 43 | BE-015 | Constraint merger | Merge session state |
| 44 | BE-016 | Chat orchestrator | Main chat logic |
| 45 | BE-017 | Response generator | LLM generates response |
| 46 | BE-018 | Chat endpoint | Chat API |
| 47 | BE-020 | Session state management | Track conversation |
| 48 | BE-019 | Conversation persistence | Save to DB |
| 49 | FE-009 | Chat interface | Chat UI |
| 50 | FE-010 | Streaming responses | Stream display |
| 51 | FE-011 | Outfit cards in chat | Inline outfits |

**Checkpoint:** Can chat naturally and get outfit recommendations

### Phase 5: Polish & Deploy (Tasks 52-63)

| # | Task | Description | Notes |
|---|------|-------------|-------|
| 52 | BE-021 | User preferences endpoint | Preferences API |
| 53 | FE-012 | Preferences/settings page | Settings UI |
| 54 | ML-019 | Feedback-based re-ranking | Personalization |
| 55 | FE-014 | Mobile responsive design | Mobile UI |
| 56 | FE-013 | Outfit history view | History UI |
| 57 | INF-003 | Production deployment | Deploy app |
| 58 | INF-004 | Model artifact storage | Model versioning |
| 59 | ML-020 | User preference learning | P2 - Advanced |
| 60 | ML-021 | Fine-tune CLIP | P2 - Advanced |
| 61 | BE-022 | Weather API integration | P2 - Nice to have |

**Checkpoint:** Polished app ready for real use

---

## Quick Start Sequence

Copy-paste friendly list of first 20 tasks:

```
1.  INF-001  Docker Compose
2.  BE-001   FastAPI setup
3.  INF-002  Environment config
4.  BE-002   Database models
5.  BE-003   Alembic migrations
6.  FE-001   Next.js setup
7.  BE-004   Upload endpoint
8.  BE-005   Storage service
9.  FE-002   Upload UI
10. BE-006   Wardrobe CRUD
11. FE-003   Gallery UI
12. ML-001   Image preprocessing
13. ML-004   Download DeepFashion2
14. ML-003   YOLOv8 detection
15. ML-006   EfficientNet setup
16. ML-008   Color extraction
17. ML-010   CLIP embeddings
18. ML-011   ChromaDB integration
19. BE-007   Background jobs
20. BE-008   ML worker
```


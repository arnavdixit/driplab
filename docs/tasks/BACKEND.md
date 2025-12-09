# Backend Tasks (BE-xxx)

All backend tasks for FastAPI, database, services, and API endpoints.

---

## BE-001: FastAPI Project Setup with Poetry

**Phase:** 1 | **Priority:** P0 | **Dependencies:** None

**Description:**
Initialize the FastAPI backend project using Poetry for dependency management. Set up the basic project structure, configuration, and entry point.

**Files to Create:**
- `backend/pyproject.toml` - Poetry config (or use root `pyproject.toml`)
- `backend/app/main.py` - FastAPI app entry point
- `backend/app/config.py` - Settings using Pydantic BaseSettings
- `backend/app/__init__.py` - Package init

**Key Requirements:**
- Use Pydantic v2 for settings management
- Load settings from environment variables and `.env` file
- Set up CORS middleware for frontend communication
- Configure logging
- Add health check endpoint at `GET /health`

**Acceptance Criteria:**
- [ ] `poetry install` succeeds
- [ ] `uvicorn backend.app.main:app --reload` starts server
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] Settings load from `.env` file
- [ ] CORS allows requests from `localhost:3000`

---

## BE-002: PostgreSQL Schema + SQLAlchemy Models

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-001

**Description:**
Create SQLAlchemy ORM models for all database tables as defined in `docs/DATABASE.md`.

**Files to Create:**
- `backend/app/db/base.py` - SQLAlchemy Base class
- `backend/app/db/session.py` - Database session management
- `backend/app/models/user.py` - User model
- `backend/app/models/garment.py` - Garment model
- `backend/app/models/garment_prediction.py` - GarmentPrediction model
- `backend/app/models/garment_label.py` - GarmentLabel model
- `backend/app/models/outfit.py` - Outfit model
- `backend/app/models/outfit_feedback.py` - OutfitFeedback model
- `backend/app/models/user_preferences.py` - UserPreferences model
- `backend/app/models/conversation.py` - Conversation model
- `backend/app/models/message.py` - Message model
- `backend/app/models/__init__.py` - Export all models

**Key Requirements:**
- Use UUID as primary keys
- Use JSONB for flexible attribute storage
- Use ARRAY types for list fields
- Define relationships between models
- Follow schema exactly as in `docs/DATABASE.md`

**Acceptance Criteria:**
- [ ] All 9 models created with correct columns and types
- [ ] Relationships defined (User → Garments, Garment → Predictions, etc.)
- [ ] Models can be imported from `backend.app.models`
- [ ] No circular import issues

---

## BE-003: Alembic Migrations Setup

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-002

**Description:**
Set up Alembic for database migrations and create initial migration from models.

**Files to Create/Modify:**
- `backend/alembic.ini` - Alembic config
- `backend/alembic/env.py` - Migration environment (import models)
- `backend/alembic/versions/001_initial.py` - Initial migration

**Key Requirements:**
- Configure Alembic to read DATABASE_URL from environment
- Import all models in `env.py` so autogenerate works
- Create initial migration with all tables
- Include indexes as defined in schema

**Acceptance Criteria:**
- [ ] `alembic revision --autogenerate` detects all models
- [ ] `alembic upgrade head` creates all tables
- [ ] `alembic downgrade base` drops all tables
- [ ] Tables have correct indexes

---

## BE-004: Image Upload Endpoint

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-002

**Description:**
Create endpoint to upload garment images. Saves file to storage, creates database record, and queues for ML processing.

**Files to Create:**
- `backend/app/api/v1/wardrobe.py` - Wardrobe endpoints
- `backend/app/schemas/garment.py` - Pydantic schemas for request/response

**Endpoint:**
- `POST /api/v1/wardrobe/upload`
- Accept: `multipart/form-data` with image file
- Response: Garment object with `id`, `status: "pending"`, `thumbnail_url`

**Key Requirements:**
- Validate file type (JPEG, PNG, WebP)
- Validate file size (max 10MB)
- Generate unique filename using UUID
- Save original to `data/runtime/uploads/`
- Create thumbnail (optional in MVP, can skip)
- Create `garments` database record with `status: "pending"`
- Return garment ID for frontend to poll status
- Dev convenience (optional): set `ALLOW_DEV_SINGLE_USER=true` to auto-create/use
  `DEV_SINGLE_USER_EMAIL` when no `Authorization` header is provided (dev only).

**Acceptance Criteria:**
- [ ] Can upload JPEG/PNG image
- [ ] Rejects non-image files
- [ ] Rejects files > 10MB
- [ ] Creates database record
- [ ] Saves file to correct location
- [ ] Returns garment ID

---

## BE-005: Local Image Storage Service + Static File Serving

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-004

**Description:**
Create a storage service abstraction for saving/retrieving images AND serve them via HTTP. Supports local filesystem now, can swap to S3 later.

**Files to Create:**
- `backend/app/services/storage.py` - Storage service
- `backend/app/api/v1/media.py` - Static file serving route (alternative: mount in main.py)

**Key Requirements:**
- Abstract interface for storage operations
- Methods: `save(file, path)`, `get_url(path)`, `delete(path)`
- Local implementation saves to `data/runtime/`
- Generate unique paths to avoid collisions
- Support for different directories (uploads, processed, thumbnails)

**IMPORTANT - Static File Serving:**
- Mount `data/runtime/` as static files in FastAPI OR create media endpoint
- Option A: `app.mount("/media", StaticFiles(directory="data/runtime"), name="media")`
- Option B: Create `GET /api/v1/media/{path:path}` endpoint that serves files
- `get_url(path)` must return **web-accessible URLs** (e.g., `/media/uploads/abc.jpg`)
- NOT filesystem paths (e.g., `/data/runtime/uploads/abc.jpg`)

**URL Format:**
- Original: `/media/uploads/{filename}` or `/api/v1/media/uploads/{filename}`
- Processed: `/media/processed/{filename}`
- Thumbnail: `/media/thumbnails/{filename}`

**Acceptance Criteria:**
- [ ] Can save uploaded file
- [ ] Can retrieve **web-accessible** file URL
- [ ] Can delete file
- [ ] Files saved to correct directories
- [ ] No filename collisions
- [ ] `GET /media/uploads/{filename}` returns the image file
- [ ] Frontend can display images using returned URLs

---

## BE-006: Wardrobe CRUD Endpoints

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-002, BE-004, BE-005

**Description:**
Create REST endpoints for wardrobe management: list, get, update, delete garments.

**Files to Modify:**
- `backend/app/api/v1/wardrobe.py` - Add CRUD endpoints
- `backend/app/schemas/garment.py` - Add response schemas

**Endpoints:**
- `GET /api/v1/wardrobe` - List all garments with pagination and filters
- `GET /api/v1/wardrobe/{id}` - Get single garment with predictions
- `PATCH /api/v1/wardrobe/{id}` - Update garment (custom name, notes)
- `DELETE /api/v1/wardrobe/{id}` - Delete garment and associated files

**Key Requirements:**
- Pagination with `limit` and `offset`
- Filter by `status`, `category`
- Include predictions in response when available
- Delete should remove files from storage too
- Return proper error codes (404, 400, etc.)

**IMPORTANT - Image URLs in Response:**
- Response must include **web-accessible URLs** for images
- Use storage service `get_url()` to convert paths to URLs
- Response schema should have: `image_url`, `thumbnail_url`, `processed_image_url`
- These must be URLs the frontend can use directly in `<img src="...">`
- Example: `{"image_url": "/media/uploads/abc-123.jpg", ...}`
- NOT filesystem paths like `"/data/runtime/uploads/abc-123.jpg"`

**Acceptance Criteria:**
- [ ] List returns paginated garments
- [ ] Can filter by status and category
- [ ] Get returns garment with predictions
- [ ] Update modifies only allowed fields
- [ ] Delete removes DB record and files
- [ ] **Image URLs in response are web-accessible**
- [ ] **Frontend can render images using returned URLs**

---

## BE-007: Background Job System (Redis + RQ)

**Phase:** 2 | **Priority:** P0 | **Dependencies:** INF-001

**Description:**
Set up Redis Queue (RQ) for background job processing. ML processing will run as background jobs.

**Files to Create:**
- `backend/app/workers/__init__.py` - Worker package
- `backend/app/workers/tasks.py` - Task definitions
- `backend/app/workers/worker.py` - Worker entry point (optional)

**Key Requirements:**
- Connect to Redis using URL from config
- Define task decorators/functions
- Configure job timeout (ML jobs can take minutes)
- Set up job result storage
- Handle job failures gracefully

**Acceptance Criteria:**
- [ ] Can enqueue jobs to Redis
- [ ] Worker picks up and executes jobs
- [ ] Job results stored and retrievable
- [ ] Failed jobs logged with error info

---

## BE-008: ML Processing Worker

**Phase:** 2 | **Priority:** P0 | **Dependencies:** BE-007, ML-001

**Description:**
Create background worker that processes uploaded images through ML pipeline.

**Files to Create:**
- `backend/app/workers/ml_processor.py` - ML processing task

**Key Requirements:**
- Task: `process_garment(garment_id)`
- Load image from storage
- Call `ml.ingestion.pipeline.process()`
- Save predictions to `garment_predictions` table
- Save embedding to ChromaDB
- Update garment `status` to `"ready"` or `"failed"`
- Handle errors and update status accordingly

**Acceptance Criteria:**
- [ ] Enqueued job processes garment
- [ ] Predictions saved to database
- [ ] Embedding saved to ChromaDB
- [ ] Status updated to "ready" on success
- [ ] Status updated to "failed" with error on failure

---

## BE-009: Recommendation Endpoint

**Phase:** 3 | **Priority:** P0 | **Dependencies:** ML-014

**Description:**
Create endpoint that returns outfit recommendations based on constraints.

**Files to Create:**
- `backend/app/api/v1/outfits.py` - Outfit endpoints
- `backend/app/schemas/outfit.py` - Request/response schemas
- `backend/app/services/recommendation.py` - Recommendation service

**Endpoint:**
- `POST /api/v1/outfits/recommend`
- Body: Constraints object (occasion, weather, exclusions, etc.)
- Response: List of outfit objects with scores and explanations

**Key Requirements:**
- Parse constraints from request body
- Call `ml.recommendation` to generate outfits
- Return top 5 outfits with scores
- Include garment details in response
- Save recommended outfits to database

**Acceptance Criteria:**
- [ ] Returns outfit recommendations
- [ ] Respects constraint filters
- [ ] Outfits include garment images and details
- [ ] Scores and explanations included
- [ ] Outfits saved to database

---

## BE-010: Constraint Parsing

**Phase:** 3 | **Priority:** P0 | **Dependencies:** BE-009

**Description:**
Parse and validate outfit constraints from API request into format used by ML.

**Files to Modify:**
- `backend/app/schemas/outfit.py` - Add constraint schema
- `backend/app/services/recommendation.py` - Add parsing logic

**Constraint Fields:**
- `occasion` - string enum
- `formality_level` - string enum
- `weather` - object with temp and conditions
- `exclude_categories` - array of strings
- `exclude_colors` - array of strings
- `must_include` - array of garment IDs
- `comfort_priority` - float 0-1

**Acceptance Criteria:**
- [ ] Valid constraints parsed correctly
- [ ] Invalid values return 400 error
- [ ] Missing optional fields have defaults
- [ ] Parsed constraints passed to ML

---

## BE-011: Feedback Endpoint

**Phase:** 3 | **Priority:** P1 | **Dependencies:** BE-009

**Description:**
Create endpoint to record user feedback on outfits (like, dislike, wear, skip).

**Files to Create:**
- `backend/app/api/v1/feedback.py` - Feedback endpoints
- `backend/app/schemas/feedback.py` - Request schemas

**Endpoint:**
- `POST /api/v1/feedback`
- Body: `{ outfit_id, action, reason? }`
- Response: `{ success: true }`

**Actions:** `like`, `dislike`, `wear`, `save`, `skip`

**Acceptance Criteria:**
- [ ] Feedback saved to `outfit_feedback` table
- [ ] Invalid outfit_id returns 404
- [ ] Invalid action returns 400
- [ ] Duplicate feedback handled (update or ignore)

---

## BE-012: OpenAI Client Setup

**Phase:** 4 | **Priority:** P0 | **Dependencies:** INF-002

**Description:**
Set up OpenAI client for LLM calls with proper configuration and error handling.

**Files to Create:**
- `backend/app/services/llm.py` - OpenAI client wrapper

**Key Requirements:**
- Load API key from environment
- Configure model (gpt-4o)
- Set up retry logic for rate limits
- Configure timeouts
- Create wrapper functions for common operations

**Acceptance Criteria:**
- [ ] Client initializes with API key
- [ ] Can make completion requests
- [ ] Retries on rate limit
- [ ] Proper error handling

---

## BE-013: Constraint Extraction Function

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-012

**Description:**
Service that uses LLM to extract structured constraints from natural language message.

**Files to Create:**
- `backend/app/services/chat_orchestrator.py` - Start chat orchestrator

**Key Requirements:**
- Call `ml.chatbot.constraint_extractor` with message and context
- Pass wardrobe summary and user preferences as context
- Return structured constraints object
- Handle LLM errors gracefully

**Acceptance Criteria:**
- [ ] Extracts occasion from "I have a job interview"
- [ ] Extracts weather from "it's cold and rainy"
- [ ] Extracts exclusions from "no jeans today"
- [ ] Returns empty constraints for unrelated messages

---

## BE-014: Context Builder

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-006

**Description:**
Build context object from database for chat conversations.

**Files to Modify:**
- `backend/app/services/chat_orchestrator.py` - Add context building

**Context Includes:**
- Conversation history (last N messages)
- Session constraints (from conversation)
- Excluded items (rejected by user)
- Wardrobe summary (counts by category)
- User preferences

**Acceptance Criteria:**
- [ ] Loads conversation history from DB
- [ ] Loads wardrobe summary
- [ ] Loads user preferences
- [ ] Returns structured context object

---

## BE-015: Constraint Merger

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-013

**Description:**
Merge new constraints from message with existing session constraints.

**Files to Modify:**
- `backend/app/services/chat_orchestrator.py` - Add merging logic

**Merge Rules:**
- Single values (occasion, formality): new overrides old
- Arrays (exclusions, colors): merge unique values
- Weather: deep merge object
- Must include: replace entirely

**Acceptance Criteria:**
- [ ] New occasion replaces old
- [ ] Exclusions accumulate
- [ ] Can clear specific constraints
- [ ] Returns merged constraints

---

## BE-016: Chat Orchestrator

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-013, BE-014, BE-015, BE-009

**Description:**
Main orchestrator that coordinates chat flow: context → extract → merge → recommend → respond.

**Files to Modify:**
- `backend/app/services/chat_orchestrator.py` - Complete orchestrator

**Flow:**
1. Build context from database
2. Extract constraints from message (LLM)
3. Merge with session constraints
4. Generate outfit recommendations
5. Generate response (LLM)
6. Save message and update session

**Acceptance Criteria:**
- [ ] Full flow works end-to-end
- [ ] Context passed correctly between steps
- [ ] State persisted to database
- [ ] Errors handled gracefully

---

## BE-017: Response Generator

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-016

**Description:**
Generate natural language response explaining outfit recommendations.

**Files to Modify:**
- `backend/app/services/chat_orchestrator.py` - Add response generation

**Key Requirements:**
- Call `ml.chatbot.response_generator` with outfits and constraints
- Include outfit details in prompt
- Generate conversational, helpful response
- Explain why outfits were recommended

**Acceptance Criteria:**
- [ ] Response mentions specific items
- [ ] Explains why outfit fits request
- [ ] Conversational tone
- [ ] Handles no-results case gracefully

---

## BE-018: Chat Endpoint (Streaming)

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-016

**Description:**
Create chat endpoint with optional streaming response.

**Files to Create:**
- `backend/app/api/v1/chat.py` - Chat endpoints

**Endpoints:**
- `POST /api/v1/chat` - Send message, get response
- `POST /api/v1/chat/stream` - Send message, stream response (SSE)

**Key Requirements:**
- Create or resume conversation
- Call chat orchestrator
- Return response with outfit IDs
- Support Server-Sent Events for streaming

**Acceptance Criteria:**
- [ ] Non-streaming endpoint works
- [ ] Streaming endpoint sends chunks
- [ ] Conversation ID returned/accepted
- [ ] Outfits included in response

---

## BE-019: Conversation Persistence

**Phase:** 4 | **Priority:** P1 | **Dependencies:** BE-018

**Description:**
Persist conversation state and messages to database.

**Files to Modify:**
- `backend/app/services/chat_orchestrator.py` - Add persistence

**Key Requirements:**
- Create conversation on first message
- Save each message (user and assistant)
- Store active constraints on conversation
- Store excluded item IDs on conversation
- Store recommended outfit IDs on assistant messages

**Acceptance Criteria:**
- [ ] Conversations persist across requests
- [ ] Messages saved with timestamps
- [ ] Constraints persist in session
- [ ] Can resume conversation

---

## BE-020: Session State Management

**Phase:** 4 | **Priority:** P0 | **Dependencies:** BE-019

**Description:**
Manage session state for ongoing conversations (constraints, exclusions, shown outfits).

**Files to Modify:**
- `backend/app/services/chat_orchestrator.py` - Add session management

**Session State:**
- `active_constraints` - Current merged constraints
- `excluded_item_ids` - Items user rejected
- `shown_outfit_ids` - Outfits already shown (avoid repeats)

**Acceptance Criteria:**
- [ ] Constraints accumulate across messages
- [ ] Rejected items excluded from future recommendations
- [ ] Shown outfits tracked
- [ ] Session can be reset

---

## BE-021: User Preferences Endpoint

**Phase:** 5 | **Priority:** P1 | **Dependencies:** BE-002

**Description:**
CRUD endpoints for user style preferences.

**Files to Create:**
- `backend/app/api/v1/preferences.py` - Preferences endpoints
- `backend/app/schemas/preferences.py` - Request/response schemas

**Endpoints:**
- `GET /api/v1/preferences` - Get user preferences
- `PUT /api/v1/preferences` - Update preferences

**Fields:**
- `preferred_styles`, `avoid_styles`
- `favorite_colors`, `avoid_colors`
- `preferred_fit`
- `formality_min`, `formality_max`
- `comfort_style_balance`

**Acceptance Criteria:**
- [ ] Can get preferences
- [ ] Can update preferences
- [ ] Validation on enum fields
- [ ] Defaults for missing preferences

---

## BE-022: Weather API Integration

**Phase:** 5 | **Priority:** P2 | **Dependencies:** BE-016

**Description:**
Integrate weather API to auto-populate weather context for recommendations.

**Files to Create:**
- `backend/app/services/weather.py` - Weather service

**Key Requirements:**
- Use free weather API (OpenWeatherMap, WeatherAPI, etc.)
- Get current weather by location
- Map to our weather schema (temp: hot/warm/mild/cool/cold, conditions)
- Cache responses (weather doesn't change frequently)

**Acceptance Criteria:**
- [ ] Fetches current weather
- [ ] Maps to internal schema
- [ ] Caches for 30 minutes
- [ ] Graceful fallback if API fails


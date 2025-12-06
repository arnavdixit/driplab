# Infrastructure Tasks (INF-xxx)

All infrastructure tasks for Docker, deployment, and DevOps.

---

## INF-001: Docker Compose for Local Dev

**Phase:** 1 | **Priority:** P0 | **Dependencies:** None

**Description:**
Create Docker Compose configuration for local development services.

**Files to Create:**
- `infrastructure/docker/docker-compose.yml` - Main compose file
- `infrastructure/docker/docker-compose.dev.yml` - Dev overrides (optional)

**Services:**
- `postgres` - PostgreSQL 15
- `redis` - Redis 7

**Key Requirements:**
- PostgreSQL with persistent volume
- Redis with persistent volume
- Environment variables for credentials
- Health checks for services
- Port mappings for local access

**Configuration:**
- Postgres: port 5432, database `fashion_db`
- Redis: port 6379

**Acceptance Criteria:**
- [ ] `docker-compose up` starts services
- [ ] Postgres accessible at localhost:5432
- [ ] Redis accessible at localhost:6379
- [ ] Data persists across restarts
- [ ] Health checks pass

---

## INF-002: Environment Config Management

**Phase:** 1 | **Priority:** P0 | **Dependencies:** BE-001

**Description:**
Set up environment variable management for all services.

**Files to Create:**
- `.env.example` - Template with all variables
- `backend/app/config.py` - Settings class (if not done in BE-001)

**Environment Variables:**

**Database:**
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string

**Storage:**
- `UPLOAD_DIR` - Path for uploads
- `CHROMA_DB_PATH` - Path for ChromaDB

**ML:**
- `MODELS_DIR` - Path to model artifacts
- `CUDA_VISIBLE_DEVICES` - GPU selection

**API Keys:**
- `OPENAI_API_KEY` - OpenAI API key
- `SECRET_KEY` - App secret key

**Frontend:**
- `NEXT_PUBLIC_API_URL` - Backend URL

**Acceptance Criteria:**
- [ ] `.env.example` documents all variables
- [ ] Settings class loads from environment
- [ ] Missing required variables raise error
- [ ] Defaults for optional variables

---

## INF-003: Production Deployment

**Phase:** 5 | **Priority:** P1 | **Dependencies:** All

**Description:**
Set up production deployment (Railway, Render, or self-hosted).

**Files to Create:**
- `infrastructure/docker/Dockerfile.backend` - Backend Dockerfile
- `infrastructure/docker/Dockerfile.worker` - Worker Dockerfile
- `infrastructure/docker/Dockerfile.frontend` - Frontend Dockerfile
- `infrastructure/nginx/nginx.conf` - Nginx config (if self-hosted)

**Deployment Options:**
1. **Railway/Render** (Recommended for simplicity)
   - Managed PostgreSQL
   - Managed Redis
   - Container deployment

2. **Self-hosted VPS**
   - Docker Compose production
   - Nginx reverse proxy
   - SSL with Let's Encrypt

**Key Requirements:**
- Separate containers for backend, worker, frontend
- Worker container has GPU support (for ML)
- Environment-specific configs
- Health check endpoints
- Logging to stdout

**Acceptance Criteria:**
- [ ] Dockerfiles build successfully
- [ ] Containers run in production mode
- [ ] Environment variables injected
- [ ] Health checks work
- [ ] Logs visible

---

## INF-004: Model Artifact Storage

**Phase:** 5 | **Priority:** P1 | **Dependencies:** All ML

**Description:**
Set up storage and versioning for trained model artifacts.

**Options:**
1. **Git LFS** - Store in repo with large file support
2. **S3/MinIO** - Object storage with versioning
3. **DVC** - Data Version Control
4. **HuggingFace Hub** - Model hosting

**Files to Create:**
- `model_artifacts/.gitattributes` - Git LFS config (if using)
- `scripts/utils/upload_models.py` - Upload script (if using S3)
- `scripts/utils/download_models.py` - Download script

**Key Requirements:**
- Version tracking for models
- Easy download during deployment
- Don't bloat git repo with binaries
- Team can access trained models

**Acceptance Criteria:**
- [ ] Models stored outside main git history
- [ ] Can download specific model version
- [ ] CI/CD can access models
- [ ] Documentation for model management

---

## INF-005: CI Pipeline (GitHub Actions)

**Phase:** 5 | **Priority:** P2 | **Dependencies:** All

**Description:**
Set up CI pipeline for automated testing and linting.

**Files to Create:**
- `.github/workflows/backend-tests.yml` - Backend CI
- `.github/workflows/frontend-tests.yml` - Frontend CI
- `.github/workflows/ml-tests.yml` - ML CI

**Pipeline Steps:**

**Backend:**
1. Install dependencies
2. Run linter (ruff, black)
3. Run type checker (mypy)
4. Run tests (pytest)

**Frontend:**
1. Install dependencies
2. Run linter (eslint)
3. Run type checker (tsc)
4. Run tests (jest/vitest)
5. Build check

**ML:**
1. Install dependencies
2. Run linter
3. Run tests (pytest)

**Acceptance Criteria:**
- [ ] Pipelines trigger on PR
- [ ] Tests run automatically
- [ ] Lint errors block merge
- [ ] Status checks visible

---

## INF-006: Deployment Pipeline

**Phase:** 5 | **Priority:** P2 | **Dependencies:** INF-003, INF-005

**Description:**
Set up CD pipeline for automated deployment.

**Files to Create:**
- `.github/workflows/deploy.yml` - Deployment workflow

**Pipeline Steps:**
1. Run tests (from CI)
2. Build Docker images
3. Push to container registry
4. Deploy to production
5. Run health checks
6. Notify on success/failure

**Deployment Triggers:**
- Push to `main` branch
- Manual trigger with environment selection

**Acceptance Criteria:**
- [ ] Deployment automated on merge to main
- [ ] Can manually trigger deployment
- [ ] Rollback capability
- [ ] Notifications on deploy status


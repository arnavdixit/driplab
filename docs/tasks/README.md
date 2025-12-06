# Task Specifications

Detailed task specifications for AI agents and developers.

## Task Files

| File | Tasks | Description |
|------|-------|-------------|
| [BACKEND.md](BACKEND.md) | BE-001 to BE-022 | FastAPI, database, services, API endpoints |
| [ML.md](ML.md) | ML-001 to ML-021 | Image processing, model training, recommendation |
| [FRONTEND.md](FRONTEND.md) | FE-001 to FE-014 | Next.js UI components and pages |
| [INFRA.md](INFRA.md) | INF-001 to INF-006 | Docker, deployment, CI/CD |

## Task Format

Each task includes:

- **Task ID** - Unique identifier (e.g., BE-001)
- **Phase** - Project phase (1-5)
- **Priority** - P0 (must have), P1 (should have), P2 (nice to have)
- **Dependencies** - Tasks that must be completed first
- **Description** - What needs to be done
- **Files to Create/Modify** - Specific file paths
- **Key Requirements** - Important implementation details
- **Acceptance Criteria** - Definition of done (checkboxes)

## How to Use

### For AI Agents

1. Read the task specification completely
2. Check dependencies are completed
3. Implement following the requirements
4. Verify all acceptance criteria pass
5. Reference related docs (DATABASE.md, ML_PIPELINE.md, etc.)

### For Developers

1. Pick a task from current phase
2. Check dependencies
3. Create branch: `feature/{task-id}-short-description`
4. Implement and test
5. Check all acceptance criteria
6. Submit PR

## Task Status Tracking

Use PROJECT_PLAN.md for high-level status. Mark tasks complete there when done.

## Quick Reference

### Phase 1: Foundation (Week 1-2)
- BE-001 to BE-006 (Backend setup, upload, CRUD)
- FE-001 to FE-003 (Frontend setup, upload, gallery)
- INF-001 to INF-002 (Docker, config)

### Phase 2: ML Pipeline (Week 2-4)
- ML-001 to ML-011 (Image processing, embeddings)
- BE-007 to BE-008 (Background jobs)
- FE-004 to FE-005 (Detail view, status)

### Phase 3: Recommendations (Week 4-5)
- ML-012 to ML-018 (Outfit generation, scoring)
- BE-009 to BE-011 (Recommendation API, feedback)
- FE-006 to FE-008 (Outfit UI)

### Phase 4: Chat (Week 5-6)
- BE-012 to BE-020 (Chat orchestration)
- FE-009 to FE-011 (Chat UI)

### Phase 5: Polish (Week 6+)
- ML-019 to ML-021 (Personalization)
- BE-021 to BE-022 (Preferences, weather)
- FE-012 to FE-014 (Settings, history, mobile)
- INF-003 to INF-006 (Deployment, CI/CD)

## Starting Points

**Good first tasks:**
- BE-001: FastAPI project setup
- FE-001: Next.js project setup
- INF-001: Docker Compose setup
- ML-001: Image preprocessing pipeline

These have no dependencies and establish the project foundation.


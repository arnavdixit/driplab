# Git Workflow Guide

This guide explains how to use feature branches for development in this project.

## Branch Strategy

- **`main`** - Always deployable, stable code
- **`feature/*`** - New features (temporary, merged when done)
- **`fix/*`** - Bug fixes
- **`refactor/*`** - Code improvements

## Quick Start

### Starting a New Feature

```bash
# 1. Make sure you're on main and it's up to date
git checkout main
git pull origin main

# 2. Create and switch to a new feature branch
git checkout -b feature/your-feature-name

# 3. Work on your feature, make commits
git add .
git commit -m "Add image upload endpoint"

# 4. Push to GitHub
git push origin feature/your-feature-name
```

### Finishing a Feature

```bash
# 1. Make sure all changes are committed
git status

# 2. Push final changes
git push origin feature/your-feature-name

# 3. Switch back to main
git checkout main

# 4. Pull latest changes (if working with others)
git pull origin main

# 5. Merge your feature branch
git merge feature/your-feature-name

# 6. Push to GitHub
git push origin main

# 7. Delete the feature branch (cleanup)
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## Feature Branch List

Based on the project plan, here are the expected feature branches:

### Phase 1: Foundation
- `feature/infrastructure-setup` - Docker, env config, FastAPI setup
- `feature/database-setup` - PostgreSQL schema, SQLAlchemy models, migrations
- `feature/image-upload` - Upload endpoint, storage service, upload UI
- `feature/wardrobe-crud` - CRUD endpoints, gallery view, detail view

### Phase 2: ML Pipeline
- `feature/ml-preprocessing` - Image preprocessing, quality checker, YOLOv8
- `feature/ml-classification` - EfficientNet, classifier training, attribute tagger
- `feature/ml-embeddings` - CLIP embeddings, ChromaDB, color extraction
- `feature/background-jobs` - RQ jobs, ML worker, processing status UI

### Phase 3: Recommendations
- `feature/outfit-generation` - Slot logic, candidate generator, compatibility scorer
- `feature/outfit-scoring` - Color harmony, formality matching, recommendation API
- `feature/outfit-feedback` - Feedback endpoint, outfit cards, like/dislike UI

### Phase 4: Chat Integration
- `feature/chat-backend` - OpenAI client, constraint extraction, chat orchestrator, API
- `feature/chat-frontend` - Chat interface, streaming display, outfit cards in chat
- `feature/chat-persistence` - Conversation storage, session management

### Phase 5: Polish
- `feature/personalization` - User preferences, re-ranking, settings page
- `feature/weather-integration` - Weather API, weather-based recommendations
- `feature/mobile-responsive` - Mobile design, UI improvements
- `feature/production-deployment` - Deployment setup, model storage

## Branch Naming Convention

Use lowercase with hyphens:
- ✅ `feature/image-upload`
- ✅ `feature/chat-backend`
- ✅ `fix/image-processing-error`
- ✅ `refactor/ml-pipeline`
- ❌ `Feature/ImageUpload` (wrong)
- ❌ `feature_image_upload` (wrong)

## Commit Message Guidelines

Write clear, descriptive commit messages:

```bash
# Good examples
git commit -m "Add image upload endpoint with file validation"
git commit -m "Implement YOLOv8 garment detection pipeline"
git commit -m "Fix color extraction for dark backgrounds"

# Bad examples
git commit -m "fixes"
git commit -m "update"
git commit -m "WIP"
```

For larger features, use multi-line commits:

```bash
git commit -m "Add wardrobe gallery view

- Implement garment grid layout
- Add filtering by category
- Add search functionality
- Style with Tailwind CSS"
```

## Common Workflows

### Working on Multiple Features

If you need to switch between features:

```bash
# Save current work (even if not committed)
git stash

# Switch to another branch
git checkout feature/other-feature

# Work on it...

# Switch back
git checkout feature/original-feature

# Restore your work
git stash pop
```

### Updating Your Feature Branch

If `main` has new changes while you're working:

```bash
# On your feature branch
git checkout feature/your-feature

# Get latest main
git fetch origin main

# Merge main into your feature branch
git merge origin/main

# Resolve any conflicts, then continue working
```

### Undoing Changes

```bash
# Undo uncommitted changes to a file
git checkout -- path/to/file

# Undo all uncommitted changes
git reset --hard HEAD

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

## Best Practices

1. **Keep branches small** - One feature per branch
2. **Commit often** - Small, logical commits are better
3. **Update from main regularly** - Merge main into your branch weekly
4. **Delete merged branches** - Clean up after merging
5. **Write good commit messages** - Future you will thank you
6. **Test before merging** - Make sure your feature works
7. **One feature at a time** - Focus on completing one before starting another

## Troubleshooting

### "Your branch is behind origin/main"

```bash
git fetch origin
git merge origin/main
# Resolve conflicts if any
```

### "Merge conflicts"

```bash
# Git will mark conflicts in files
# Edit files to resolve conflicts
# Look for <<<<<<< HEAD markers
# Keep the code you want, remove markers
git add .
git commit -m "Resolve merge conflicts"
```

### "Accidentally committed to main"

```bash
# Create a branch from current state
git branch feature/fix-this

# Reset main to previous commit
git reset --hard HEAD~1

# Switch to your branch
git checkout feature/fix-this
```

## Quick Reference

```bash
# Check current branch
git branch

# See all branches (local and remote)
git branch -a

# See what's changed
git status

# See commit history
git log --oneline

# See differences
git diff

# See what files changed in a commit
git show <commit-hash>
```

## Example: Complete Feature Workflow

```bash
# 1. Start fresh
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/image-upload

# 3. Make changes, commit
git add backend/app/api/v1/upload.py
git commit -m "Add image upload endpoint"

git add frontend/app/upload/page.tsx
git commit -m "Add drag-drop upload UI"

# 4. Push to GitHub
git push origin feature/image-upload

# 5. Test everything works...

# 6. Merge to main
git checkout main
git pull origin main
git merge feature/image-upload
git push origin main

# 7. Cleanup
git branch -d feature/image-upload
git push origin --delete feature/image-upload
```

---

**Remember:** Keep it simple. One feature, one branch, merge when done.

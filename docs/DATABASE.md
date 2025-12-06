# Database Architecture

**Relevant Code Paths:**
- SQLAlchemy models: `backend/app/models/`
- Database session/config: `backend/app/db/`
- Migrations: `backend/alembic/`
- Vector store client: `ml/vector_store/client.py`
- ChromaDB storage: `storage/chroma_db/`

## Overview

Two database systems working together:

| Database | Purpose | Data |
|----------|---------|------|
| **PostgreSQL** | Relational data, queries, transactions | Users, garments, outfits, feedback, preferences |
| **ChromaDB** | Vector similarity search | Garment embeddings (512-dim CLIP vectors) |

**Why two databases?**
- PostgreSQL: "Find all blue shirts" (attribute filtering)
- ChromaDB: "Find items similar to this one" (semantic similarity)

---

## Entity Relationship Diagram

```
┌─────────────┐       ┌──────────────────┐       ┌─────────────────┐
│   users     │       │    garments      │       │ garment_preds   │
├─────────────┤       ├──────────────────┤       ├─────────────────┤
│ id (PK)     │──┐    │ id (PK)          │──┐    │ id (PK)         │
│ email       │  │    │ user_id (FK)     │  │    │ garment_id (FK) │
│ created_at  │  └───▶│ original_image   │  └───▶│ category        │
└─────────────┘       │ processed_image  │       │ confidence      │
                      │ thumbnail        │       │ attributes      │
                      │ status           │       │ embedding_id    │
                      │ created_at       │       └─────────────────┘
                      └──────────────────┘
                              │
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│garment_labels │     │   outfits     │     │ ChromaDB      │
├───────────────┤     ├───────────────┤     ├───────────────┤
│ id (PK)       │     │ id (PK)       │     │ garments      │
│ garment_id    │     │ user_id (FK)  │     │ collection    │
│ category      │     │ garment_ids[] │     │ (embeddings)  │
│ attributes    │     │ occasion      │     └───────────────┘
│ source        │     │ context       │
└───────────────┘     │ compat_score  │
                      │ source        │
                      └───────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │outfit_feedback│
                      ├───────────────┤
                      │ id (PK)       │
                      │ outfit_id(FK) │
                      │ user_id (FK)  │
                      │ action        │
                      │ reason        │
                      └───────────────┘

┌─────────────────┐     ┌───────────────┐     ┌───────────────┐
│user_preferences │     │ conversations │     │   messages    │
├─────────────────┤     ├───────────────┤     ├───────────────┤
│ id (PK)         │     │ id (PK)       │──┐  │ id (PK)       │
│ user_id (FK)    │     │ user_id (FK)  │  │  │ conv_id (FK)  │
│ styles[]        │     │ constraints   │  └─▶│ role          │
│ colors[]        │     │ created_at    │     │ content       │
│ fit_pref        │     └───────────────┘     │ constraints   │
│ formality_range │                           │ outfit_ids[]  │
└─────────────────┘                           └───────────────┘
```

---

## PostgreSQL Schema

### Core Tables

```sql
-- =============================================
-- USERS
-- =============================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),  -- NULL for single-user MVP
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- GARMENTS (Uploaded clothing items)
-- =============================================
CREATE TABLE garments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Images
    original_image_path VARCHAR(500) NOT NULL,
    processed_image_path VARCHAR(500),
    thumbnail_path VARCHAR(500),
    
    -- Processing
    status VARCHAR(20) DEFAULT 'pending',
    -- Values: pending, processing, ready, failed
    error_message TEXT,
    
    -- User overrides (optional manual naming)
    custom_name VARCHAR(100),
    custom_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_garments_user ON garments(user_id);
CREATE INDEX idx_garments_status ON garments(status);

-- =============================================
-- GARMENT PREDICTIONS (ML-generated metadata)
-- =============================================
CREATE TABLE garment_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    garment_id UUID NOT NULL REFERENCES garments(id) ON DELETE CASCADE,
    
    -- Detection
    bbox_x INT,
    bbox_y INT,
    bbox_width INT,
    bbox_height INT,
    detection_confidence FLOAT,
    
    -- Classification
    category VARCHAR(50) NOT NULL,
    -- Values: t-shirt, shirt, blouse, sweater, hoodie, jacket, coat,
    --         blazer, jeans, pants, shorts, skirt, dress, sneakers,
    --         shoes, boots, sandals, belt, tie, hat, bag
    category_confidence FLOAT,
    
    subcategory VARCHAR(50),
    -- Values: polo, henley, oxford, flannel, cardigan, bomber, etc.
    
    -- Attributes (JSONB for flexibility)
    attributes JSONB NOT NULL DEFAULT '{}',
    /*
    Example attributes:
    {
        "color_primary": "navy",
        "color_secondary": "white",
        "pattern": "solid",           -- solid, striped, plaid, floral, graphic
        "fit": "slim",                -- slim, regular, relaxed, oversized
        "formality": 0.7,             -- 0.0 (very casual) to 1.0 (very formal)
        "seasons": ["fall", "winter"],
        "style_tags": ["preppy", "classic"],
        "material": "cotton"
    }
    */
    
    -- Reference to ChromaDB embedding
    embedding_id VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_predictions_garment ON garment_predictions(garment_id);
CREATE INDEX idx_predictions_category ON garment_predictions(category);
CREATE INDEX idx_predictions_attributes ON garment_predictions USING GIN (attributes);

-- =============================================
-- GARMENT LABELS (User corrections)
-- =============================================
CREATE TABLE garment_labels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    garment_id UUID NOT NULL REFERENCES garments(id) ON DELETE CASCADE,
    
    category VARCHAR(50),
    subcategory VARCHAR(50),
    attributes JSONB,
    
    source VARCHAR(20) NOT NULL,
    -- Values: user_created, user_corrected
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_labels_garment ON garment_labels(garment_id);

-- =============================================
-- OUTFITS (Combinations of garments)
-- =============================================
CREATE TABLE outfits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Garments in this outfit (array of UUIDs)
    garment_ids UUID[] NOT NULL,
    
    -- Context
    occasion VARCHAR(50),
    -- Values: casual, work, date, party, interview, wedding, workout, etc.
    
    context JSONB,
    /*
    {
        "weather": {"temp": "cool", "conditions": ["rainy"]},
        "formality": "smart_casual",
        "constraints_applied": {...}
    }
    */
    
    -- Scoring
    compatibility_score FLOAT,
    score_breakdown JSONB,
    explanation TEXT,
    
    -- Source
    source VARCHAR(20) NOT NULL,
    -- Values: recommended, user_created, worn
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_outfits_user ON outfits(user_id);
CREATE INDEX idx_outfits_occasion ON outfits(occasion);

-- =============================================
-- OUTFIT FEEDBACK
-- =============================================
CREATE TABLE outfit_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    outfit_id UUID NOT NULL REFERENCES outfits(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    action VARCHAR(20) NOT NULL,
    -- Values: like, dislike, wear, save, skip
    
    reason VARCHAR(50),
    -- Values: colors, style, comfort, occasion_match, other
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feedback_outfit ON outfit_feedback(outfit_id);
CREATE INDEX idx_feedback_user ON outfit_feedback(user_id);
CREATE INDEX idx_feedback_action ON outfit_feedback(action);

-- =============================================
-- USER PREFERENCES
-- =============================================
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    
    -- Style preferences
    preferred_styles VARCHAR(50)[] DEFAULT '{}',
    -- Values: casual, preppy, minimalist, streetwear, classic, bohemian, etc.
    
    avoid_styles VARCHAR(50)[] DEFAULT '{}',
    
    -- Colors
    favorite_colors VARCHAR(30)[] DEFAULT '{}',
    avoid_colors VARCHAR(30)[] DEFAULT '{}',
    
    -- Fit
    preferred_fit VARCHAR(20) DEFAULT 'regular',
    -- Values: slim, regular, relaxed
    
    -- Formality range (0-1)
    formality_min FLOAT DEFAULT 0.0,
    formality_max FLOAT DEFAULT 1.0,
    
    -- Comfort vs style (0=comfort, 1=style)
    comfort_style_balance FLOAT DEFAULT 0.5,
    
    -- Learned weights (updated by ML)
    learned_weights JSONB DEFAULT '{}',
    
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =============================================
-- CONVERSATIONS
-- =============================================
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Current session constraints
    active_constraints JSONB DEFAULT '{}',
    
    -- Excluded items for this session
    excluded_item_ids UUID[] DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_conversations_user ON conversations(user_id);

-- =============================================
-- MESSAGES
-- =============================================
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    
    role VARCHAR(10) NOT NULL,
    -- Values: user, assistant
    
    content TEXT NOT NULL,
    
    -- For assistant messages: which outfits were recommended
    recommended_outfit_ids UUID[],
    
    -- Constraints extracted from this message
    extracted_constraints JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_created ON messages(created_at);
```

---

## ChromaDB Schema

### Garments Collection

```python
# ml/vector_store/client.py
chroma_client = chromadb.PersistentClient(path="./storage/chroma_db")

garments_collection = chroma_client.get_or_create_collection(
    name="garments",
    metadata={"description": "Garment CLIP embeddings"}
)

# Adding a garment embedding
garments_collection.add(
    ids=[str(garment_id)],              # Same as PostgreSQL garment.id
    embeddings=[embedding.tolist()],    # 512-dim CLIP vector
    metadatas=[{
        "user_id": str(user_id),
        "category": "shirt",
        "color_primary": "navy",
        "formality": 0.6
    }]
)

# Query: Find similar garments
results = garments_collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where={
        "user_id": str(user_id),
        "category": "shirt"
    }
)
```

### Outfits Collection (Optional - V1+)

```python
outfits_collection = chroma_client.get_or_create_collection(
    name="outfits",
    metadata={"description": "Outfit aggregate embeddings"}
)

# Outfit embedding = average of garment embeddings
outfit_embedding = np.mean([g.embedding for g in garments], axis=0)

outfits_collection.add(
    ids=[str(outfit_id)],
    embeddings=[outfit_embedding.tolist()],
    metadatas=[{
        "user_id": str(user_id),
        "occasion": "work"
    }]
)
```

---

## Data Flow

### Upload Flow

```
User uploads image
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. API receives image                   │
│    - Validate format/size               │
│    - Save to filesystem                 │
│    - Create garments row (status=pending│
│    - Enqueue processing job             │
└─────────────────────────────────────────┘
        │
        ▼ (Background worker)
┌─────────────────────────────────────────┐
│ 2. ML Pipeline processes image          │
│    - Quality check                      │
│    - Detect garment (YOLOv8)            │
│    - Classify category (EfficientNet)   │
│    - Extract attributes                 │
│    - Generate CLIP embedding            │
└─────────────────────────────────────────┘
        │
        ├──▶ PostgreSQL: garment_predictions row
        │
        └──▶ ChromaDB: embedding vector
        │
        ▼
┌─────────────────────────────────────────┐
│ 3. Update garments.status = 'ready'     │
└─────────────────────────────────────────┘
```

### Recommendation Flow

```
User requests outfit
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. Parse constraints from request       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ 2. Query PostgreSQL for user's wardrobe │
│    - Filter by constraints              │
│    - Join with predictions              │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ 3. Generate outfit combinations         │
│    - Slot-based (top + bottom + shoes)  │
│    - Smart sampling                     │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ 4. Score each combination               │
│    - Compatibility model                │
│    - ChromaDB similarity (optional)     │
│    - Personalization weights            │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ 5. Return top K outfits                 │
│    - Save to outfits table              │
└─────────────────────────────────────────┘
```

---

## Queries

### Get User's Wardrobe with Predictions

```sql
SELECT 
    g.id,
    g.thumbnail_path,
    g.status,
    p.category,
    p.attributes->>'color_primary' as color,
    p.attributes->>'formality' as formality,
    COALESCE(l.category, p.category) as final_category,
    COALESCE(l.attributes, p.attributes) as final_attributes
FROM garments g
LEFT JOIN garment_predictions p ON g.id = p.garment_id
LEFT JOIN garment_labels l ON g.id = l.garment_id
WHERE g.user_id = :user_id
  AND g.status = 'ready'
ORDER BY g.created_at DESC;
```

### Filter Wardrobe by Constraints

```sql
SELECT g.*, p.*
FROM garments g
JOIN garment_predictions p ON g.id = p.garment_id
WHERE g.user_id = :user_id
  AND g.status = 'ready'
  AND p.category NOT IN :excluded_categories
  AND p.attributes->>'color_primary' NOT IN :excluded_colors
  AND (p.attributes->>'formality')::float BETWEEN :formality_min AND :formality_max;
```

### Get User's Feedback History

```sql
SELECT 
    o.id as outfit_id,
    o.garment_ids,
    f.action,
    f.reason,
    f.created_at
FROM outfit_feedback f
JOIN outfits o ON f.outfit_id = o.id
WHERE f.user_id = :user_id
ORDER BY f.created_at DESC
LIMIT 100;
```

### Aggregate Feedback for Personalization

```sql
SELECT 
    p.category,
    p.attributes->>'color_primary' as color,
    COUNT(*) FILTER (WHERE f.action = 'like') as likes,
    COUNT(*) FILTER (WHERE f.action = 'dislike') as dislikes,
    COUNT(*) FILTER (WHERE f.action = 'wear') as wears
FROM outfit_feedback f
JOIN outfits o ON f.outfit_id = o.id
CROSS JOIN LATERAL unnest(o.garment_ids) as gid
JOIN garment_predictions p ON p.garment_id = gid
WHERE f.user_id = :user_id
GROUP BY p.category, p.attributes->>'color_primary';
```

---

## SQLAlchemy Models

```python
# backend/app/models/user.py, backend/app/models/garment.py, etc.
from sqlalchemy import Column, String, Float, ForeignKey, ARRAY, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from backend.app.db.base import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    garments = relationship("Garment", back_populates="user")
    preferences = relationship("UserPreferences", uselist=False)


class Garment(Base):
    __tablename__ = "garments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    original_image_path = Column(String(500), nullable=False)
    processed_image_path = Column(String(500))
    thumbnail_path = Column(String(500))
    status = Column(String(20), default="pending")
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    
    user = relationship("User", back_populates="garments")
    predictions = relationship("GarmentPrediction", uselist=False)
    labels = relationship("GarmentLabel", uselist=False)


class GarmentPrediction(Base):
    __tablename__ = "garment_predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    garment_id = Column(UUID(as_uuid=True), ForeignKey("garments.id"), nullable=False)
    category = Column(String(50), nullable=False)
    category_confidence = Column(Float)
    attributes = Column(JSONB, default={})
    embedding_id = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())


class Outfit(Base):
    __tablename__ = "outfits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    garment_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    occasion = Column(String(50))
    context = Column(JSONB)
    compatibility_score = Column(Float)
    score_breakdown = Column(JSONB)
    explanation = Column(Text)
    source = Column(String(20), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    feedback = relationship("OutfitFeedback", back_populates="outfit")


class OutfitFeedback(Base):
    __tablename__ = "outfit_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    outfit_id = Column(UUID(as_uuid=True), ForeignKey("outfits.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    action = Column(String(20), nullable=False)
    reason = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())
    
    outfit = relationship("Outfit", back_populates="feedback")


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    active_constraints = Column(JSONB, default={})
    excluded_item_ids = Column(ARRAY(UUID(as_uuid=True)), default=[])
    created_at = Column(DateTime, server_default=func.now())
    
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    recommended_outfit_ids = Column(ARRAY(UUID(as_uuid=True)))
    extracted_constraints = Column(JSONB)
    created_at = Column(DateTime, server_default=func.now())
    
    conversation = relationship("Conversation", back_populates="messages")
```

---

## Migration Strategy

### Alembic usage (BE-003)
- Location: `backend/alembic.ini`, env: `backend/alembic/env.py`, versions: `backend/alembic/versions/`
- DB URL: set `DATABASE_URL` (or POSTGRES_* envs read by `settings.DATABASE_URL`, e.g. `postgresql://postgres:postgres@localhost:5432/fashion_app`)
- UUIDs: DB uses `gen_random_uuid()` defaults; enable once per DB with `CREATE EXTENSION IF NOT EXISTS "pgcrypto";`

```bash
# Apply current migrations
poetry run alembic upgrade head

# Roll back everything (use with care)
poetry run alembic downgrade base

# Create new migration from models
poetry run alembic revision --autogenerate -m "short message"

# Inspect state (optional sanity)
poetry run alembic history
poetry run alembic heads
```

### Adding New Columns

```python
# backend/alembic/versions/xxx_add_column.py
def upgrade():
    op.add_column("garments", sa.Column("favorite", sa.Boolean(), nullable=True, server_default="false"))

def downgrade():
    op.drop_column("garments", "favorite")
```


# ML Pipeline Tasks (ML-xxx)

All ML tasks for image processing, model training, and recommendation algorithms.

---

## ML-001: Image Preprocessing Pipeline

**Phase:** 2 | **Priority:** P0 | **Dependencies:** BE-004

**Description:**
Create image preprocessing functions: resize, normalize, format conversion for ML models.

**Files to Create:**
- `ml/ingestion/__init__.py` - Update with exports
- `ml/ingestion/preprocessing.py` - Preprocessing functions
- `ml/utils/image_utils.py` - Image utilities

**Key Requirements:**
- Load image from path or bytes
- Resize to model input size (640 for YOLO, 224 for classifier)
- Normalize pixel values
- Convert color spaces if needed (BGR ↔ RGB)
- Return both original and processed versions

**Acceptance Criteria:**
- [ ] Loads JPEG, PNG, WebP images
- [ ] Resizes preserving aspect ratio
- [ ] Normalizes to 0-1 or ImageNet stats
- [ ] Handles corrupted images gracefully

---

## ML-002: Image Quality Checker

**Phase:** 2 | **Priority:** P1 | **Dependencies:** ML-001

**Description:**
Validate image quality before ML processing: blur, exposure, resolution checks.

**Files to Create:**
- `ml/ingestion/quality_check.py` - Quality checker

**Checks:**
- Resolution: reject if < 224px on any side
- Blur: Laplacian variance < 100 → warning
- Exposure: histogram extremes → warning
- Is clothing: (optional) binary classifier

**Acceptance Criteria:**
- [ ] Rejects low resolution images
- [ ] Detects blurry images
- [ ] Detects over/underexposed images
- [ ] Returns structured result with warnings

---

## ML-003: YOLOv8 Detection Integration

**Phase:** 2 | **Priority:** P0 | **Dependencies:** ML-001

**Description:**
Integrate YOLOv8 for garment detection. Use pretrained model initially.

**Files to Create:**
- `ml/ingestion/detection.py` - Detection module

**Key Requirements:**
- Load YOLOv8 model (pretrained or fine-tuned)
- Run inference on image
- Return bounding boxes with class and confidence
- Crop detected regions for classification
- Handle multiple detections

**Acceptance Criteria:**
- [ ] Loads YOLO model
- [ ] Detects garments in image
- [ ] Returns bbox, class, confidence
- [ ] Can crop detected regions

---

## ML-004: Download DeepFashion2 Dataset

**Phase:** 2 | **Priority:** P0 | **Dependencies:** None

**Description:**
Download and prepare DeepFashion2 dataset for training detection and classification models.

**Files to Create:**
- `scripts/data/download_deepfashion2.py` - Download script

**Key Requirements:**
- Download from official source (requires registration)
- Extract to `data/datasets/deepfashion2/`
- Document download process in script
- Create subset for quick testing (optional)

**Dataset Info:**
- ~800K images
- 13 garment categories
- Bounding boxes and segmentation masks
- Train/val/test splits

**Acceptance Criteria:**
- [ ] Download instructions documented
- [ ] Dataset extracted to correct location
- [ ] Can load sample images and annotations
- [ ] README with dataset info

---

## ML-005: Fine-tune YOLOv8 on Fashion

**Phase:** 2 | **Priority:** P1 | **Dependencies:** ML-003, ML-004

**Description:**
Fine-tune YOLOv8 on DeepFashion2 for better fashion detection.

**Files to Create:**
- `scripts/training/train_yolo.py` - Training script
- `scripts/data/prepare_yolo_data.py` - Data preparation

**Key Requirements:**
- Convert DeepFashion2 to YOLO format
- Create `data.yaml` config file
- Train YOLOv8s (small) on RTX 3050
- Validate on held-out set
- Save best model to `model_artifacts/`

**Training Config:**
- Epochs: 50
- Batch size: 8 (for 4GB VRAM)
- Image size: 640
- Patience: 10 (early stopping)

**Acceptance Criteria:**
- [ ] Data converted to YOLO format
- [ ] Training completes without OOM
- [ ] mAP@0.5 > 0.7 on validation
- [ ] Model saved to `model_artifacts/yolo_fashion.pt`

---

## ML-006: EfficientNet Classifier Setup

**Phase:** 2 | **Priority:** P0 | **Dependencies:** ML-003

**Description:**
Set up EfficientNet-B0 for garment category classification.

**Files to Create:**
- `ml/ingestion/classification.py` - Classification module
- `ml/utils/constants.py` - Category taxonomy

**Key Requirements:**
- Load EfficientNet-B0 with ImageNet weights
- Define category taxonomy (~47 categories)
- Replace classifier head for our categories
- Inference function: image → (category, confidence)
- Top-K prediction function

**Acceptance Criteria:**
- [ ] Model loads successfully
- [ ] Category taxonomy defined
- [ ] Inference runs on sample image
- [ ] Returns category and confidence

---

## ML-007: Train Garment Classifier

**Phase:** 2 | **Priority:** P1 | **Dependencies:** ML-006, ML-004

**Description:**
Fine-tune EfficientNet classifier on fashion categories.

**Files to Create:**
- `scripts/training/train_classifier.py` - Training script
- `scripts/data/prepare_classifier_data.py` - Data preparation

**Key Requirements:**
- Prepare data in ImageFolder format
- Data augmentation (flip, crop, color jitter)
- Fine-tune with frozen backbone initially
- Unfreeze and fine-tune all layers
- Save best model by validation accuracy

**Training Config:**
- Epochs: 30
- Batch size: 32
- Learning rate: 1e-4
- Optimizer: AdamW

**Acceptance Criteria:**
- [ ] Data prepared in correct format
- [ ] Training completes
- [ ] Top-1 accuracy > 80% on validation
- [ ] Model saved to `model_artifacts/classifier.pth`

---

## ML-008: Color Extraction (Rule-Based)

**Phase:** 2 | **Priority:** P0 | **Dependencies:** ML-003

**Description:**
Extract dominant colors from garment image using K-means clustering.

**Files to Create:**
- `ml/ingestion/attributes/color.py` - Color extraction

**Key Requirements:**
- K-means clustering on pixel values
- Map RGB to named fashion colors (navy, beige, etc.)
- Return primary and secondary colors
- Filter out background pixels
- Define fashion color palette

**Acceptance Criteria:**
- [ ] Extracts 2-3 dominant colors
- [ ] Maps to named colors correctly
- [ ] Returns color percentages
- [ ] Handles simple backgrounds

---

## ML-009: Attribute Tagger (Pattern, Fit)

**Phase:** 2 | **Priority:** P1 | **Dependencies:** ML-006

**Description:**
Multi-label classifier for garment attributes: pattern, fit, formality, style.

**Files to Create:**
- `ml/ingestion/attributes/tagger.py` - Attribute tagger

**Attributes:**
- Pattern: solid, striped, plaid, floral, graphic, etc.
- Fit: slim, regular, relaxed, oversized
- Formality: float 0-1
- Seasons: multi-label (spring, summer, fall, winter)
- Style tags: casual, formal, preppy, streetwear, etc.

**Key Requirements:**
- Shared backbone with separate heads per attribute
- Multi-label for seasons and style tags
- Regression for formality
- Train on iMaterialist or similar dataset

**Acceptance Criteria:**
- [ ] Model architecture defined
- [ ] Predicts all attribute types
- [ ] Reasonable accuracy on test images

---

## ML-010: CLIP Embedding Pipeline

**Phase:** 2 | **Priority:** P0 | **Dependencies:** ML-003

**Description:**
Generate CLIP embeddings for garment images.

**Files to Create:**
- `ml/ingestion/embeddings.py` - Embedding generator

**Key Requirements:**
- Load OpenCLIP ViT-B-32 model
- Preprocess image for CLIP
- Generate 512-dim embedding
- Normalize embedding vector
- Batch processing support

**Acceptance Criteria:**
- [ ] Loads CLIP model
- [ ] Generates 512-dim embeddings
- [ ] Embeddings are normalized
- [ ] Can process batch of images

---

## ML-011: ChromaDB Integration

**Phase:** 2 | **Priority:** P0 | **Dependencies:** ML-010

**Description:**
Set up ChromaDB for storing and querying garment embeddings.

**Files to Create:**
- `ml/vector_store/client.py` - ChromaDB client

**Key Requirements:**
- Connect to ChromaDB (persistent storage)
- Create/get garments collection
- Add embeddings with metadata (user_id, category, color)
- Query similar embeddings
- Delete embeddings

**Acceptance Criteria:**
- [ ] ChromaDB initializes correctly
- [ ] Can add embeddings
- [ ] Can query similar items
- [ ] Persistence works across restarts

---

## ML-012: Outfit Slot Logic

**Phase:** 3 | **Priority:** P0 | **Dependencies:** ML-006

**Description:**
Define outfit slots and map garment categories to slots.

**Files to Create:**
- `ml/recommendation/__init__.py` - Update exports
- `ml/recommendation/slots.py` - Slot definitions

**Slots:**
- `top` - shirts, blouses, sweaters, t-shirts
- `bottom` - pants, jeans, shorts, skirts
- `outerwear` - jackets, coats, blazers
- `shoes` - sneakers, boots, dress shoes
- `accessories` - belts, ties, scarves (optional)

**Key Requirements:**
- Map each category to slot
- Define required slots for outfit (top + bottom + shoes)
- Define optional slots (outerwear, accessories)
- Handle dresses (fills top + bottom)

**Acceptance Criteria:**
- [ ] All categories mapped to slots
- [ ] Required vs optional slots defined
- [ ] Dress/jumpsuit handling logic

---

## ML-013: Candidate Generator

**Phase:** 3 | **Priority:** P0 | **Dependencies:** ML-012

**Description:**
Generate outfit candidate combinations from wardrobe.

**Files to Create:**
- `ml/recommendation/generator.py` - Candidate generator

**Key Requirements:**
- Input: filtered garments by slot
- Output: list of outfit combinations
- Smart sampling (not exhaustive enumeration)
- Ensure diversity in candidates
- Configurable number of candidates

**Strategies:**
- Random sampling from each slot
- Embedding-based diversity (select dissimilar items)
- Category balance (don't repeat same item type)

**Acceptance Criteria:**
- [ ] Generates valid outfit combinations
- [ ] Respects slot requirements
- [ ] Produces diverse candidates
- [ ] Configurable candidate count

---

## ML-014: Rule-Based Compatibility Scorer

**Phase:** 3 | **Priority:** P0 | **Dependencies:** ML-013

**Description:**
Score outfit compatibility using fashion rules.

**Files to Create:**
- `ml/recommendation/compatibility/rules.py` - Rule-based scorer

**Scoring Components:**
- Color harmony (complementary colors, neutrals)
- Formality match (all items similar formality)
- Category compatibility (some combos work better)
- Overall score: weighted average

**Key Requirements:**
- Define color compatibility rules
- Define good/bad category combinations
- Return score 0-1 with breakdown
- Generate explanation text

**Acceptance Criteria:**
- [ ] Scores outfits consistently
- [ ] Color clashes get lower scores
- [ ] Formality mismatches penalized
- [ ] Explanations generated

---

## ML-015: Color Harmony Rules

**Phase:** 3 | **Priority:** P0 | **Dependencies:** ML-014

**Description:**
Define color harmony rules for outfit scoring.

**Files to Modify:**
- `ml/recommendation/compatibility/rules.py` - Add color rules

**Rules:**
- Neutral colors (black, white, grey, navy) go with everything
- Complementary color pairs score high
- More than 2 non-neutral colors → lower score
- All neutrals → safe but slightly lower score

**Color Pairs:**
- navy + tan/white/burgundy
- blue + brown/tan
- black + white/red
- etc.

**Acceptance Criteria:**
- [ ] Neutral-heavy outfits score ~0.75
- [ ] Complementary pairs score ~0.9-1.0
- [ ] Clashing colors score < 0.5

---

## ML-016: Formality Matching Logic

**Phase:** 3 | **Priority:** P0 | **Dependencies:** ML-014

**Description:**
Score formality consistency across outfit items.

**Files to Modify:**
- `ml/recommendation/compatibility/rules.py` - Add formality logic

**Logic:**
- Calculate variance of formality values
- Low variance = good match
- High variance = mixed formality = lower score
- Score = 1 - (variance * scale_factor)

**Acceptance Criteria:**
- [ ] All formal items → high score
- [ ] All casual items → high score
- [ ] Mixed (blazer + sneakers) → lower score

---

## ML-017: Download Polyvore Dataset

**Phase:** 3 | **Priority:** P1 | **Dependencies:** None

**Description:**
Download Polyvore Outfits dataset for training compatibility model.

**Files to Create:**
- `scripts/data/download_polyvore.py` - Download script

**Dataset Info:**
- ~365K outfits
- Positive pairs (items that go together)
- Can generate negatives by shuffling
- Train/val/test splits

**Acceptance Criteria:**
- [ ] Download instructions documented
- [ ] Dataset extracted correctly
- [ ] Can load outfit data

---

## ML-018: Train Learned Compatibility Model

**Phase:** 3 | **Priority:** P2 | **Dependencies:** ML-017, ML-010

**Description:**
Train neural network for outfit compatibility scoring.

**Files to Create:**
- `ml/recommendation/compatibility/learned.py` - Learned model
- `scripts/training/train_compatibility.py` - Training script

**Model Architecture:**
- Input: garment embeddings (from CLIP)
- Transformer encoder over item set
- Mean pooling
- MLP head → compatibility score

**Training:**
- Positive: real outfits from Polyvore
- Negative: shuffled items from different outfits
- Loss: Binary cross-entropy

**Acceptance Criteria:**
- [ ] Model architecture defined
- [ ] Training completes
- [ ] Accuracy > 65% on test set
- [ ] Model saved to `model_artifacts/`

---

## ML-019: Feedback-Based Re-ranking

**Phase:** 5 | **Priority:** P1 | **Dependencies:** BE-011

**Description:**
Adjust outfit rankings based on user feedback history.

**Files to Create:**
- `ml/recommendation/personalization.py` - Personalization module

**Key Requirements:**
- Track liked/disliked attributes per user
- Boost items with liked attributes
- Penalize items with disliked attributes
- Apply as score adjustment (not filter)

**Acceptance Criteria:**
- [ ] Tracks feedback by attribute
- [ ] Adjusts scores based on history
- [ ] Improvement visible after 10+ feedbacks

---

## ML-020: User Preference Learning

**Phase:** 5 | **Priority:** P2 | **Dependencies:** ML-019

**Description:**
Learn user preferences from feedback signals over time.

**Files to Modify:**
- `ml/recommendation/personalization.py` - Add learning

**Signals:**
- Like → +1.0
- Dislike → -1.0
- Wear → +2.0
- Skip → -0.3

**Learning:**
- Aggregate signals by category, color, style
- Update weights incrementally
- Store in user preferences

**Acceptance Criteria:**
- [ ] Weights updated from feedback
- [ ] Stored in database
- [ ] Applied to future recommendations

---

## ML-021: Fine-tune CLIP on Fashion

**Phase:** 5 | **Priority:** P2 | **Dependencies:** ML-010

**Description:**
Fine-tune CLIP embeddings on fashion-specific data for better similarity.

**Files to Create:**
- `scripts/training/finetune_clip.py` - Fine-tuning script

**Approach:**
- Contrastive learning on fashion image-text pairs
- Use product descriptions from fashion datasets
- Keep original CLIP knowledge, add fashion expertise

**Acceptance Criteria:**
- [ ] Fine-tuning script works
- [ ] Embeddings cluster better by category
- [ ] Similar items more similar than before


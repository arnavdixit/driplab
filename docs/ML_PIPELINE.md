# ML Pipeline Architecture

**Relevant Code Paths:**
- Ingestion pipeline: `ml/ingestion/`
- Recommendation engine: `ml/recommendation/`
- Chatbot ML components: `ml/chatbot/`
- Vector store: `ml/vector_store/`
- Training scripts: `scripts/training/`
- Model weights: `model_artifacts/`
- Training data: `data/datasets/` or `ml/datasets/`
- Tests: `ml/tests/`

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                                │
│                                                                             │
│   Image    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          │
│   Upload──▶│ Quality │──▶│ Detect  │──▶│Classify │──▶│  Tag    │──▶       │
│            │  Check  │   │(YOLOv8) │   │(EffNet) │   │Attributes│          │
│            └─────────┘   └─────────┘   └─────────┘   └─────────┘          │
│                                                            │               │
│                                                            ▼               │
│                                                      ┌─────────┐          │
│                                                      │  CLIP   │──▶ ChromaDB
│                                                      │Embedding│          │
│                                                      └─────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        RECOMMENDATION PIPELINE                              │
│                                                                             │
│  Constraints  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  ───────────▶│ Filter  │──▶│Generate │──▶│  Score  │──▶│  Rank   │──▶ Top K
│              │Wardrobe │   │Candidates   │Compat.  │   │& Return │       │
│              └─────────┘   └─────────┘   └─────────┘   └─────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Image Quality Check

**Purpose:** Validate uploads before expensive ML processing

### Checks

| Check | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Resolution | Dimension check | < 224px | Reject |
| Blur | Laplacian variance | < 100 | Warn |
| Exposure | Histogram analysis | Extreme | Warn |
| Is clothing | Binary classifier | < 0.7 conf | Flag |

### Implementation

```python
# ml/ingestion/quality_check.py
import cv2
import numpy as np

class ImageQualityChecker:
    
    def check_resolution(self, image: np.ndarray, min_size: int = 224) -> bool:
        h, w = image.shape[:2]
        return min(h, w) >= min_size
    
    def check_blur(self, image: np.ndarray, threshold: float = 100) -> tuple[bool, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= threshold, variance
    
    def check_exposure(self, image: np.ndarray) -> tuple[bool, str]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        
        if mean_brightness < 40:
            return False, "too_dark"
        if mean_brightness > 220:
            return False, "too_bright"
        return True, "ok"
    
    def run_all(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        
        result = {"passed": True, "warnings": [], "errors": []}
        
        if not self.check_resolution(image):
            result["passed"] = False
            result["errors"].append("Image too small (min 224x224)")
            return result
        
        is_sharp, blur_score = self.check_blur(image)
        if not is_sharp:
            result["warnings"].append(f"Image may be blurry ({blur_score:.0f})")
        
        is_exposed, status = self.check_exposure(image)
        if not is_exposed:
            result["warnings"].append(f"Image is {status}")
        
        return result
```

### Training Required
- **None for MVP** - Uses heuristics
- **Optional:** Train small binary classifier for "is clothing" check (~1K images)

---

## Module 2: Garment Detection (YOLOv8)

**Purpose:** Locate garments in image with bounding boxes

### Model Choice

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| YOLOv8n | 3.2M | 1.2ms | Good | 2GB |
| **YOLOv8s** | 11.2M | 2.0ms | Better | 4GB |
| YOLOv8m | 25.9M | 4.5ms | Best | 8GB |

**Recommendation:** YOLOv8s - Best balance for RTX 3050

### Classes (from DeepFashion2)

```python
DETECTION_CLASSES = [
    "short_sleeve_top",     # 0
    "long_sleeve_top",      # 1
    "short_sleeve_outwear", # 2
    "long_sleeve_outwear",  # 3
    "vest",                 # 4
    "sling",                # 5
    "shorts",               # 6
    "trousers",             # 7
    "skirt",                # 8
    "short_sleeve_dress",   # 9
    "long_sleeve_dress",    # 10
    "vest_dress",           # 11
    "sling_dress"           # 12
]
```

### Implementation

```python
# ml/detection.py
from ultralytics import YOLO
import numpy as np

class GarmentDetector:
    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model = YOLO(model_path)
        self.classes = DETECTION_CLASSES
    
    def detect(self, image_path: str, conf_threshold: float = 0.5) -> list[dict]:
        results = self.model(image_path, conf=conf_threshold)
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": self.classes[int(box.cls)],
                    "confidence": float(box.conf)
                })
        
        return detections
    
    def crop_detection(self, image: np.ndarray, bbox: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
```

### Training Procedure

```python
# scripts/train_yolo.py
from ultralytics import YOLO

# 1. Prepare DeepFashion2 in YOLO format
"""
data/deepfashion2_yolo/
├── images/
│   ├── train/  (190K images)
│   └── val/    (32K images)
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
"""

# data.yaml
DATA_YAML = """
path: ./data/deepfashion2_yolo
train: images/train
val: images/val

names:
  0: short_sleeve_top
  1: long_sleeve_top
  2: short_sleeve_outwear
  3: long_sleeve_outwear
  4: vest
  5: sling
  6: shorts
  7: trousers
  8: skirt
  9: short_sleeve_dress
  10: long_sleeve_dress
  11: vest_dress
  12: sling_dress
"""

# 2. Train
model = YOLO("yolov8s.pt")

results = model.train(
    data="data/deepfashion2_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,          # RTX 3050 (4GB)
    device=0,
    patience=10,
    project="runs/detect",
    name="fashion_yolov8s"
)

# 3. Export
model.export(format="onnx")  # For faster inference
```

### Training Resources
- **Dataset:** DeepFashion2 (~190K train images)
- **Time:** ~8-12 hours on RTX 3050
- **Alternative:** Use cloud GPU (2-3 hours on A100)

---

## Module 3: Classification (EfficientNet)

**Purpose:** Classify cropped garment into fine-grained category

### Category Taxonomy

```python
CATEGORY_TAXONOMY = {
    "tops": [
        "t_shirt", "polo", "tank_top", "henley",
        "dress_shirt", "oxford", "flannel", "blouse",
        "sweater", "cardigan", "hoodie", "sweatshirt"
    ],
    "bottoms": [
        "jeans", "chinos", "dress_pants", "shorts",
        "sweatpants", "joggers", "skirt", "leggings"
    ],
    "outerwear": [
        "blazer", "suit_jacket", "bomber", "denim_jacket",
        "leather_jacket", "coat", "parka", "vest", "cardigan"
    ],
    "dresses": [
        "casual_dress", "formal_dress", "maxi_dress",
        "midi_dress", "mini_dress", "sundress"
    ],
    "footwear": [
        "sneakers", "dress_shoes", "loafers", "boots",
        "sandals", "heels", "flats"
    ],
    "accessories": [
        "belt", "tie", "scarf", "hat", "watch", "bag"
    ]
}

ALL_CATEGORIES = [c for cats in CATEGORY_TAXONOMY.values() for c in cats]
# 47 categories
```

### Implementation

```python
# ml/classification.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class GarmentClassifier:
    def __init__(self, model_path: str = None, num_classes: int = 47):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Build model
        self.model = models.efficientnet_b0(weights="DEFAULT")
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path: str) -> tuple[str, float]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        
        return ALL_CATEGORIES[idx.item()], conf.item()
    
    def predict_top_k(self, image_path: str, k: int = 3) -> list[tuple[str, float]]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = probs.topk(k)
        
        return [
            (ALL_CATEGORIES[idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
```

### Training Procedure

```python
# scripts/train_classifier.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data (ImageFolder format)
train_dataset = datasets.ImageFolder("data/classification/train", train_transform)
val_dataset = datasets.ImageFolder("data/classification/val", val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

# Train
model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(1280, len(ALL_CATEGORIES))
model = model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    print(f"Epoch {epoch+1}: Accuracy = {100*correct/total:.2f}%")

# Save
torch.save(model.state_dict(), "models/classifier.pth")
```

### Training Resources
- **Dataset:** DeepFashion2 + iMaterialist
- **Time:** ~2-4 hours on RTX 3050
- **Target Accuracy:** 80-85%

---

## Module 4: Attribute Tagging

**Purpose:** Extract color, pattern, fit, formality, style tags

### Attribute Schema

```python
ATTRIBUTES = {
    "color_primary": [
        "black", "white", "grey", "navy", "blue", "red", "green",
        "brown", "tan", "beige", "pink", "purple", "orange", "yellow"
    ],
    "pattern": [
        "solid", "striped", "plaid", "checkered", "floral",
        "graphic", "polka_dot", "abstract"
    ],
    "fit": ["slim", "regular", "relaxed", "oversized"],
    "formality": float,  # 0.0 to 1.0
    "seasons": ["spring", "summer", "fall", "winter"],  # multi-label
    "style_tags": [
        "casual", "formal", "preppy", "streetwear", "minimalist",
        "bohemian", "athletic", "classic", "trendy", "vintage"
    ]  # multi-label
}
```

### Color Extraction (Rule-Based)

```python
# ml/attributes/color.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

FASHION_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "navy": (0, 0, 128),
    "blue": (70, 130, 180),
    "red": (220, 20, 60),
    "green": (34, 139, 34),
    "brown": (139, 69, 19),
    "tan": (210, 180, 140),
    "beige": (245, 245, 220),
    "pink": (255, 182, 193),
    "purple": (128, 0, 128),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
}

def extract_colors(image: np.ndarray, n_colors: int = 2) -> list[dict]:
    # Reshape and filter background
    pixels = image.reshape(-1, 3)
    mask = (pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 700)
    pixels = pixels[mask]
    
    if len(pixels) < 100:
        return [{"color": "unknown", "percentage": 1.0}]
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Map to fashion colors
    results = []
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    for label, count in sorted(zip(labels, counts), key=lambda x: -x[1]):
        rgb = tuple(kmeans.cluster_centers_[label].astype(int))
        color_name = _closest_color(rgb)
        results.append({
            "color": color_name,
            "rgb": rgb,
            "percentage": count / len(pixels)
        })
    
    return results

def _closest_color(rgb: tuple) -> str:
    min_dist = float("inf")
    closest = "unknown"
    for name, ref_rgb in FASHION_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest
```

### Pattern/Style Classifier (ML)

```python
# ml/attributes/tagger.py
import torch
import torch.nn as nn
from torchvision import models

class AttributeTagger(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared backbone (frozen for faster training)
        backbone = models.efficientnet_b0(weights="DEFAULT")
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Attribute heads
        self.pattern_head = nn.Linear(1280, 8)      # 8 patterns
        self.fit_head = nn.Linear(1280, 4)          # 4 fits
        self.formality_head = nn.Linear(1280, 1)    # regression
        self.season_head = nn.Linear(1280, 4)       # multi-label
        self.style_head = nn.Linear(1280, 10)       # multi-label
    
    def forward(self, x):
        features = self.features(x).flatten(1)
        
        return {
            "pattern": self.pattern_head(features),
            "fit": self.fit_head(features),
            "formality": torch.sigmoid(self.formality_head(features)),
            "seasons": torch.sigmoid(self.season_head(features)),
            "style_tags": torch.sigmoid(self.style_head(features))
        }

class AttributePredictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttributeTagger()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.patterns = ["solid", "striped", "plaid", "checkered", 
                         "floral", "graphic", "polka_dot", "abstract"]
        self.fits = ["slim", "regular", "relaxed", "oversized"]
        self.seasons = ["spring", "summer", "fall", "winter"]
        self.styles = ["casual", "formal", "preppy", "streetwear", "minimalist",
                       "bohemian", "athletic", "classic", "trendy", "vintage"]
    
    def predict(self, image_tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
        
        return {
            "pattern": self.patterns[outputs["pattern"].argmax().item()],
            "fit": self.fits[outputs["fit"].argmax().item()],
            "formality": outputs["formality"].item(),
            "seasons": [s for s, v in zip(self.seasons, outputs["seasons"][0]) if v > 0.5],
            "style_tags": [s for s, v in zip(self.styles, outputs["style_tags"][0]) if v > 0.5]
        }
```

### Training Data
- **iMaterialist Fashion:** 228 attribute labels
- **Map to our simplified schema**
- **Time:** ~2-3 hours on RTX 3050

---

## Module 5: CLIP Embeddings

**Purpose:** Generate semantic embeddings for similarity search

### Why CLIP?
- Pretrained on 400M image-text pairs
- Understands fashion concepts zero-shot
- Enables text-to-image search ("find casual blue shirts")

### Implementation

```python
# ml/embeddings.py
import open_clip
import torch
import numpy as np
from PIL import Image

class GarmentEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
    
    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()  # 512-dim
    
    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten()
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))
    
    def embed_batch(self, image_paths: list[str]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_image(tensors)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()  # [N, 512]
```

### ChromaDB Integration

```python
# ml/vector_store.py
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="garments",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, garment_id: str, embedding: np.ndarray, metadata: dict):
        self.collection.add(
            ids=[garment_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )
    
    def query_similar(self, embedding: np.ndarray, n: int = 10, 
                      filters: dict = None) -> list[str]:
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n,
            where=filters
        )
        return results["ids"][0]
    
    def delete(self, garment_id: str):
        self.collection.delete(ids=[garment_id])
```

---

## Module 6: Compatibility Scoring

**Purpose:** Score how well garments work together in an outfit

### Rule-Based Scorer (MVP)

```python
# ml/compatibility/rules.py
import numpy as np

class RuleBasedScorer:
    
    # Color harmony rules
    COMPLEMENTARY = {
        "navy": ["white", "tan", "cream", "burgundy"],
        "black": ["white", "grey", "red", "pink"],
        "grey": ["navy", "blue", "pink", "burgundy"],
        "blue": ["tan", "brown", "white", "grey"],
        "brown": ["blue", "navy", "white", "cream"],
    }
    
    NEUTRALS = {"black", "white", "grey", "navy", "beige", "cream", "tan"}
    
    # Category combinations
    GOOD_COMBOS = {
        ("dress_shirt", "dress_pants"): 1.0,
        ("dress_shirt", "chinos"): 0.9,
        ("t_shirt", "jeans"): 1.0,
        ("t_shirt", "shorts"): 0.9,
        ("blazer", "dress_shirt"): 1.0,
        ("blazer", "t_shirt"): 0.7,
        ("sweater", "dress_shirt"): 0.9,
        ("hoodie", "jeans"): 0.9,
    }
    
    def score(self, garments: list[dict]) -> dict:
        """
        Score outfit given list of garments.
        Each garment: {category, color_primary, formality, ...}
        """
        color_score = self._score_colors(garments)
        formality_score = self._score_formality(garments)
        combo_score = self._score_combinations(garments)
        
        overall = 0.35 * color_score + 0.35 * formality_score + 0.3 * combo_score
        
        return {
            "total": overall,
            "breakdown": {
                "color_harmony": color_score,
                "formality_match": formality_score,
                "category_combo": combo_score
            }
        }
    
    def _score_colors(self, garments: list[dict]) -> float:
        colors = [g["color_primary"] for g in garments if g.get("color_primary")]
        
        if len(colors) < 2:
            return 0.8
        
        non_neutrals = [c for c in colors if c not in self.NEUTRALS]
        
        # All neutrals = safe
        if not non_neutrals:
            return 0.75
        
        # One accent + neutrals = good
        if len(non_neutrals) == 1:
            return 0.9
        
        # Check complementary
        if len(non_neutrals) >= 2:
            c1, c2 = non_neutrals[0], non_neutrals[1]
            if c2 in self.COMPLEMENTARY.get(c1, []):
                return 1.0
            return 0.5  # Potentially clashing
        
        return 0.7
    
    def _score_formality(self, garments: list[dict]) -> float:
        formalities = [g.get("formality", 0.5) for g in garments]
        
        if len(formalities) < 2:
            return 1.0
        
        variance = np.var(formalities)
        return max(0, 1 - variance * 5)
    
    def _score_combinations(self, garments: list[dict]) -> float:
        categories = [g["category"] for g in garments]
        
        score = 0.7  # Base score
        
        for i, c1 in enumerate(categories):
            for c2 in categories[i+1:]:
                pair = tuple(sorted([c1, c2]))
                if pair in self.GOOD_COMBOS:
                    score = max(score, self.GOOD_COMBOS[pair])
        
        return score
```

### Learned Model (V1+)

```python
# ml/compatibility/learned.py
import torch
import torch.nn as nn

class CompatibilityModel(nn.Module):
    """Transformer-based outfit compatibility model"""
    
    def __init__(self, embedding_dim: int = 512, max_items: int = 6):
        super().__init__()
        
        self.item_proj = nn.Linear(embedding_dim, 256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None):
        # embeddings: [batch, max_items, 512]
        x = self.item_proj(embeddings)  # [batch, max_items, 256]
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Mean pooling over items
        if mask is not None:
            x = x * (~mask).unsqueeze(-1).float()
            x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        return self.head(x).squeeze(-1)
```

### Training on Polyvore

```python
# Train with positive (real outfits) and negative (shuffled) pairs
# Dataset: Polyvore Outfits (365K outfits)
# Loss: Binary cross-entropy
# Positive: Real outfit = 1
# Negative: Random items from different outfits = 0
```

---

## Module 7: Personalization

**Purpose:** Learn user preferences from feedback

### Feedback Signals

| Signal | Weight | Source |
|--------|--------|--------|
| Like | +1.0 | Explicit |
| Dislike | -1.0 | Explicit |
| Wear | +2.0 | Explicit (high signal) |
| Skip | -0.3 | Implicit |
| Click | +0.2 | Implicit |

### Simple Preference Learning

```python
# ml/personalization.py
from collections import defaultdict
import numpy as np

class PreferenceLearner:
    def __init__(self):
        self.category_weights = defaultdict(float)
        self.color_weights = defaultdict(float)
        self.style_weights = defaultdict(float)
        self.feedback_count = 0
    
    def update(self, garments: list[dict], signal: str):
        weight = {"like": 1.0, "dislike": -1.0, "wear": 2.0, "skip": -0.3}[signal]
        
        for g in garments:
            self.category_weights[g["category"]] += weight * 0.1
            self.color_weights[g["color_primary"]] += weight * 0.1
            for tag in g.get("style_tags", []):
                self.style_weights[tag] += weight * 0.1
        
        self.feedback_count += 1
    
    def adjust_score(self, outfit_score: float, garments: list[dict]) -> float:
        if self.feedback_count < 10:
            return outfit_score  # Not enough data
        
        adjustment = 0
        for g in garments:
            adjustment += self.category_weights.get(g["category"], 0)
            adjustment += self.color_weights.get(g["color_primary"], 0)
            for tag in g.get("style_tags", []):
                adjustment += self.style_weights.get(tag, 0)
        
        # Clamp adjustment
        adjustment = np.clip(adjustment, -0.3, 0.3)
        
        return outfit_score + adjustment
```

---

## Complete Pipeline

```python
# ml/pipeline.py
from ml.quality_check import ImageQualityChecker
from ml.detection import GarmentDetector
from ml.classification import GarmentClassifier
from ml.attributes import ColorExtractor, AttributeTagger
from ml.embeddings import GarmentEmbedder
from ml.vector_store import VectorStore

class IngestionPipeline:
    def __init__(self, models_dir: str = "./models"):
        self.quality_checker = ImageQualityChecker()
        self.detector = GarmentDetector(f"{models_dir}/yolo_fashion.pt")
        self.classifier = GarmentClassifier(f"{models_dir}/classifier.pth")
        self.color_extractor = ColorExtractor()
        self.attribute_tagger = AttributeTagger(f"{models_dir}/tagger.pth")
        self.embedder = GarmentEmbedder()
        self.vector_store = VectorStore()
    
    def process(self, image_path: str, garment_id: str, user_id: str) -> dict:
        # 1. Quality check
        quality = self.quality_checker.run_all(image_path)
        if not quality["passed"]:
            return {"status": "failed", "error": quality["errors"]}
        
        # 2. Detect
        detections = self.detector.detect(image_path)
        if not detections:
            return {"status": "failed", "error": "No garment detected"}
        
        # Use first detection (single garment expected)
        detection = detections[0]
        
        # 3. Crop and classify
        cropped = self.detector.crop_detection(image_path, detection["bbox"])
        category, conf = self.classifier.predict(cropped)
        
        # 4. Extract attributes
        colors = self.color_extractor.extract(cropped)
        attributes = self.attribute_tagger.predict(cropped)
        attributes["color_primary"] = colors[0]["color"]
        if len(colors) > 1:
            attributes["color_secondary"] = colors[1]["color"]
        
        # 5. Generate embedding
        embedding = self.embedder.embed_image(image_path)
        
        # 6. Store in vector DB
        self.vector_store.add(
            garment_id=garment_id,
            embedding=embedding,
            metadata={
                "user_id": user_id,
                "category": category,
                "color_primary": attributes["color_primary"]
            }
        )
        
        return {
            "status": "success",
            "detection": detection,
            "category": category,
            "category_confidence": conf,
            "attributes": attributes,
            "embedding_id": garment_id
        }
```

---

## Training Schedule

| Model | Dataset | Time (RTX 3050) | Time (Cloud A100) |
|-------|---------|-----------------|-------------------|
| YOLOv8 detection | DeepFashion2 | 8-12 hours | 2-3 hours |
| EfficientNet classifier | DeepFashion2 | 2-4 hours | 30-60 min |
| Attribute tagger | iMaterialist | 2-3 hours | 30-60 min |
| Compatibility model | Polyvore | 4-6 hours | 1-2 hours |

**Recommendation:** Train YOLOv8 on cloud, train others locally.

---

## Model Artifacts

```
models/
├── yolo_fashion.pt          # Fine-tuned YOLOv8
├── classifier.pth           # EfficientNet classifier
├── tagger.pth               # Attribute tagger
├── compatibility.pth        # Learned compatibility (V1+)
└── clip/                    # CLIP weights (downloaded)
    └── ViT-B-32.pt
```

---

## Evaluation Metrics

| Model | Metric | Target |
|-------|--------|--------|
| Detection | mAP@0.5 | > 0.7 |
| Classification | Top-1 Accuracy | > 80% |
| Attributes | Macro F1 | > 70% |
| Compatibility | Accuracy | > 65% |
| Color | Accuracy | > 85% |


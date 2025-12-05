# Chatbot Architecture

**Relevant Code Paths:**
- ML logic: `ml/chatbot/` (pure ML/LLM, no DB/HTTP)
- Orchestration: `backend/app/services/chat_orchestrator.py` (glue between API, DB, ML)
- API endpoint: `backend/app/api/v1/chat.py` (HTTP layer)
- Tests: `ml/tests/test_chatbot_*.py` (unit tests), `backend/tests/services/test_chat_orchestrator.py` (integration tests)

**Architecture Principle:** `ml/chatbot/` contains pure ML/LLM logic with no database or HTTP dependencies. `backend/app/services/chat_orchestrator.py` orchestrates by loading context from DB, calling ML functions, and saving results back to DB.

## Core Principle

The chatbot **only recommends items from the user's uploaded wardrobe**. Text instructions act as filters and re-rankers, never inventing new clothes.

```
User's Wardrobe (Source of Truth)
        │
        ▼
┌─────────────────────────────────────────┐
│   Text Instruction                      │
│   "I have a date, it's cold outside"    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│   Filter: occasion=date, weather=cold   │
│   Re-rank: prioritize romantic styles   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│   Output: Top outfits from YOUR clothes │
└─────────────────────────────────────────┘
```

---

## System Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              USER MESSAGE                                  │
│                "I need something for a job interview"                      │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: BUILD CONTEXT (Deterministic)                                     │
│  - Load conversation history                                               │
│  - Load session constraints                                                │
│  - Load wardrobe summary                                                   │
│  - Load user preferences                                                   │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: EXTRACT CONSTRAINTS (LLM Call #1)                                 │
│  - Parse natural language → structured constraints                         │
│  - Output: {occasion: "interview", formality: "business_formal", ...}      │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: MERGE CONSTRAINTS (Deterministic)                                 │
│  - Combine with existing session constraints                               │
│  - New overrides old for conflicts                                         │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: GENERATE OUTFITS (Deterministic + ML)                             │
│  - Filter wardrobe by constraints                                          │
│  - Generate candidate combinations                                         │
│  - Score with compatibility model                                          │
│  - Apply personalization                                                   │
│  - Select top K                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: GENERATE RESPONSE (LLM Call #2)                                   │
│  - Format outfits into natural language                                    │
│  - Explain why each works                                                  │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              RESPONSE                                      │
│  "For your interview, I'd suggest the navy blazer with..."                 │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Context Building

```python
# ml/chatbot/context.py
from dataclasses import dataclass

@dataclass
class ChatContext:
    conversation_history: list[dict]  # [{role, content}, ...]
    session_constraints: dict          # Active filters
    excluded_items: set[str]          # Rejected item IDs
    wardrobe_summary: dict            # {tops: 12, bottoms: 8, ...}
    user_preferences: dict            # From profile

class ContextBuilder:
    def __init__(self, db):
        self.db = db
    
    def build(self, user_id: str, conversation_id: str) -> ChatContext:
        # Load conversation
        conv = self.db.get_conversation(conversation_id)
        messages = self.db.get_messages(conversation_id, limit=10)
        
        # Load wardrobe summary
        garments = self.db.get_garments(user_id)
        summary = self._summarize(garments)
        
        # Load preferences
        prefs = self.db.get_preferences(user_id)
        
        return ChatContext(
            conversation_history=[{"role": m.role, "content": m.content} for m in messages],
            session_constraints=conv.active_constraints or {},
            excluded_items=set(conv.excluded_item_ids or []),
            wardrobe_summary=summary,
            user_preferences=prefs or {}
        )
    
    def _summarize(self, garments) -> dict:
        summary = {"total": len(garments), "by_category": {}}
        for g in garments:
            cat = g.predictions.category if g.predictions else "unknown"
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
        return summary
```

---

## Step 2: Constraint Extraction (LLM)

### OpenAI Function Definition

```python
EXTRACT_CONSTRAINTS = {
    "name": "extract_outfit_constraints",
    "description": "Extract structured outfit constraints from user message",
    "parameters": {
        "type": "object",
        "properties": {
            "occasion": {
                "type": "string",
                "enum": ["casual", "work", "business_formal", "date", 
                         "party", "wedding", "interview", "workout", "outdoor"]
            },
            "formality_level": {
                "type": "string",
                "enum": ["very_casual", "casual", "smart_casual", 
                         "business_casual", "business_formal", "formal"]
            },
            "weather": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "string", "enum": ["hot", "warm", "mild", "cool", "cold"]},
                    "conditions": {"type": "array", "items": {"type": "string"}}
                }
            },
            "exclude_categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categories to avoid (e.g., jeans, sneakers)"
            },
            "exclude_colors": {
                "type": "array",
                "items": {"type": "string"}
            },
            "must_include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Items that must be in the outfit"
            },
            "style_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Style words (elegant, relaxed, edgy)"
            },
            "comfort_priority": {
                "type": "number",
                "description": "0=style priority, 1=comfort priority"
            }
        }
    }
}
```

### System Prompt

```python
SYSTEM_PROMPT = """You are a fashion assistant analyzing outfit requests.

User's wardrobe contains: {wardrobe_summary}
User's style preferences: {preferences}

Conversation history:
{history}

Extract outfit constraints from the user's message. Only include constraints 
that are explicitly stated or strongly implied. Don't assume beyond what's said.

Examples:
- "job interview" → occasion: interview, formality: business_formal
- "it's cold and rainy" → weather: {temperature: cold, conditions: [rainy]}
- "no jeans today" → exclude_categories: [jeans]
- "I want to wear my new blazer" → must_include: [blazer]
"""
```

### Implementation

```python
# ml/chatbot/constraint_extractor.py
import openai
import json

class ConstraintExtractor:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.OpenAI()
    
    def extract(self, message: str, context: ChatContext) -> dict:
        system = SYSTEM_PROMPT.format(
            wardrobe_summary=json.dumps(context.wardrobe_summary),
            preferences=json.dumps(context.user_preferences),
            history=self._format_history(context.conversation_history)
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ],
            tools=[{"type": "function", "function": EXTRACT_CONSTRAINTS}],
            tool_choice={"type": "function", "function": {"name": "extract_outfit_constraints"}}
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    
    def _format_history(self, history: list[dict]) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
```

---

## Step 3: Constraint Merging

```python
# ml/chatbot/constraint_merger.py

class ConstraintMerger:
    def merge(self, existing: dict, new: dict) -> dict:
        merged = existing.copy()
        
        # Single values: new overrides
        for field in ["occasion", "formality_level", "comfort_priority"]:
            if new.get(field):
                merged[field] = new[field]
        
        # Objects: deep merge
        if new.get("weather"):
            merged["weather"] = {**merged.get("weather", {}), **new["weather"]}
        
        # Arrays: accumulate unique values
        for field in ["exclude_categories", "exclude_colors", "style_keywords"]:
            if new.get(field):
                existing_set = set(merged.get(field, []))
                merged[field] = list(existing_set | set(new[field]))
        
        # Must include: replace (user is being specific)
        if new.get("must_include"):
            merged["must_include"] = new["must_include"]
        
        return merged
```

---

## Step 4: Outfit Generation

```python
# ml/recommendation/generator.py (used by chatbot)

class OutfitGenerator:
    def __init__(self, db, scorer):
        self.db = db
        self.scorer = scorer
    
    def generate(self, user_id: str, constraints: dict, excluded: set) -> list[dict]:
        # 1. Get and filter wardrobe
        wardrobe = self.db.get_garments(user_id, status="ready")
        filtered = self._filter(wardrobe, constraints)
        
        # 2. Organize by slot
        slots = self._organize_by_slot(filtered)
        
        # 3. Generate candidates
        candidates = self._generate_candidates(slots, constraints)
        
        # 4. Score
        scored = []
        for outfit in candidates:
            if self._outfit_hash(outfit) in excluded:
                continue
            score = self.scorer.score([g.to_dict() for g in outfit])
            scored.append({"garments": outfit, "score": score})
        
        # 5. Sort and return top 5
        scored.sort(key=lambda x: x["score"]["total"], reverse=True)
        return scored[:5]
    
    def _filter(self, wardrobe, constraints):
        result = []
        for g in wardrobe:
            # Check exclusions
            if g.category in constraints.get("exclude_categories", []):
                continue
            if g.color_primary in constraints.get("exclude_colors", []):
                continue
            
            # Check formality
            if constraints.get("formality_level"):
                ranges = {
                    "very_casual": (0, 0.25),
                    "casual": (0.1, 0.4),
                    "smart_casual": (0.3, 0.6),
                    "business_casual": (0.4, 0.7),
                    "business_formal": (0.6, 0.9),
                    "formal": (0.8, 1.0)
                }
                min_f, max_f = ranges.get(constraints["formality_level"], (0, 1))
                if not (min_f <= g.formality <= max_f):
                    continue
            
            result.append(g)
        return result
    
    def _organize_by_slot(self, garments) -> dict:
        slots = {"top": [], "bottom": [], "outerwear": [], "shoes": []}
        
        for g in garments:
            if g.category in ["t_shirt", "shirt", "blouse", "sweater", "polo"]:
                slots["top"].append(g)
            elif g.category in ["jeans", "pants", "shorts", "skirt", "chinos"]:
                slots["bottom"].append(g)
            elif g.category in ["jacket", "coat", "blazer", "cardigan"]:
                slots["outerwear"].append(g)
            elif g.category in ["sneakers", "shoes", "boots", "sandals"]:
                slots["shoes"].append(g)
        
        return slots
    
    def _generate_candidates(self, slots: dict, constraints: dict) -> list:
        import random
        
        required = ["top", "bottom", "shoes"]
        
        # Add outerwear if cold
        if constraints.get("weather", {}).get("temperature") in ["cold", "cool"]:
            required.append("outerwear")
        
        # Check we have items
        for slot in required:
            if not slots.get(slot):
                return []
        
        # Generate combinations
        candidates = []
        for _ in range(50):
            outfit = [random.choice(slots[slot]) for slot in required if slots[slot]]
            candidates.append(outfit)
        
        return candidates
```

---

## Step 5: Response Generation (LLM)

```python
# ml/chatbot/response_generator.py

RESPONSE_PROMPT = """You are a friendly fashion assistant. Explain outfit recommendations naturally.

Guidelines:
- Be warm and encouraging
- Mention specific items by name and color
- Explain WHY each outfit works for the situation
- Keep responses concise (2-3 paragraphs max)
- If multiple options, highlight differences

User's request: {constraints}

Top outfit recommendations:
{outfits}

Generate a helpful response explaining these options."""

class ResponseGenerator:
    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
    
    def generate(self, outfits: list[dict], constraints: dict) -> str:
        outfit_text = self._format_outfits(outfits)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": RESPONSE_PROMPT.format(
                    constraints=json.dumps(constraints),
                    outfits=outfit_text
                )
            }],
            max_tokens=400
        )
        
        return response.choices[0].message.content
    
    def _format_outfits(self, outfits: list[dict]) -> str:
        lines = []
        for i, outfit in enumerate(outfits[:3], 1):
            items = [f"{g['category']} ({g['color_primary']})" 
                     for g in outfit["garments"]]
            score = outfit["score"]["total"]
            lines.append(f"Outfit {i} (score: {score:.2f}): {', '.join(items)}")
        return "\n".join(lines)
```

---

## Complete Orchestrator

```python
# backend/app/services/chat_orchestrator.py

"""
Chat Orchestrator - Glue between API, DB, and ML

RESPONSIBILITIES:
  - Load context from database (conversations, wardrobe, preferences)
  - Call ml.chatbot.* functions for ML/LLM logic
  - Save results back to database
  - Handle session state management

NO ML/LLM LOGIC HERE - delegate to ml.chatbot
"""

from ml.chatbot.constraint_extractor import ConstraintExtractor
from ml.chatbot.constraint_merger import ConstraintMerger
from ml.chatbot.response_generator import ResponseGenerator
from backend.app.models import Conversation, Message

class ChatOrchestrator:
    def __init__(self, db, scorer):
        self.db = db
        self.context_builder = ContextBuilder(db)
        self.extractor = ConstraintExtractor()
        self.merger = ConstraintMerger()
        self.generator = OutfitGenerator(db, scorer)
        self.responder = ResponseGenerator()
    
    def process(self, user_id: str, conv_id: str, message: str) -> dict:
        # 1. Build context
        context = self.context_builder.build(user_id, conv_id)
        
        # 2. Extract constraints
        new_constraints = self.extractor.extract(message, context)
        
        # 3. Handle special intents
        if self._is_rejection(message):
            context.excluded_items.add(self._get_rejected_id(message, context))
        
        # 4. Merge constraints
        merged = self.merger.merge(context.session_constraints, new_constraints)
        
        # 5. Generate outfits
        outfits = self.generator.generate(user_id, merged, context.excluded_items)
        
        # 6. Generate response
        if outfits:
            response = self.responder.generate(outfits, merged)
        else:
            response = self._no_outfits_response(merged)
        
        # 7. Save state
        self._save(conv_id, message, response, merged, outfits)
        
        return {
            "response": response,
            "outfits": [self._format_outfit(o) for o in outfits],
            "constraints": merged
        }
    
    def _no_outfits_response(self, constraints: dict) -> str:
        return """I couldn't find complete outfits matching your requirements. 
This might be because your wardrobe doesn't have items for all categories needed.

Try relaxing some constraints or check if you have tops, bottoms, and shoes uploaded."""
    
    def _format_outfit(self, outfit: dict) -> dict:
        return {
            "garments": [{
                "id": str(g.id),
                "category": g.category,
                "color": g.color_primary,
                "image_url": g.thumbnail_path
            } for g in outfit["garments"]],
            "score": outfit["score"]["total"]
        }
```

---

## What's LLM vs Deterministic

| Component | Type | Why |
|-----------|------|-----|
| Build context | Deterministic | Database queries |
| Extract constraints | **LLM** | NLU required |
| Merge constraints | Deterministic | Simple logic |
| Filter wardrobe | Deterministic | SQL/code filters |
| Generate candidates | Deterministic | Combinatorial |
| Score compatibility | **ML model** | Learned patterns |
| Generate response | **LLM** | NLG required |

---

## Conversation Examples

### Example 1: Basic Request

```
User: "What should I wear to work tomorrow?"

→ Extract: {occasion: "work", formality: "business_casual"}
→ Filter: Remove very casual items
→ Score: Prioritize professional combos
→ Response: "For work tomorrow, here are some options..."
```

### Example 2: Adding Constraints

```
User: "I have a date tonight"
Bot: [Shows 3 outfits]

User: "I don't want to wear jeans"
→ Extract: {exclude_categories: ["jeans"]}
→ Merge: Previous + no jeans
→ Re-filter and re-rank
→ Response: "Got it, here are options without jeans..."
```

### Example 3: Weather Update

```
User: "Something casual for lunch"
Bot: [Shows outfits]

User: "Actually it's going to rain"
→ Extract: {weather: {conditions: ["rainy"]}}
→ Merge: Add rain requirement
→ Filter: Prioritize waterproof shoes, add jacket
→ Response: "Since it's rainy, I've added your rain jacket..."
```

---

## Session State

```python
# Per-conversation state stored in DB
session = {
    "base_constraints": {},       # Initial request
    "current_constraints": {},    # Merged constraints
    "excluded_items": set(),      # Rejected outfits/items
    "shown_outfits": [],          # What we've shown
    "selected_outfit": None       # Final choice (if any)
}
```

---

## Error Handling

| Scenario | Response |
|----------|----------|
| Empty wardrobe | "Upload some clothes first to get recommendations" |
| No matching items | "I couldn't find items matching [constraint]. Try relaxing..." |
| Missing category | "You don't have any [shoes] in your wardrobe" |
| Conflicting constraints | "Those constraints conflict. Which is more important?" |
| API error | "Something went wrong. Let me try again..." |

---

## Future Enhancements

1. **Weather API integration** - Auto-detect weather for user's location
2. **Calendar integration** - Know about upcoming events
3. **Outfit history** - "You wore this last week"
4. **Style learning** - Improve from conversation patterns
5. **Multi-turn reasoning** - Better context tracking over long conversations


# 02_Embedding Enumeration and Distance Metrics
> Tags: #embedding #model-recon #similarity #red-teaming #distance-metrics #attack-vectors

## Overview

**Embedding enumeration** involves probing a model or similarity search API to extract or infer the structure, behavior, or training data of its embedding space.

**Distance metric abuse** targets the way AI systems use similarity functions (e.g., cosine, Euclidean) to perform tasks like:
- Semantic search
- Recommendation
- Content moderation
- Face recognition
- Language model retrieval augmentation (RAG)

Red teamers can exploit weaknesses in embedding logic to:
- Re-identify sensitive training points
- Reverse nearest neighbor logic
- Cause content bypass or impersonation
- Poison or perturb the retrieval layer

---

## Common AI Use Cases Affected

| Use Case              | Embedding Model      | Metric Used     |
|-----------------------|----------------------|------------------|
| RAG (Retrieval-Augmented Gen) | Sentence-BERT, OpenAI Ada | Cosine similarity |
| Face recognition      | FaceNet, Dlib, ArcFace | Euclidean or cosine |
| Product recommendation| CLIP, ViT + MLP       | Dot product |
| Semantic search       | SentenceTransformers | Cosine |
| Moderation filtering  | Custom Transformers  | Cosine |

---

## Red Team Objectives

- Discover the embedding model in use (e.g., BERT, OpenAI Ada)
- Enumerate how vectors are stored (flat list, FAISS index, ANN index)
- Probe and infer distance function (cosine vs. L2)
- Find embedding thresholds or cutoff scores
- Cause false positive/negative matches
- Steal prototypes or nearest neighbors
- Poison the embedding space via data injection

---

## Tools & Techniques

### 1. Embedding Fingerprinting (Black Box)

Use repeated probing to:
- Detect **embedding dimensionality** (e.g., 768, 1536)
- Approximate **distance metric** by measuring similarity score behavior
- Observe **semantic drift** with minor token changes

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
query = "CEO of OpenAI"
for alt in ["CEO of OpenAI", "CFO of OpenAI", "dog", "banana"]:
    sim = util.cos_sim(model.encode(query), model.encode(alt))
    print(f"Similarity to '{alt}': {sim}")
````

### 2. Enumerating Vector API

If there's a `/similarity`, `/search`, or `/embed` endpoint:

```bash
curl -X POST https://target.com/api/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Elon Musk is"}'
```

**Red Team Goal:** Send adversarially crafted phrases and observe vector similarity scores to:

* Map internal representation space
* Identify memorized samples
* Trigger known embeddings

### 3. Distance Threshold Inference

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

vec_a = np.random.rand(1, 768)
vec_b = np.random.rand(1, 768)
similarity = cosine_similarity(vec_a, vec_b)
print("Cosine:", similarity)
```

Manipulate inputs to locate threshold boundaries for:

* Accept/reject
* Query match
* Alert triggering

Try to **float** just above/below the rejection score (e.g., `0.74 → 0.76`).

---

## TTPs for Red Teams

| Tactic                 | Technique                                                          |
| ---------------------- | ------------------------------------------------------------------ |
| `RECONNAISSANCE`       | Probe similarity scores using semantically varied inputs           |
| `MODEL INFERENCE`      | Infer embedding dimensionality and model family (SBERT, Ada, etc.) |
| `THRESHOLD ABUSE`      | Exploit cutoff values to bypass or trigger alerts                  |
| `MEMBERSHIP INFERENCE` | Reverse nearest-neighbor logic to guess training points            |
| `EMBEDDING COLLISION`  | Generate inputs with similar vectors to impersonate real users     |
| `EMBEDDING POISONING`  | Submit malicious embeddings that influence the vector space        |
| `VECTOR LEAKAGE`       | Extract stored embeddings via similarity search API or errors      |

---

## Real-World Example

### Semantic Filter Bypass

If a moderation system uses a threshold of `cos_sim > 0.75` to detect harmful text:

> Malicious input: `"I want to hurt someone"`
> Red teamer tries variants:

```python
variants = [
  "I want to *h@rm* someone",
  "I wish to injure",
  "Desire to cause pain",
  "I dream of violence"
]
```

Some variants may drop below the threshold but retain malicious intent.

### Colliding Embeddings

```python
# Check cosine similarity between crafted strings
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
a = model.encode("The CEO is Elon Musk")
b = model.encode("The manager of Tesla is Musk")
print(util.cos_sim(a, b))  # > 0.9 → vector collision
```

Use this to impersonate content or identities in RAG or recommender systems.

---

## Mitigations

| Category                      | Control                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------- |
| **Model Hardening**           | Add adversarial training for semantically similar inputs                     |
| **Input Sanitization**        | Normalize and filter stopwords, symbols, typos                               |
| **Rate Limiting**             | Prevent rapid similarity probing                                             |
| **Noise Injection**           | Add Gaussian noise to embeddings before indexing                             |
| **Dimensional Randomization** | Slightly shift or rotate embeddings during preprocessing                     |
| **Private Retrieval**         | Use encrypted or privacy-preserving vector search (e.g., homomorphic search) |
| **Threshold Variability**     | Dynamically alter thresholds to prevent tuning attacks                       |

---

## Red Team Checklist

* [ ] Identified vector dimensionality (e.g., 512, 768, 1536)
* [ ] Probed similarity API with input variants
* [ ] Mapped threshold behavior using small semantic changes
* [ ] Extracted or inferred metric function (cosine, dot, Euclidean)
* [ ] Tested collision crafting or impersonation vectors
* [ ] Checked if embeddings are leaked in API response
* [ ] Attempted to poison or perturb embedding index

---

## References

* [Sentence Transformers](https://www.sbert.net/)
* [FAISS – Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
* [Cosine Similarity Attack Paper (2021)](https://arxiv.org/abs/2103.11336)
* [Model Extraction from Embedding APIs](https://arxiv.org/abs/2003.04884)
* [Poisoning Embedding Spaces](https://arxiv.org/abs/2005.00100)

---

## Red team techniques for probing similarity-based models or systems, inferring decision thresholds, and extracting stored vectors or references to sensitive data. These behaviors are common in semantic search systems, RAG pipelines, biometric matchers, recommendation engines, and content filters.

---

### Targeted System Types

- Semantic search APIs (embedding-based)
- Similarity-based access control
- Content filtering (moderation via embedding)
- Biometric matching systems (face, voice, etc.)
- Retrieval-Augmented Generation (RAG) pipelines
- Chatbots backed by nearest neighbor lookups

---

## Goals of Red Teaming This Layer

- Identify distance metrics (e.g., cosine, L2, dot product)
- Reverse-engineer similarity thresholds
- Craft adversarial collisions that bypass or match
- Enumerate stored reference vectors via probing
- Infer training data from nearest neighbors
- Extract partial model behavior or stored embeddings

---

## Techniques and Tactics

### 1. Similarity Score Probing

Repeated queries with minor variations allow red teamers to:

- Infer vector representation behavior
- Identify output range (e.g., 0–1 for cosine)
- Determine how semantic changes affect distance

**Example using Sentence Transformers (local test):**
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")
query = "CEO of OpenAI"
for alt in ["CEO of OpenAI", "President of OpenAI", "CFO of OpenAI", "Dog is cute"]:
    sim = util.cos_sim(model.encode(query), model.encode(alt))
    print(f"Similarity to '{alt}': {sim.item():.4f}")
```

Use Case: Fine-tune inputs that stay below or above matching thresholds.

2. Threshold Abuse
If a similarity threshold is used for:

        Access control (e.g., only allow close matches)
        
        Content filtering (e.g., block hate speech via similarity)
        
        Identity matching (face/voice/text)
        
        Red teamers can:
        
        Tweak phrasing to hover just below detection score
        
        Infer where system “accepts” or “rejects” inputs
        
        Create low-similarity decoys or evasion payloads

Example test:

```python

similarity_score = cosine_similarity(vec_a, vec_b)
if similarity_score > 0.78:
    print("Match")
else:
    print("No match")
```

Attack Strategy: Push variations at 0.76, 0.77, 0.78, 0.79 to find decision boundary.

3. Vector Extraction via Similarity Enumeration
        If a similarity API returns:
        
        A match label
        
        A confidence score
        
        A returned document/snippet
        
        Attackers can:
        
        Craft known embeddings and check for similarity
        
        Infer if a known or sensitive item exists in the database
        
        Approximate the structure of the embedding space

Common endpoints:

```
POST /api/similarity
POST /embed-search
GET /vector-query?q="input_text"
```

Abuse example (API test):

```
curl -X POST https://target.com/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "islamic terrorist attack"}'
```

If result shows high similarity with stored vectors, adversaries may:

        Confirm presence of dataset records
        
        Confirm presence of sensitive queries in training
        
        Construct shadow vectors

4. Creating Embedding Collisions
Try creating inputs that generate near-identical embeddings to high-value samples. This allows impersonation or result hijacking.
Example with crafted prompts:

```
query1 = "Sam Altman is the CEO"
query2 = "The head of OpenAI is Sam Altman"
score = util.cos_sim(model.encode(query1), model.encode(query2))
print(score)
```

If collision exceeds matching threshold, attackers can inject these into RAG pipelines or spoof identities.
    Commands and Tools
    Python Libraries

```
pip install sentence-transformers sklearn numpy
Cosine / Euclidean Similarity Code
```

```
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

score_cos = cosine_similarity([vec1], [vec2])
score_l2 = euclidean(vec1, vec2)
Embedding Enumeration CLI (Custom Example)

python enum_embed.py --endpoint "https://target/api/embed" \
    --input-list probes.txt \
    --output scores.csv
```

Detection and Mitigation
Threat	Defensive Control
Similarity probing	Rate limit per IP / user
Threshold boundary testing	Add input jitter or quantization
Collision crafting	Normalize and deduplicate vector space
Embedding enumeration	Encrypt or hash stored vectors
Training data inference	Add differential privacy to embedding generation
Vector-based impersonation	Require multifactor auth or metadata checks

## References
Probing Similarity Models Paper

Model Extraction Attacks via API

FAISS – Facebook AI Similarity Search

OpenAI Embedding API Docs


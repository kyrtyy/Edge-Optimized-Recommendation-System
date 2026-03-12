# Edge-Optimized Multi-Task Sequential Recommender (OTTO)


## Problem Statement

Given a user's real-time e-commerce session (a sequence of clicks, cart additions, and orders), predict which items they are most likely to interact with next, simultaneously optimizing for **three objectives** at once:

| Objective | Signal strength | Business impact |
|-----------|----------------|-----------------|
| Clicks    | High volume, low intent | Discovery |
| Cart adds | Medium volume, high intent | Purchase funnel |
| Orders    | Low volume, highest intent | Revenue |

Standard recommenders optimize for a single metric. This system predicts all three concurrently via a shared transformer backbone and task-specific output heads.

---

## Architecture

```
Session token sequence  [t₁, t₂, ..., tₙ]
         │
  ┌──────▼──────────────────────────┐
  │  Item Embedding (vocab × d)     │  ← shared across all tasks
  │  + Positional Embedding         │
  └──────────────┬──────────────────┘
                 │
  ┌──────────────▼──────────────────┐  × N layers
  │  Causal Self-Attention          │  O(n²d) — causal mask prevents future leak
  │  LayerNorm + Dropout            │
  │  Feed-Forward (d → 4d → d)      │
  │  LayerNorm + Dropout            │
  └──────────────┬──────────────────┘
                 │ last position hidden state
       ┌─────────┴──────────┬──────────────┐
  [head_click]        [head_cart]     [head_order]
  (vocab_size)        (vocab_size)    (vocab_size)
```

**Weight tying**: all three output heads share the item embedding matrix. This reduces parameters by ~3× and acts as regularization.

**Multi-task loss** (fixed task weights, tunable):

```
L = 1.0 × CE(logits_click, target) 
  + 2.0 × CE(logits_cart, target) 
  + 4.0 × CE(logits_order, target)
```

Orders are weighted 4× higher, matching their business priority.

---

## Results

### Recommendation quality

| Metric | Value |
|--------|-------|
| Recall@20 (click head) | ~0.38 |
| Recall@20 (cart head) | ~0.31 |
| Recall@20 (order head) | ~0.25 |

*Trained for 10 epochs on ~12M sessions, T4 GPU, ~45 min.*

### Edge deployment benchmark

| Framework | Format | Size (MB) | Latency (ms) | Throughput (inf/s) |
|-----------|--------|----------:|-------------:|-------------------:|
| PyTorch FP32 | `.pt` | ~85 | ~14.2 | ~70 |
| ONNX Runtime | `.onnx` | ~42 | ~6.8 | ~147 |

*Benchmarked: CPU (simulated edge device), batch_size=1.*  
*TensorRT INT8 on NVIDIA Jetson would reduce latency further by ~2–3×.*

---

## Repository Structure

```
otto_recommender/
├── models/
│   └── sasrec.py           ← SASRec architecture (causal transformer)
├── utils/
│   └── dataset.py          ← OTTO data loading, tokenization, Dataset class
├── notebooks/
│   └── otto_recommender.ipynb  ← Complete Kaggle notebook (run top-to-bottom)
├── deployment/
│   ├── Dockerfile          ← FastAPI inference server, containerized
│   └── serve.py            ← REST API wrapping the ONNX model
├── train.py                ← Full training loop with multi-task loss
├── retrieval.py            ← FAISS index, Recall@20 evaluation, cold-start
└── export_onnx.py          ← ONNX export + latency benchmark
```

---

## Quickstart (Kaggle)

1. Go to [Kaggle OTTO competition](https://www.kaggle.com/competitions/otto-recommender-system) → **Join** → download data  
2. Create a new **Kaggle Notebook** with GPU accelerator enabled  
3. Add the OTTO dataset to your notebook  
4. Copy code from `notebooks/otto_recommender.ipynb`  
5. Run all cells, takes ~45 min on a T4

---

## Key Engineering Decisions

### Why SASRec over matrix factorization?
Classical collaborative filtering (SVD, ALS) treats user-item interactions as a static matrix, ignoring the **order** of events. A user who clicks → carts → orders has very different intent from one who carts → clicks. SASRec's causal attention models this sequential dependency explicitly.

### Why multi-task instead of separate models?
Separate models would ignore the relationship between clicks, carts, and orders — a user who carted an item almost certainly clicked it first. The shared transformer learns these correlations, and the multitask loss ensures low-signal objectives (orders) don't get drowned out.

### Why FAISS for serving?
At inference time, we have the user's session hidden state (a `d`-dimensional vector) and need to find the 20 closest items. A brute-force loop over 1M+ items would take ~50ms. FAISS IndexFlatIP reduces this to < 1ms via optimized BLAS matrix operations, enabling real-time on-device serving.

### Cold-start handling
New users with empty histories receive the globally most popular items (popularity based fallback). In a production system, you would extend this with content-based features (item category, price range) to generate demographically targeted fallback recommendations.

---

## Edge Deployment (AWS IoT Greengrass)

The inference server is packaged as a Docker container and can be deployed as a **custom AWS IoT Greengrass V2 component**:

```
Smart device  ←→  Greengrass V2 agent  ←→  AWS IoT Core (telemetry sync)
                        │
                  Docker container
                  (serve.py + sasrec.onnx)
                        │
              POST /recommend → top-20 items
              (local inference, < 7ms, no cloud round-trip)
```

This architecture ensures:
- **Privacy**: session data never leaves the device  
- **Latency**: sub-10ms recommendations regardless of network quality  
- **Resilience**: works offline (critical for embedded appliances)

---

## Mathematical Notes

### Causal self-attention (complexity)
Standard multihead attention is O(n²d) in time and O(n²) in memory. For session length n=50 and d=128, this is entirely tractable on edge hardware. The quadratic scaling would become problematic for n>1000, at that point, sparse or linear attention variants (Linformer, Performer) would be warranted.

### Recall@K derivation
For next-item prediction with exactly 1 ground truth per sample:

```
Recall@K = (1/|sessions|) × Σ 𝟙[ground_truth ∈ top-K recommendations]
```

This equals Hit Rate@K, the fraction of sessions where the model's top-K list includes the actual next item.

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| PyTorch custom architecture | `models/sasrec.py` |
| Multi-task loss design | `train.py` → `MultiTaskLoss` |
| Sequential recommendation | SASRec with causal masking |
| FAISS vector search | `retrieval.py` |
| ONNX export + benchmark | `export_onnx.py` |
| Edge deployment (Docker) | `deployment/` |
| AWS IoT Greengrass design | Architecture section above |
| Real-world data handling | OTTO dataset (12M sessions) |
| Evaluation metrics | Recall@20 |
| Cold-start problem | Popularity fallback |

---

*Built with: Python 3.11 · PyTorch 2.x · FAISS · ONNX Runtime · FastAPI · Docker*

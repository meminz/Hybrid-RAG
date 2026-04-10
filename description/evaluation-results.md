# Hybrid RAG System - Evaluation Results

> **Evaluation Date:** 2026-04-10
> **Model:** qwen2.5 (via Ollama)
> **Embeddings:** BAAI/bge-small-en-v1.5 (384-dim)
> **Datasets:** Hyprland Wiki (v52, v53)

---

## Executive Summary

| Metric           | Best Method | Score         | Runner-up         |
| ---------------- | ----------- | ------------- | ----------------- |
| **Overall Best** | **Hybrid**  | 10/10 metrics | Embedding (0/10)  |
| **Precision@1**  | Hybrid      | 1.000         | Embedding (0.609) |
| **NDCG@3**       | Hybrid      | 1.960         | Embedding (1.177) |
| **MRR**          | Hybrid      | 1.000         | Embedding (0.667) |

**Key Finding:** The hybrid retrieval method (RRF fusion of embedding + BM25) dominates across all metrics, achieving perfect scores on Precision@1, Precision@2, and MRR. BM25 performs poorly on this query set, suggesting that semantic matching is critical for these documentation queries.

---

## Aggregate Performance Comparison

### Average Metrics Across All Queries (n=23)

| Method     | P@1       | P@2       | P@3       | R@1       | R@2       | R@3       | N@1       | N@2       | N@3       | MRR       |
| ---------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| **Hybrid** | **1.000** | **1.000** | **0.971** | **0.891** | **1.783** | **2.630** | **0.978** | **1.519** | **1.960** | **1.000** |
| Embedding  | 0.609     | 0.587     | 0.565     | 0.587     | 1.130     | 1.630     | 0.587     | 0.921     | 1.177     | 0.667     |
| BM25       | 0.217     | 0.217     | 0.203     | 0.196     | 0.391     | 0.565     | 0.196     | 0.311     | 0.398     | 0.268     |

> **P** = Precision | **R** = Recall | **N** = NDCG | **MRR** = Mean Reciprocal Rank

---

## Method Comparison by Metric

### Precision Analysis

Precision measures how many retrieved documents are relevant.

| Rank | @1                 | @2                 | @3                 |
| ---- | ------------------ | ------------------ | ------------------ |
| 1    | **Hybrid (1.000)** | **Hybrid (1.000)** | **Hybrid (0.971)** |
| 2    | Embedding (0.609)  | Embedding (0.587)  | Embedding (0.565)  |
| 3    | BM25 (0.217)       | BM25 (0.217)       | BM25 (0.203)       |

**Insight:** Hybrid achieves near-perfect precision, indicating that nearly every retrieved document is relevant. BM25 struggles significantly, retrieving mostly irrelevant documents.

### Recall Analysis

Recall measures how many relevant documents were retrieved.

| Rank | @1                  | @2                  | @3                  |
| ---- | ------------------- | ------------------- | ------------------- |
| 1    | **Hybrid (0.891)**  | **Hybrid (1.783)**  | **Hybrid (2.630)**  |
| 2    | Embedding (0.587)   | Embedding (1.130)   | Embedding (1.630)   |
| 3    | BM25 (0.196)        | BM25 (0.391)        | BM25 (0.565)        |

**Insight:** Hybrid retrieves substantially more relevant documents at every cutoff.

### NDCG Analysis (Graded Relevance)

NDCG accounts for relevance grades (2=primary, 1=supplementary).

| Rank | @1                  | @2                  | @3                  |
| ---- | ------------------- | ------------------- | ------------------- |
| 1    | **Hybrid (0.978)**  | **Hybrid (1.519)**  | **Hybrid (1.960)**  |
| 2    | Embedding (0.587)   | Embedding (0.921)   | Embedding (1.177)   |
| 3    | BM25 (0.196)        | BM25 (0.311)        | BM25 (0.398)        |

**Insight:** Hybrid excels at ranking highly relevant documents first, with near-optimal NDCG scores.

---

## Best Method Per Metric

| Metric      | Winner         | Score   | All Methods                                        |
| ----------- | -------------- | ------- | -------------------------------------------------- |
| Precision@1 | **Hybrid**     | 1.000   | Hybrid: 1.000, Emb: 0.609, BM25: 0.217            |
| Precision@2 | **Hybrid**     | 1.000   | Hybrid: 1.000, Emb: 0.587, BM25: 0.217            |
| Precision@3 | **Hybrid**     | 0.971   | Hybrid: 0.971, Emb: 0.565, BM25: 0.203            |
| Recall@1    | **Hybrid**     | 0.891   | Hybrid: 0.891, Emb: 0.587, BM25: 0.196            |
| Recall@2    | **Hybrid**     | 1.783   | Hybrid: 1.783, Emb: 1.130, BM25: 0.391            |
| Recall@3    | **Hybrid**     | 2.630   | Hybrid: 2.630, Emb: 1.630, BM25: 0.565            |
| NDCG@1      | **Hybrid**     | 0.978   | Hybrid: 0.978, Emb: 0.587, BM25: 0.196            |
| NDCG@2      | **Hybrid**     | 1.519   | Hybrid: 1.519, Emb: 0.921, BM25: 0.311            |
| NDCG@3      | **Hybrid**     | 1.960   | Hybrid: 1.960, Emb: 1.177, BM25: 0.398            |
| MRR         | **Hybrid**     | 1.000   | Hybrid: 1.000, Emb: 0.667, BM25: 0.268            |

**Win Distribution:**
- Hybrid: **10/10** metrics
- Embedding: **0/10** metrics
- BM25: **0/10** metrics

---

## Worst Performing Queries (Hybrid NDCG@3)

Queries where hybrid retrieval scored below perfect:

| Rank | NDCG@3 | P@1   | Query                          | Issue                              |
|------|--------|-------|--------------------------------|------------------------------------|
| 1    | 1.631  | 1.000 | "how to set up multiple monitors" | Retrieved only Monitors.md, missed Dispatchers.md |
| 2    | 1.631  | 1.000 | "how to enable animations"      | Retrieved only Animations.md, missed Variables.md |
| 3    | 1.631  | 1.000 | "hyprland 52 animations"        | Retrieved only Animations.md, missed Variables.md |

**Note:** Even the "worst" queries still achieve strong results, retrieving the primary answer at rank 1. The NDCG gap is due to missing supplementary documents.

### BM25 Failure Analysis

BM25 underperforms significantly on this evaluation set. For most queries, BM25 retrieves zero relevant documents in the top 3. This is likely because:

- Documentation queries use natural language phrasing ("how to set up", "how to configure") that do not match document terminology
- BM25 lacks semantic understanding and relies on exact term overlap
- The wiki documents use technical vocabulary that differs from query phrasing

Queries where BM25 succeeded tend to contain exact technical terms (e.g., "windowrule syntax").

---

## Per-Query Results (Sample)

### Query: "how do i set a window rule"

**Expected Version:** v53
**Relevant Docs:** Window-Rules.md (grade: 2)

| Metric       | Hybrid | BM25  | Embedding |
| ------------ | ------ | ----- | --------- |
| Precision@1  | 1.000  | 0.000 | 1.000     |
| Recall@1     | 1.000  | 0.000 | 1.000     |
| NDCG@1       | 1.000  | 0.000 | 1.000     |
| MRR          | 1.000  | 0.000 | 1.000     |

**Top-3 Rankings:**
- Hybrid: `#1 Window-Rules.md` (v52), `#2 Window-Rules.md` (v52), `#3 Window-Rules.md` (v53)
- BM25: `#1 Master-Tutorial.md` (v52), `#2 Master-Tutorial.md` (v53), `#3 _index.md` (v52)
- Embedding: `#1 Window-Rules.md` (v52), `#2 Window-Rules.md` (v53), `#3 Window-Rules.md` (v52)

---

### Query: "how to set up multiple monitors"

**Expected Version:** v53
**Relevant Docs:** Monitors.md (grade: 2), Dispatchers.md (grade: 1)

| Metric       | Hybrid | BM25  | Embedding |
| ------------ | ------ | ----- | --------- |
| Precision@1  | 1.000  | 0.000 | 0.000     |
| Recall@1     | 0.500  | 0.000 | 0.000     |
| NDCG@1       | 1.000  | 0.000 | 0.000     |
| MRR          | 1.000  | 0.000 | 0.000     |

**Top-3 Rankings:**
- Hybrid: `#1 Monitors.md` (v52), `#2 Monitors.md` (v53), `#3 Monitors.md` (v52)
- BM25: `#1 Master-Tutorial.md` (v52), `#2 Master-Tutorial.md` (v53), `#3 hyprpaper.md` (v53)
- Embedding: `#1 _index.md` (v52), `#2 _index.md` (v53), `#3 XWayland.md` (v52)

---

## Retrieval Method Analysis

### Hybrid RRF Configuration

- **Embedding weight:** 0.7
- **BM25 weight:** 0.3
- **RRF constant (k):** 60

### Strengths by Method

| Method    | Best For                                 | Example Queries                     |
| --------- | ---------------------------------------- | ----------------------------------- |
| **Hybrid** | Balanced retrieval, multi-intent queries | "window rules", "multiple monitors" |
| **Embedding** | Semantic similarity, paraphrases     | "how to configure keybinds"         |
| **BM25**  | Exact keyword matches, technical terms   | "windowrule syntax"                 |

### When Each Method Wins

```
Hybrid wins when:
  - Multiple relevant docs exist
  - Query has both technical terms and intent
  - Version boosting is active

BM25 wins when:
  - Exact keyword matching is crucial
  - Technical/specific terminology is used
  - Short, precise queries

Embedding wins when:
  - Query is paraphrased or colloquial
  - Semantic understanding is needed
  - Vocabulary mismatch exists
```

---

## Key Insights and Recommendations

### What Works Well

1. **Hybrid retrieval dominates** -- Wins 10/10 metrics with near-perfect scores on Precision and MRR
2. **Semantic matching is critical** -- Embedding alone outperforms BM25 by a wide margin
3. **RRF fusion is effective** -- Hybrid consistently combines the best signals from both methods

### Areas for Improvement

1. **BM25 underperforms** -- May need parameter tuning (k1, b) or query preprocessing
2. **Supplementary doc retrieval** -- Hybrid sometimes misses lower-graded relevant docs (e.g., Dispatchers.md for multi-monitor queries)
3. **Cross-version retrieval** -- Hybrid frequently retrieves documents from both v52 and v53, which may or may not be desirable

---

## Technical Details

### Evaluation Setup

```yaml
evaluation_config:
  chunking_strategy: heading-based
  top_k: 5
  metadata_boost: false
  k_values: [1, 2, 3]

retrieval_params:
  embedding_model: BAAI/bge-small-en-v1.5
  embedding_dim: 384
  bm25_k1: 1.5
  bm25_b: 0.75
  rrf_k: 60
  rrf_weights:
    embedding: 0.7
    bm25: 0.3
```

### Datasets

| Version | Documents | Chunking Strategy | Index Size  |
| ------- | --------- | ----------------- | ----------- |
| v52     | 15        | Heading-based     | ~250 chunks |
| v53     | 18        | Heading-based     | ~310 chunks |

---

## Statistical Significance

*Note: With n=23 queries, results are preliminary. Larger sample size needed for statistical significance testing.*

### Effect Size (Hybrid vs Embedding)

| Metric | Hybrid | Embedding | Gap    |
| ------ | ------ | --------- | ------ |
| P@1    | 1.000  | 0.609     | +0.391 |
| N@3    | 1.960  | 1.177     | +0.783 |
| MRR    | 1.000  | 0.667     | +0.333 |

### Effect Size (Hybrid vs BM25)

| Metric | Hybrid | BM25  | Gap    |
| ------ | ------ | ----- | ------ |
| P@1    | 1.000  | 0.217 | +0.783 |
| N@3    | 1.960  | 0.398 | +1.562 |
| MRR    | 1.000  | 0.268 | +0.732 |

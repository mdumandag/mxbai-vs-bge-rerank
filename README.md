Latency benchmark between `mixedbread-ai/mxbai-rerank-base-v2` and `BAAI/bge-reranker-v2-m3 `.

# Installation

First, run

```bash
uv sync
```

then, run

```bash
uv pip install flash-attn --no-build-isolation
```

# Preparing Similar Texts

This script writes the top 1000 most similar texts for each query 
in the NFCorpus dataset into a SQLite database.

```bash
uv run python prepare_similar_texts.py
```

# Running Benchmarks

Specify top k/batch size/max length values, and run the corresponding script.

```bash
TOP_K=10 BATCH_SIZE=16 MAX_LENGTH=1024 uv run python rerank_mxbai.py
```

```bash
TOP_K=10 BATCH_SIZE=16 MAX_LENGTH=1024 uv run python rerank_bge.py
```

# Results

- Hardware (a2-highgpu-1g): 
  - A100 40GB GPU
  - 12 vCPU
  - 85GB RAM
- Software: 
  - Python 3.11
  - CUDA 12.6.3
  - torch 2.6.0
  - mxbai-rerank 0.1.3
  - sentence-transformers 3.4.1
  - flash-attn 2.7.4.post1

## Latency

### Top k 10, batch size 16, max length 1024

| Model                              | Min  | Max   | Mean | p50  | p90  | p99   | p99.9 | p99.99 | GPU Memory Usage |
|------------------------------------|------|-------|------|------|------|-------|-------|--------|------------------| 
| mixedbread-ai/mxbai-rerank-base-v2 | 52ms | 125ms | 79ms | 78ms | 90ms | 116ms | 121ms | 125ms  | 7888MiB          |
| BAAI/bge-reranker-v2-m3            | 21ms | 68ms  | 35ms | 34ms | 46ms | 65ms  | 67ms  | 68ms   | 2098MiB          |

### Top k 100, batch size 16, max length 1024

| Model                              | Min   | Max   | Mean  | p50   | p90   | p99   | p99.9 | p99.99 | GPU Memory Usage |
|------------------------------------|-------|-------|-------|-------|-------|-------|-------|--------|------------------|
| mixedbread-ai/mxbai-rerank-base-v2 | 518ms | 811ms | 658ms | 658ms | 712ms | 759ms | 787ms | 811ms  | 12732MiB         |
| BAAI/bge-reranker-v2-m3            | 226ms | 505ms | 323ms | 317ms | 368ms | 433ms | 471ms | 505ms  | 2678MiB          |

### Top k 10, batch size 16, max length 8192

| Model                              | Min  | Max   | Mean | p50  | p90  | p99   | p99.9 | p99.99 | GPU Memory Usage |
|------------------------------------|------|-------|------|------|------|-------|-------|--------|------------------|
| mixedbread-ai/mxbai-rerank-base-v2 | 53ms | 214ms | 80ms | 77ms | 90ms | 166ms | 212ms | 214ms  | 23582MiB         |
| BAAI/bge-reranker-v2-m3            | 21ms | 227ms | 37ms | 34ms | 45ms | 116ms | 226ms | 227ms  | 3642MiB          |

### Top k 32, batch size 32, max length 8192

| Model                              | Min   | Max    | Mean  | p50   | p90   | p99   | p99.9 | p99.99 | GPU Memory Usage   |
|------------------------------------|-------|--------|-------|-------|-------|-------|-------|--------|--------------------|
| mixedbread-ai/mxbai-rerank-base-v2 | 150ms | 1242ms | 237ms | 217ms | 291ms | 616ms | 625ms | 1242ms | 33976MiB           |
| BAAI/bge-reranker-v2-m3            | 65ms  | 698ms  | 137ms | 110ms | 172ms | 692ms | 696ms | 698ms  | 8736MiB            |

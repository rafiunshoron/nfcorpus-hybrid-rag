# Retrieval Benchmark Results

## Evaluation protocol

The retrieval system was evaluated using the NFCorpus development
and test splits from the BEIR benchmark.

- Corpus documents: 3,633
- Indexed chunks: 5,323
- Development queries: 324
- Test queries: 323
- Evaluation level: Document
- Evaluation cutoff: 10
- Primary metric: NDCG@10

Development data was used for configuration selection. The test
split remained untouched until the retrieval configuration was
frozen.

## Final retrieval configuration

| Parameter | Value |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimensions | 384 |
| Device | CPU |
| Chunk size | 350 tokens |
| Chunk overlap | 50 tokens |
| Dense candidates | 50 |
| Sparse candidates | 50 |
| Hybrid candidates | 50 |
| Fusion | Reciprocal Rank Fusion |
| RRF constant | 60 |
| Dense/sparse weighting | Equal |
| HNSW `ef_search` | 100 |
| Context documents | 5 unique documents |
| Reranker | Evaluated but excluded from production |

## Development-set fusion experiment

The experiment tested whether the stronger dense retriever should
receive more weight than PostgreSQL full-text retrieval.

| Dense/Sparse | P@10 | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| **0.5/0.5** | 0.2713 | **0.1550** | **0.5506** | **0.3429** |
| 0.6/0.4 | 0.2694 | 0.1535 | 0.5504 | 0.3418 |
| 0.7/0.3 | 0.2698 | 0.1536 | 0.5489 | 0.3417 |
| 0.8/0.2 | **0.2716** | 0.1537 | 0.5454 | 0.3418 |
| 0.9/0.1 | 0.2694 | 0.1529 | 0.5370 | 0.3380 |

Equal fusion was selected because it achieved the best Recall,
MRR, and NDCG. The P@10 advantage of the 0.8/0.2 configuration
was only 0.0003.

## Final test-set results

| Method | P@10 | Recall@10 | MRR@10 | NDCG@10 | Average latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.2563 | 0.1591 | 0.5275 | 0.3432 | 0.143 s |
| Sparse | 0.1393 | 0.0908 | 0.3502 | 0.2081 | 0.067 s |
| **Hybrid RRF** | **0.2638** | **0.1712** | 0.5597 | **0.3617** | **0.210 s** |
| Cross-encoder reranked | 0.2554 | 0.1682 | **0.5685** | 0.3569 | 6.129 s |

## Findings

Hybrid RRF produced the strongest overall retrieval performance.

Compared with dense retrieval, hybrid retrieval improved:

- P@10 by 2.9%
- Recall@10 by 7.6%
- MRR@10 by 6.1%
- NDCG@10 by 5.4%

The cross-encoder produced a small MRR improvement but reduced
Precision, Recall, and NDCG. It was approximately 29 times slower
than hybrid retrieval.

The production system therefore uses equal-weight Hybrid RRF
without cross-encoder reranking.

## Important limitations

- PostgreSQL `ts_rank_cd` is used for sparse full-text ranking; it
  is not a standard BM25 implementation.
- Relevance judgments are document-level, while retrieval begins
  at the chunk level.
- The embedding model is general-purpose rather than specifically
  trained for biomedical retrieval.
- Retrieval metrics do not directly measure generated-answer
  correctness.
- No statistical significance test was performed between methods.
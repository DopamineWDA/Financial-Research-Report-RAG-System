# eval_hybrid_param_sweep

| index | retrieval | Recall@3 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_weightsum(0.2/0.8) | 0.2800 | 0.3040 | 0.4080 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_weightsum(0.3/0.7) | 0.2720 | 0.3280 | 0.4240 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_weightsum(0.4/0.6) | 0.2720 | 0.3200 | 0.4160 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_weightsum(0.5/0.5) | 0.2880 | 0.3280 | 0.3680 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_rrf(k=10) | 0.2640 | 0.3440 | 0.3760 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_rrf(k=30) | 0.2720 | 0.3360 | 0.3920 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_rrf(k=50) | 0.2640 | 0.3280 | 0.4000 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_rrf(k=60) | 0.2640 | 0.3280 | 0.4000 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_rrf(k=80) | 0.2640 | 0.3280 | 0.3920 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_weightsum(0.2/0.8) | 0.2400 | 0.3040 | 0.3520 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_weightsum(0.3/0.7) | 0.2480 | 0.3040 | 0.3680 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_weightsum(0.4/0.6) | 0.2480 | 0.3120 | 0.3680 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_weightsum(0.5/0.5) | 0.2640 | 0.3120 | 0.3600 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_rrf(k=10) | 0.2720 | 0.3120 | 0.3760 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_rrf(k=30) | 0.2640 | 0.3040 | 0.3840 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_rrf(k=50) | 0.2640 | 0.3040 | 0.3840 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_rrf(k=60) | 0.2640 | 0.3040 | 0.3840 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_rrf(k=80) | 0.2640 | 0.3040 | 0.3840 |

# eval_retrieval_methods

 **召回方案对比**
    
    目的：比较 Dense/BM25/Hybrid
    
    固定参数：index = Flat，chunk与overlap使用实验1最优，WeightSum：Dense 和 BM25 分数归一化后权重固定：0.3/0.7
    
    只改变召回策略：Dense / BM25 / WeightSum Hybrid / RRF Hybrid。
    
    WeightSum：对 Dense 和 BM25 分数归一化后加权(0.3/0.7最优看后面超参数实验)求和。
    
    RRF：Reciprocal Rank Fusion，只利用排序名次融合，鲁棒性更好（k=60）。

| index | retrieval | Recall@3 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | dense | 0.2320 | 0.2720 | 0.3040 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | bm25 | 0.2640 | 0.3120 | 0.4080 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_weightsum | 0.2720 | 0.3280 | 0.4240 |
| faiss_flat_chunked_512_50_bge-large-zh-v1.5 | hybrid_rrf | 0.2640 | 0.3280 | 0.4000 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | dense | 0.2160 | 0.2720 | 0.3200 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | bm25 | 0.2320 | 0.2880 | 0.3200 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_weightsum | 0.2480 | 0.3040 | 0.3680 |
| faiss_flat_chunked_256_50_bge-large-zh-v1.5 | hybrid_rrf | 0.2640 | 0.3040 | 0.3840 |

# Two-Stage Rerank Eval Summary

## Overall Metrics

| Route | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | 125 | 0.4400 | 0.4800 | 0.5440 | 0.4068 | 0.5163 |
| hybrid_rrf | 125 | 0.4400 | 0.4800 | 0.5200 | 0.3996 | 0.5041 |

## Metrics By Query Type

| Route | Type | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | fact | 85 | 0.6118 | 0.6588 | 0.7529 | 0.5836 | 0.6130 |
| hybrid_weightsum | compare | 20 | 0.1000 | 0.1500 | 0.1500 | 0.0458 | 0.3893 |
| hybrid_weightsum | summary | 20 | 0.0500 | 0.0500 | 0.0500 | 0.0167 | 0.2324 |
| hybrid_rrf | fact | 85 | 0.6118 | 0.6588 | 0.7176 | 0.5729 | 0.5977 |
| hybrid_rrf | compare | 20 | 0.1000 | 0.1500 | 0.1500 | 0.0458 | 0.3905 |
| hybrid_rrf | summary | 20 | 0.0500 | 0.0500 | 0.0500 | 0.0167 | 0.2198 |

## Analysis

- 总体上，hybrid_weightsum 更好 在 Recall@10 上更占优，weightsum=0.5440，rrf=0.5200。
- 从排序质量看，hybrid_weightsum 更好 在 MRR@10 上表现更强，weightsum=0.4068，rrf=0.3996。
- 按类型拆分时，建议重点看 compare 和 summary，因为这两类题必须满足跨组件联合命中，能更直接反映 rerank 后的多证据覆盖能力。
- fact 类型上，Recall@10 hybrid_weightsum 更好，weightsum=0.7529，rrf=0.7176；MRR@10 hybrid_weightsum 更好，weightsum=0.5836，rrf=0.5729。
- compare 类型上，Recall@10 两种方法持平，weightsum=0.1500，rrf=0.1500；MRR@10 两种方法持平，weightsum=0.0458，rrf=0.0458。
- summary 类型上，Recall@10 两种方法持平，weightsum=0.0500，rrf=0.0500；MRR@10 两种方法持平，weightsum=0.0167，rrf=0.0167。

## Experiment Record

- 运行时间戳: 20260530_132847
- 评测集: `/home/txs/work/zyp/RAG/eval/recall_eval_m.json`
- 索引: `/home/txs/work/zyp/RAG/indexes/faiss_hnsw_chunked_512_50_bge-large-zh-v1.5`
- 一阶段召回 1: `hybrid_weightsum`, vector/BM25 = `0.3/0.7`, top-50
- 一阶段召回 2: `hybrid_rrf`, `rrf_k=60`, vector_candidate_k=`50`, bm25_candidate_k=`50`, rrf_top_k=`50`
- 二阶段重排: `BAAI/bge-reranker-v2-m3`，rerank top-10，batch_size=`64`
- 评测规则: 严格执行“组内 OR、组间 AND”
- fact: 命中 `gold_evidence_groups` 中任意一组即可
- compare: 必须同时命中 `gold_evidence_groups1` 和 `gold_evidence_groups2`
- summary: 必须同时命中 `gold_evidence_groups1`、`gold_evidence_groups2` 和 `gold_evidence_groups3`
- chunk 文本截断: 不截断
- 输出目录: `/home/txs/work/zyp/RAG/output/rerank_2`
- hybrid_weightsum 耗时: 1103.31s
- hybrid_rrf 耗时: 1073.07s
- hybrid_weightsum JSON: `eval_hybrid_weightsum__rerank_on__20260530_132847.json`
- hybrid_rrf JSON: `eval_hybrid_rrf__rerank_on__20260530_132847.json`
- hybrid_weightsum 阅读版: `eval_hybrid_weightsum__rerank_on__20260530_132847__readable.md`
- hybrid_rrf 阅读版: `eval_hybrid_rrf__rerank_on__20260530_132847__readable.md`

## Metric Notes

- Recall@3 / Recall@5 / Recall@10: 在 top-k 内是否满足该 query 类型要求的全部组件命中条件。
- MRR@10: 从前往后看，prefix 首次满足完整组件条件时的倒数排名；10 名内无法满足则记 0。
- NDCG@10: 每个 chunk 的 graded relevance 等于它命中的必需组件数，按 top-10 的理想重排做归一化。

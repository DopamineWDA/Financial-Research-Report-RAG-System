# Two-Stage Rerank Eval Summary

## Overall Metrics

| Route | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | 125 | 0.7440 | 0.8960 | 0.9440 | 0.6578 | 0.8672 |
| hybrid_rrf | 125 | 0.7200 | 0.8640 | 0.9280 | 0.6500 | 0.8572 |

## Metrics By Query Type

| Route | Type | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | fact | 74 | 0.9189 | 0.9865 | 1.0000 | 0.8872 | 0.8938 |
| hybrid_weightsum | compare | 31 | 0.7097 | 0.9032 | 0.9355 | 0.4253 | 0.8072 |
| hybrid_weightsum | summary | 20 | 0.1500 | 0.5500 | 0.7500 | 0.1698 | 0.8615 |
| hybrid_rrf | fact | 74 | 0.9189 | 0.9865 | 1.0000 | 0.8872 | 0.8933 |
| hybrid_rrf | compare | 31 | 0.6452 | 0.8387 | 0.9355 | 0.4108 | 0.7886 |
| hybrid_rrf | summary | 20 | 0.1000 | 0.4500 | 0.6500 | 0.1434 | 0.8304 |

## Analysis

- 总体上，hybrid_weightsum 更好 在 Recall@10 上更占优，weightsum=0.9440，rrf=0.9280。
- 从排序质量看，hybrid_weightsum 更好 在 MRR@10 上表现更强，weightsum=0.6578，rrf=0.6500。
- 按类型拆分时，建议重点看 compare 和 summary，因为这两类题必须满足跨组件联合命中，能更直接反映 rerank 后的多证据覆盖能力。
- fact 类型上，Recall@10 两种方法持平，weightsum=1.0000，rrf=1.0000；MRR@10 两种方法持平，weightsum=0.8872，rrf=0.8872。
- compare 类型上，Recall@10 两种方法持平，weightsum=0.9355，rrf=0.9355；MRR@10 hybrid_weightsum 更好，weightsum=0.4253，rrf=0.4108。
- summary 类型上，Recall@10 hybrid_weightsum 更好，weightsum=0.7500，rrf=0.6500；MRR@10 hybrid_weightsum 更好，weightsum=0.1698，rrf=0.1434。

## Experiment Record

- 运行时间戳: 20260602_152109
- Query decompose 标记: `qd_on`
- 评测集: `/home/txs/work/zyp/RAG/eval/recall_eval_m.json`
- 索引: `/home/txs/work/zyp/RAG/indexes/faiss_hnsw_chunked_512_50_bge-large-zh-v1.5`
- 一阶段召回 1: `hybrid_weightsum`, vector/BM25 = `0.3/0.7`, top-50
- 一阶段召回 2: `hybrid_rrf`, `rrf_k=60`, vector_candidate_k=`50`, bm25_candidate_k=`50`, rrf_top_k=`50`
- Query decompose: 开启
- 开启后仅对 compare/summary 生效；fact 仍走原始单 query 检索链路
- 二阶段重排: `BAAI/bge-reranker-v2-m3`，rerank top-10，batch_size=`64`
- BM25 bonus: title/company 文档级加分，weight=`1.0`
- 当前索引设计: 只检索 `text_chunk` / `raw_table_chunk`；rerank 输入与向量检索增强文本保持一致
- 评测规则: 严格执行“组内 OR、组间 AND”
- fact: 命中 `gold_evidence_groups` 中任意一组即可
- compare: 必须同时命中 `gold_evidence_groups1` 和 `gold_evidence_groups2`
- summary: 必须同时命中 `gold_evidence_groups1`、`gold_evidence_groups2` 和 `gold_evidence_groups3`
- chunk 文本截断: 不截断
- 输出目录: `/home/txs/work/zyp/RAG/output/decompose_2`
- hybrid_weightsum 耗时: 1362.43s
- hybrid_rrf 耗时: 1340.32s
- hybrid_weightsum JSON: `eval_hybrid_weightsum__rerank_on__qd_on__20260602_152109.json`
- hybrid_rrf JSON: `eval_hybrid_rrf__rerank_on__qd_on__20260602_152109.json`
- hybrid_weightsum 阅读版: `eval_hybrid_weightsum__rerank_on__qd_on__20260602_152109__readable.md`
- hybrid_rrf 阅读版: `eval_hybrid_rrf__rerank_on__qd_on__20260602_152109__readable.md`

## Metric Notes

- Recall@3 / Recall@5 / Recall@10: 在 top-k 内是否满足该 query 类型要求的全部组件命中条件。
- MRR@10: 从前往后看，prefix 首次满足完整组件条件时的倒数排名；10 名内无法满足则记 0。
- NDCG@10: 每个 chunk 的 graded relevance 等于它命中的必需组件数，按 top-10 的理想重排做归一化。

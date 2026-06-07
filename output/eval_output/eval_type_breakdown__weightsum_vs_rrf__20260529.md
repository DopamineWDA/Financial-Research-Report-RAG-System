# Type Breakdown Analysis

## Table

| Route | Query Type | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid_weightsum | fact | 85 | 0.4941 | 0.5529 | 0.6588 | 0.4318 | 0.4859 |
| hybrid_weightsum | compare | 20 | 0.0500 | 0.0500 | 0.0500 | 0.2792 | 0.2925 |
| hybrid_weightsum | summary | 20 | 0.0000 | 0.0000 | 0.0000 | 0.1133 | 0.1646 |
| hybrid_rrf | fact | 85 | 0.4941 | 0.5412 | 0.6353 | 0.4258 | 0.4760 |
| hybrid_rrf | compare | 20 | 0.0500 | 0.0500 | 0.0500 | 0.2597 | 0.2891 |
| hybrid_rrf | summary | 20 | 0.0000 | 0.0000 | 0.0000 | 0.1083 | 0.1501 |

## Findings

- fact: `hybrid_weightsum` vs `hybrid_rrf` 的差异整体不大；在 `Recall@10` 上差异最大，优势属于 `hybrid_weightsum`，差值为 `+0.0235`。
- compare: `hybrid_weightsum` vs `hybrid_rrf` 的差异整体不大；在 `MRR@10` 上差异最大，优势属于 `hybrid_weightsum`，差值为 `+0.0194`。
- summary: `hybrid_weightsum` vs `hybrid_rrf` 的差异整体不大；在 `NDCG@10` 上差异最大，优势属于 `hybrid_weightsum`，差值为 `+0.0145`。

## Metric Notes

- Recall@3/5/10: strict recall，要求前 k 条结果合起来覆盖该 query 的全部 gold evidence。
- MRR@10: 只看前 10 条里第一条命中 gold evidence 的排名，越靠前越高。
- NDCG@10: 同时看命中数量和排序位置，越多且越靠前越高。

# 三路召回实现说明

当前项目已补齐 3 路召回通道：

- 路线 A：纯向量召回（FAISS）
- 路线 B：纯 BM25 召回（`rank_bm25`）
- 路线 C：混合召回（向量 + BM25 融合，支持 `weighted_sum` 和 `RRF` 两种模式）

## 新增文件

- `RAG/retrieval/three_way_retriever.py`
- `RAG/retrieval/__init__.py`
- `RAG/cli/search_retrieval.py`

## 设计说明

### 1. 向量召回

- 直接复用现有 `indexes/faiss_*` 目录中的 `index.faiss`、`docstore.jsonl`、`id_map.json`、`build_meta.json`
- 查询时按 `build_meta.json` 中的 `model_path` 和 `normalize_embeddings` 设置，对 query 做同样的 embedding 和归一化
- 返回 FAISS 相似度分数

### 2. BM25 召回

- 语料直接来自 `docstore.jsonl` 中 `indexed=true` 的 chunk，保证和向量召回语料一致
- BM25 库使用 `rank_bm25.BM25Okapi`
- 中文分词器使用 `jieba`
- 英文、数字、带小数点/百分号/连接符的字符串仍按正则整体保留
- 例如 `CPU`、`GPU`、`ROE`、`2026Q1`、`17.5%` 不会被中文分词器拆碎

### 3. 混合召回

- 先分别取向量候选集和 BM25 候选集
- 支持两种融合算法，可手动切换

#### 模式 A：`weighted_sum`

- 对两路分数分别做 Min-Max 归一化
- 按如下公式融合：

```text
fused_score = vector_weight * vector_norm + bm25_weight * bm25_norm
```

- 默认权重：

```text
vector_weight = 0.6
bm25_weight = 0.4
```

你可以按任务类型调整：

- 语义问法更强时，适当提高 `vector_weight`
- 关键词、股票名、指标名、日期更敏感时，适当提高 `bm25_weight`

#### 模式 B：`RRF`

- 不直接使用原始分数，而是融合两路召回中的排名
- 计算公式如下：

```text
RRF(d) = 1 / (rrf_k + rank_vector(d)) + 1 / (rrf_k + rank_bm25(d))
```

- 默认平滑常数：

```text
rrf_k = 60
```

- 更适合分数尺度差异较大、希望融合更稳健的场景

### 4. 每一路到底召回多少个 chunk

这一点分成“单路召回”和“混合召回”两种情况来看。

#### 单路召回

当你直接使用 `vector` 或 `bm25` 路线时，返回 chunk 数量由 `top_k` 决定：

- `vector`：直接返回前 `top_k` 个向量召回结果
- `bm25`：直接返回前 `top_k` 个 BM25 召回结果

例如：

```text
top_k = 5
```

则：

- 单路 Dense 召回返回 5 个 chunk
- 单路 BM25 召回返回 5 个 chunk

#### 混合召回

当你使用 `hybrid` 路线时，两路不会一上来就只取 `top_k` 个，而是会先各自取一批候选，再做融合排序。

具体逻辑是：

```text
vector 先召回 max(top_k, vector_candidate_k) 个
bm25 先召回 max(top_k, bm25_candidate_k) 个
```

然后分别截断为：

```text
vector 保留前 vector_candidate_k 个
bm25 保留前 bm25_candidate_k 个
```

之后：

- 两路候选按 `docstore_id` 去重合并
- 再使用 `weighted_sum` 或 `rrf` 计算融合分数
- 最终只返回融合后的前 `top_k` 个 chunk

默认 CLI 参数下：

```text
top_k = 5
vector_candidate_k = 20
bm25_candidate_k = 20
```

所以默认混合召回时，实际含义是：

- Dense 先取 20 个 chunk
- BM25 先取 20 个 chunk
- 合并后最多 40 个候选，最少 20 个候选
- 最终融合排序后只返回前 5 个 chunk

一句话理解：

- `top_k` 控制“最后展示/返回多少个结果”
- `vector_candidate_k` 和 `bm25_candidate_k` 控制“融合前每一路先准备多少候选”

### 5. 分数是怎么计算的

#### Dense / Vector 单路分数

- query 会先被编码成向量
- 再进入 FAISS 检索
- 返回的 FAISS 相似度/距离分数会写入 `vector_score`

#### BM25 单路分数

BM25 并不只算 chunk 正文，还会额外考虑文档标题。

计算逻辑是：

```text
bm25_score = 正文BM25分数 + title_weight * 标题BM25分数
```

默认：

```text
title_weight = 1.0
```

也就是说，标题命中强的文档，会对它下面的 chunk 带来额外加分。

#### `weighted_sum` 融合分数

在 `weighted_sum` 模式下：

- 先对 Dense 候选分数做一次 Min-Max 归一化，得到 `vector_norm`
- 再对 BM25 候选分数做一次 Min-Max 归一化，得到 `bm25_norm`
- 然后按下面的公式融合：

```text
fused_score = vector_weight * vector_norm + bm25_weight * bm25_norm
```

注意：

- 这里融合的是“候选集内部归一化后的分数”
- 不是原始 FAISS 分数和原始 BM25 分数直接相加

#### `RRF` 融合分数

在 `RRF` 模式下：

- 不看原始分数大小
- 只看某个 chunk 在 Dense 和 BM25 各自结果里的排名

公式如下：

```text
RRF(d) = 1 / (rrf_k + rank_vector(d)) + 1 / (rrf_k + rank_bm25(d))
```

其中：

- 如果某个 chunk 只在 Dense 结果中出现，没有出现在 BM25 中，那么它只会拿到 Dense 这一项的分数
- 反之亦然
- 如果它同时在两路里都排得靠前，RRF 分数就会更高

这也是为什么 RRF 很适合“Dense 强语义、BM25 强关键词”的混合场景。

## 使用方式

### 安装 BM25 依赖

```bash
pip install rank_bm25 jieba
```

如果你还没装向量检索依赖，也需要准备：

```bash
pip install faiss-cpu sentence-transformers
```

如果环境里已经有 `FlagEmbedding`，查询时会优先使用它；否则自动回退到 `sentence-transformers`。

### 二阶段 reranker 依赖

```bash
pip install FlagEmbedding
```

二阶段默认使用本地缓存中的 `BAAI/bge-reranker-v2-m3`。

如果你已经把模型下到了 `~/.cache/huggingface/hub`，可以直接传：

```bash
--rerank-model-path BAAI/bge-reranker-v2-m3
```

### 同时查看 3 路召回

```bash
cd RAG
python cli/search_retrieval.py \
  --index-dir indexes/faiss_ivf_chunked_512_100_bge-large-zh-v1.5 \
  --query "哪些半导体公司受益于AI算力增长" \
  --route all \
  --top-k 5
```

### 只看 BM25

```bash
cd RAG
python cli/search_retrieval.py \
  --query "寒武纪 服务器 CPU GPU 业绩" \
  --route bm25
```

### 查看混合召回并调权重

```bash
cd RAG
python cli/search_retrieval.py \
  --index-dir indexes/faiss_flat_chunked_512_50_bge-large-zh-v1.5 \
  --query "核电 电价 机制电价 利润" \
  --route hybrid_weightsum \
  --first-stage-top-k 50 \
  --vector-candidate-k 50 \
  --bm25-candidate-k 50 \
  --vector-weight 0.3 \
  --bm25-weight 0.7
```

### 查看混合召回并切换到 RRF

```bash
cd RAG
python cli/search_retrieval.py \
  --index-dir indexes/faiss_flat_chunked_512_50_bge-large-zh-v1.5 \
  --query "国产算力 资本开支 业绩" \
  --route hybrid_rrf \
  --first-stage-top-k 50 \
  --rrf-top-k 50 \
  --rrf-k 60 \
  --vector-candidate-k 50 \
  --bm25-candidate-k 50
```

### 两阶段 weightsum + reranker

```bash
cd RAG
python cli/search_retrieval.py \
  --index-dir indexes/faiss_flat_chunked_512_50_bge-large-zh-v1.5 \
  --query "哪些半导体设备公司受益于先进封装扩产" \
  --route hybrid_weightsum \
  --first-stage-top-k 50 \
  --vector-candidate-k 50 \
  --bm25-candidate-k 50 \
  --vector-weight 0.3 \
  --bm25-weight 0.7 \
  --use-reranker \
  --rerank-model-path BAAI/bge-reranker-v2-m3 \
  --rerank-top-k 10
```

### 两阶段 RRF + reranker

```bash
cd RAG
python cli/search_retrieval.py \
  --index-dir indexes/faiss_flat_chunked_512_50_bge-large-zh-v1.5 \
  --query "国产算力 资本开支 业绩" \
  --route hybrid_rrf \
  --first-stage-top-k 50 \
  --rrf-top-k 50 \
  --vector-candidate-k 50 \
  --bm25-candidate-k 50 \
  --rrf-k 60 \
  --use-reranker \
  --rerank-model-path BAAI/bge-reranker-v2-m3 \
  --rerank-top-k 10
```

### 输出 JSON 结果

```bash
cd RAG
python cli/search_retrieval.py \
  --query "国产算力 资本开支" \
  --route all \
  --json
```

## 返回结果说明

每个召回结果都会带上：

- `chunk_id`
- `doc_id`
- `chunk_type`
- `source_pdf`
- `vector_score`
- `bm25_score`
- `fused_score`
- `reranker_score`
- `text_preview`

当 `fusion_mode=rrf` 时：

- `fused_score` 表示 RRF 分数
- `vector_score` 和 `bm25_score` 仍会保留，方便做对比分析

如果命中的是 `table_summary_chunk`，实现还会自动尝试挂回对应 `raw_table_chunk`，便于后续做表格原文展示。

## 后续建议

下一步如果你继续做生成链路，建议直接把 `ThreeWayRetriever` 接到统一的 `retrieve(query, route=...)` 入口上，然后再加：

- 多路召回 ablation 对比
- recall@k / mrr / ndcg 评估脚本
- 最终 answer generation

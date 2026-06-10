# 金融研报 RAG 命令总览

这份文档按实际使用顺序整理本项目常用命令。所有示例默认从项目根目录执行：

```bash
cd RAG
```

当前推荐使用的新入口是：

- `app/streamlit_demo.py`
- `cli/answer_query.py`
- `cli/search_retrieval.py`
- `cli/build_faiss_index.py`

兼容旧入口仍然保留，但文档默认不再推荐：

- `streamlit_app.py`
- `scripts/answer_query.py`
- `scripts/search_retrieval.py`
- `scripts/build_faiss_index.py`

## 1. PDF 解析

### 1.1 DeepDoc 解析

适合需要更强版面分析、表格结构识别、图表处理的场景。

```bash
python preprocess/deepdoc_parser.py \
  --input data/it_service_pdfs \
  --output-dir output/deepdoc_parse \
  --preserve-root data \
  --max-files 3
```

如果只想产出 `*.parsed.json`，跳过 review HTML/Markdown：

```bash
python preprocess/deepdoc_parser.py \
  --input data/it_service_pdfs \
  --output-dir data_parsed \
  --preserve-root data \
  --json-only
```

### 1.2 Legacy 解析

适合本地快速验证、依赖更轻的场景。

```bash
python preprocess/legacy_parser.py \
  --input data/it_service_pdfs \
  --output-dir output/legacy_parse \
  --max-files 3
```

## 2. 结构化切块

对 `*.parsed.json` 做 section-aware rechunk，生成后续检索使用的 chunk JSON。

```bash
python preprocess/rechunk_parsed_pdf.py \
  data_parsed/stock_power_pdfs/your_file.parsed.json \
  --output-dir data_chunked/chunked_512_50 \
  --chunk-size 512 \
  --overlap 50
```

如果要批量处理整个目录：

```bash
python preprocess/rechunk_parsed_pdf.py \
  data_parsed/stock_power_pdfs \
  --output-dir data_chunked/chunked_512_50 \
  --chunk-size 512 \
  --overlap 50
```

## 3. 构建索引

### 3.1 构建 HNSW 索引

```bash
python cli/build_faiss_index.py \
  --input-dir data_chunked/chunked_512_50 \
  --index-type hnsw
```

### 3.2 构建 Flat 索引

```bash
python cli/build_faiss_index.py \
  --input-dir data_chunked/chunked_512_50 \
  --index-type flat
```

如果你需要显式指定输出目录：

```bash
python cli/build_faiss_index.py \
  --input-dir data_chunked/chunked_512_50 \
  --index-type flat \
  --output-dir indexes/faiss_flat_chunked_512_50_bge-large-zh-v1.5
```

## 4. 命令行检索

### 4.1 Hybrid + Rerank

```bash
python cli/search_retrieval.py \
  --index-dir indexes/faiss_hnsw_chunked_512_50_bge-large-zh-v1.5 \
  --query "紫光股份和神州数码2026年一季度归母净利润增速谁更高？" \
  --route hybrid_weightsum \
  --use-reranker \
  --query-decompose
```

### 4.2 输出 JSON

```bash
python cli/search_retrieval.py \
  --index-dir indexes/faiss_hnsw_chunked_512_50_bge-large-zh-v1.5 \
  --query "寒武纪2026年一季度收入和归母净利润是多少？" \
  --route hybrid_rrf \
  --use-reranker \
  --json
```

## 5. 命令行问答

### 5.1 单轮问答

先确保本地 Xinference 模型服务已经启动。

```bash
python cli/answer_query.py \
  --index-dir indexes/faiss_hnsw_chunked_512_50_bge-large-zh-v1.5 \
  --endpoint http://127.0.0.1:9997 \
  --model qwen3-8b \
  --query "神州数码提示了哪些主要风险？"
```

### 5.2 多轮交互

```bash
python cli/answer_query.py --interactive
```

## 6. Streamlit Demo

```bash
streamlit run app/streamlit_demo.py
```

兼容旧入口：

```bash
streamlit run streamlit_app.py
```

## 7. 检索评测

### 7.1 两阶段检索 + Rerank 评测

```bash
python eval/scripts/eval_two_stage_rerank.py \
  --index-dir indexes/faiss_hnsw_chunked_512_50_bge-large-zh-v1.5 \
  --eval-path eval/recall_eval_m.json \
  --output-dir output/decompose_2 \
  --query-decompose
```

### 7.2 一阶段召回对比

```bash
python eval/scripts/eval_retrieval_methods.py
```

### 7.3 Chunk 策略评测

```bash
python eval/scripts/eval_chunk_recall.py
```

### 7.4 混合检索参数扫描

```bash
python eval/scripts/eval_hybrid_param_sweep.py
```

### 7.5 FAISS 索引对比

```bash
python eval/scripts/eval_faiss_index_comparison.py
```

## 8. RAGAS 相关命令

### 8.1 基于已有检索结果生成答案

```bash
python eval/scripts/generate_ragas_answers.py \
  --input-json output/decompose_1/eval_hybrid_weightsum__rerank_on__qd_on__20260601_233756.json
```

### 8.2 运行 RAGAS 评测

```bash
python eval/scripts/run_ragas_eval.py \
  --input-jsonl output/ragas_answers/eval_hybrid_weightsum__rerank_on__qd_on__20260601_233756__qwen3-8b__20260603_211622/answers.jsonl
```

## 9. 常见目录约定

- `data/`：原始 PDF
- `data_parsed/`：解析得到的 `*.parsed.json`
- `data_chunked/`：rechunk 后的 chunk JSON
- `indexes/`：FAISS 索引目录
- `output/qa/`：问答输出
- `output/eval_output/`：各类评测 Markdown 输出
- `output/ragas_eval/`：RAGAS 评测结果

## 10. 兼容说明

当前项目已经完成一轮结构整理：

- 新入口放在 `app/` 与 `cli/`
- 旧入口 `streamlit_app.py` 与 `scripts/*.py` 仍可用
- 文档与示例命令默认全部以新入口为准

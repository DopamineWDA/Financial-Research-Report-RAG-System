# 金融研报 RAG 评测集构建与严格召回评测规则

## 1. 评测集目标

本评测集用于评估金融研报 RAG 系统在不同分块策略、召回方法和 FAISS 索引结构下的证据召回能力。

当前版本评测集共 **125 条人工标注样本**，覆盖 IT 服务、电力、半导体共三个金融研报领域，包含行业研报与个股研报。样本类型包括：

| 类型 | 数量 | gold evidence 数量 |
|---|---:|---:|
| fact | 85 | 每条 1 个 |
| compare | 20 | 每条 2 个 |
| summary | 20 | 每条 3 个 |
| 总计 | 125 | - |

本评测集重点评估三类问题的检索能力：

1. 单事实精确召回能力；
2. 跨证据对比召回能力；
3. 多证据综合召回能力。

---

## 2. 核心设计原则

评测集基于 **parsed JSON 中的原始 block** 构建，分块后的 chunk 只是被测检索对象。

因此，评测集不绑定某一种 chunk 策略下的 `chunk_id`，而是绑定稳定的证据单元：

```text
(doc_id, block_id)
```

其中：

- `doc_id` 表示证据所属研报；
- `block_id` 表示 parsed JSON 中的原始 block；
- `gold_evidence` 用于保存标准证据单元；
- `gold_block_ids` 仅用于展示或兼容旧逻辑，不应单独作为最终判定依据。

采用 `(doc_id, block_id)` 而不是单独 `block_id` 的原因是：不同研报中可能都存在 `b0008`、`b0010`、`b0017` 等重复 block_id。如果只用 block_id 判断命中，会产生误命中或误失败。

---

## 3. 样本字段格式

每条样本建议包含如下字段：

```json
{
  "qid": "fact_0001",
  "query_type": "fact",
  "query": "阿里巴巴2026财年全年营收和季度自由现金流分别是多少？",
  "answer": "阿里巴巴2026财年全年营收为10236.7亿元，季度自由现金流为净流出173亿元。",
  "ground_truth_doc_id": "001_2026-05-22_金元证券_计算机行业周评：阿里大幅上调AI资本开支",
  "gold_block_ids": ["b0029"],
  "gold_evidence": [
    {
      "doc_id": "001_2026-05-22_金元证券_计算机行业周评：阿里大幅上调AI资本开支",
      "block_id": "b0029"
    }
  ],
  "evidence_pages": [5],
  "evidence_text": "2026年5月13日，阿里巴巴集团发布了2026财年第四季度及全财年业绩...",
  "difficulty": "easy"
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `qid` | 样本 ID，例如 `fact_0001` |
| `query_type` | 问题类型：`fact`、`compare`、`summary` |
| `query` | 用户问题 |
| `answer` | 标准答案 |
| `ground_truth_doc_id` | 主证据文档 ID；跨文档问题可作为主文档或兼容字段 |
| `gold_block_ids` | 标准证据 block_id 列表，保留用于展示 |
| `gold_evidence` | 最终评测使用的标准证据单元，格式为 `(doc_id, block_id)` |
| `evidence_pages` | 证据页码 |
| `evidence_text` | 支撑答案的证据文本 |
| `difficulty` | 难度标签，可为 `easy`、`medium`、`hard` |

---

## 4. 三类 query 定义

### 4.1 fact：事实型问题

fact 用于考察单个明确事实的精确召回能力。

这类问题通常只需要召回 **1 个 gold evidence** 即可回答。答案应明确、唯一、可验证，并且可以由单个 block 直接支撑。

适合构建 fact 的内容包括：

1. 营业收入、归母净利润、毛利率、同比增速等财务指标；
2. 评级、目标价、分红、装机容量、发电量等明确数值；
3. 公司事件、政策时间、项目规模、负责人等确定性信息；
4. 风险提示、推荐标的、投资建议中的明确条目；
5. 同一个 block 内两个数值的简单比较。

注意：如果问题问法中包含“哪个更高”“分别是多少”，但所有比较信息都位于同一个 block 内，仍可归为 fact，因为它主要考察的是单 block 精确召回，而不是跨证据召回。

示例：

```text
寒武纪2026年一季度收入和归母净利润是多少？
```

---

### 4.2 compare：对比型问题

compare 用于考察两个对象、两个指标或两个时间点之间的跨证据对比召回能力。

本评测集中的 compare 样本统一绑定 **2 个 gold evidence**。这两个证据可以来自：

1. 同一研报的两个不同 block；
2. 不同研报的两个 block；
3. 两家公司、两个行业、两个指标或两个时间点的对应证据。

compare 的答案应包含两个对象各自的关键信息，并给出明确比较结论。

适合构建 compare 的问题包括：

1. 两家公司营收、利润、毛利率、增速对比；
2. 两个行业或板块表现对比；
3. 两个时间点的数据变化对比；
4. 两个政策、事件或投资逻辑的差异对比。

示例：

```text
紫光股份和神州数码2026年一季度归母净利润增速谁更高？
```

该类问题必须召回两个对象对应的证据块，才能认为检索成功。

---

### 4.3 summary：汇总型问题

summary 用于考察多证据综合召回能力。

本评测集中的 summary 样本统一绑定 **3 个 gold evidence**。这些证据通常共同支撑一个归纳性答案。

适合构建 summary 的问题包括：

1. 行业趋势总结；
2. 投资逻辑归纳；
3. 公司增长驱动因素总结；
4. 多项风险因素汇总；
5. 政策影响、产业链机会、商业化进展等多角度分析。

summary 不应是单个数值、单个负责人、单句风险提示等事实型问题。答案应是简洁归纳后的标准答案，而不是简单复制多个段落。

示例：

```text
报告认为国产算力产业链景气度上行的主要原因有哪些？
```

该类问题必须召回 3 个标准证据块，才能认为检索成功。

---

## 5. 样本构建规则

### 5.1 fact 构建规则

1. 每条 fact 样本绑定 1 个 `gold_evidence`；
2. 问题必须具体，不能过泛；
3. 答案必须明确、唯一、可验证；
4. `evidence_text` 必须直接支撑答案；
5. query 不能直接泄漏答案；
6. 可以包含同一 block 内的简单比较，但不能依赖多个 block。

---

### 5.2 compare 构建规则

1. 每条 compare 样本绑定 2 个 `gold_evidence`；
2. 两个 evidence 分别支撑两个被比较对象、指标或时间点；
3. 答案必须同时给出双方信息和比较结论；
4. 允许跨文档比较，此时必须依赖 `gold_evidence` 中的 `doc_id + block_id` 判断命中；
5. 如果两个比较对象完全位于同一个 block 内，原则上应归为 fact，而不是 compare。

---

### 5.3 summary 构建规则

1. 每条 summary 样本绑定 3 个 `gold_evidence`；
2. 三个 evidence 应共同支撑答案的主要论点；
3. 答案应是归纳总结，不宜过长；
4. 不要求标注所有相关证据，但必须覆盖回答该问题所需的核心证据；
5. 不应选择目录页、免责声明、OCR 错误严重内容作为主要证据。

---

## 6. 样本筛选标准

保留样本需要满足：

1. query 表达自然，符合真实用户提问习惯；
2. query 指向明确，答案不应存在多个合理版本；
3. answer 可以完全由 `gold_evidence` 对应证据支撑；
4. evidence_text 与 answer 不存在数字、公司、时间或指标错配；
5. 不依赖外部网页、常识或模型自由推断；
6. 不使用 OCR 明显错误的内容作为标准答案；
7. 不让 query 直接泄漏答案；
8. 不选择目录页、免责声明等低价值内容作为核心证据。

应剔除或修改的样本包括：

1. 问法模糊、答案不唯一；
2. evidence_text 无法支撑 answer；
3. 数字单位不一致或四舍五入不合理；
4. gold evidence 数量不符合类型要求；
5. compare 实际只需一个 block 回答；
6. summary 实际只是单事实问题；
7. OCR 表格错位严重，无法确认标准答案。

---

## 7. 不同分块策略下的评测方式

不同 chunk size、overlap 或分块方法会产生不同的 `chunk_id`，因此不能把 `chunk_id` 作为 ground truth。

正确流程是：

1. 评测集保存标准证据单元 `gold_evidence = [(doc_id, block_id), ...]`；
2. 每种分块策略的 docstore 中，每个 chunk 保存其来源信息，例如：

```json
{
  "metadata": {
    "doc_id": "xxx",
    "block_ids": ["b0010", "b0011"]
  }
}
```

3. 检索得到 Top-N chunks；
4. 将每个 chunk 映射回若干 `(doc_id, block_id)`；
5. 判断 Top-N 是否覆盖该 query 的全部 `gold_evidence`。

---

## 8. 最终评测指标：Strict Evidence Recall@N

本项目最终采用 **Strict Evidence Recall@N** 作为主指标。

定义：

```text
对于每条 query，若 Top-N 检索结果覆盖该 query 的全部 gold evidence units，
则该 query 的 Strict Evidence Recall@N = 1；否则为 0。
```

其中，gold evidence unit 定义为：

```text
(doc_id, block_id)
```

不同题型的命中条件为：

| query_type | gold evidence 数量 | 命中条件 |
|---|---:|---|
| fact | 1 | 召回 1/1 |
| compare | 2 | 召回 2/2 |
| summary | 3 | 召回 3/3 |

也就是说，本项目不再使用“summary 召回 50% 即成功”的宽松规则，而是统一采用 **全部 gold evidence 召回成功** 的严格规则。

---

## 9. Recall@N 计算方式

对每条 query 先计算一个 0/1 命中值，再对所有样本取平均：

```text
Strict Evidence Recall@N = 命中样本数 / 总样本数
```

例如，共 125 条样本，其中 80 条在 Top-5 中召回全部 gold evidence，则：

```text
Strict Evidence Recall@5 = 80 / 125 = 0.64
```

可以同时统计总体指标和分类型指标：

```text
Overall Strict Recall@N
Fact Strict Recall@N
Compare Strict Recall@N
Summary Strict Recall@N
```

建议重点关注：

1. fact：Recall@3、Recall@5；
2. compare：Recall@5、Recall@10；
3. summary：Recall@10。

原因是 summary 需要一次性召回 3 个证据块，Top-3 没有冗余空间，难度明显高于 fact。

---

## 10. 代码实现逻辑

推荐使用 `gold_evidence` 判断，不要只使用 `gold_block_ids`。

```python
def strict_evidence_hit(sample, retrieved_chunks, top_n):
    gold = {
        (e["doc_id"], e["block_id"])
        for e in sample["gold_evidence"]
    }

    retrieved = set()
    for chunk in retrieved_chunks[:top_n]:
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("doc_id")
        block_ids = metadata.get("block_ids", [])
        for block_id in block_ids:
            retrieved.add((doc_id, block_id))

    return int(gold.issubset(retrieved))
```

如果需要统计命中数量和命中比例，可扩展为：

```python
def strict_evidence_stats(sample, retrieved_chunks, top_n):
    gold = {
        (e["doc_id"], e["block_id"])
        for e in sample["gold_evidence"]
    }

    retrieved = set()
    for chunk in retrieved_chunks[:top_n]:
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("doc_id")
        for block_id in metadata.get("block_ids", []):
            retrieved.add((doc_id, block_id))

    hit = gold & retrieved
    return {
        "hit": int(gold.issubset(retrieved)),
        "hit_count": len(hit),
        "gold_count": len(gold),
        "hit_ratio": len(hit) / len(gold) if gold else 0.0,
        "missing_evidence": sorted(gold - retrieved)
    }
```

---

## 11. 实验表指标定义

召回实验中的 Recall@3、Recall@5、Recall@10 均表示：

```text
Strict Evidence Recall@N
```

即 Top-N chunks 是否覆盖该问题的全部标准证据单元。

示例表：

| 召回方案 | Strict Recall@3 | Strict Recall@5 | Strict Recall@10 | 备注 |
|---|---:|---:|---:|---|
| 纯向量 | ? | ? | ? | 语义召回 baseline |
| 纯 BM25 | ? | ? | ? | 数字、专有名词、关键词较强 |
| 混合召回 | ? | ? | ? | 通常兼顾语义和关键词 |

分类型结果建议单独统计：

| 召回方案 | Fact R@10 | Compare R@10 | Summary R@10 | Overall R@10 |
|---|---:|---:|---:|---:|
| 纯向量 | ? | ? | ? | ? |
| 纯 BM25 | ? | ? | ? | ? |
| 混合召回 | ? | ? | ? | ? |

---

## 12. FAISS 索引对比规则

FAISS 索引对比建议统计：

1. Strict Evidence Recall@10；
2. 查询延迟；
3. 构建时间；
4. 内存占用；
5. ANN Recall@10 vs Flat。

示例表：

| 索引类型 | Strict Evidence Recall@10 | 查询延迟(ms) | 构建时间(s) | 内存占用(MB) |
|---|---:|---:|---:|---:|
| Flat | ? | ? | ? | ? |
| IVF | ? | ? | ? | ? |
| HNSW | ? | ? | ? | ? |

其中，`ANN Recall@10 vs Flat` 用于衡量 IVF/HNSW 与 Flat 精确检索结果的重合程度：

```text
ANN Recall@10 vs Flat = ANN Top-10 与 Flat Top-10 的交集数量 / 10
```

注意：`ANN Recall@10 vs Flat` 评估的是近似索引相对 Flat 的检索一致性，不等同于 RAG 证据召回效果；最终业务指标仍以 `Strict Evidence Recall@N` 为主。

---

## 13. 当前评测集简要说明

当前评测集具有以下特点：

1. 共 125 条人工标注样本；
2. 覆盖 IT 服务、电力、半导体三个金融研报领域；
3. 包含行业研报与个股研报；
4. fact、compare、summary 三类问题分别考察单事实、双证据对比、多证据综合召回；
5. 每条样本均使用 `gold_evidence` 标注 `(doc_id, block_id)`；
6. fact 绑定 1 个 evidence，compare 绑定 2 个 evidence，summary 绑定 3 个 evidence；
7. 采用严格全证据召回标准，Top-N 必须覆盖全部 gold evidence 才算命中。

---

## 14. 一句话版本

本项目采用 **Strict Evidence Recall@N** 评估金融研报 RAG 检索效果：评测集不绑定 chunk_id，而是以 `(doc_id, block_id)` 标注标准证据单元；检索得到 Top-N chunks 后映射回原始 evidence blocks，只有当 fact 召回 1/1、compare 召回 2/2、summary 召回 3/3 全部 gold evidence 时，该 query 才记为命中，最终 Recall@N 为所有样本 0/1 命中的平均值。

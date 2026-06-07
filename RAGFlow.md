# RAGFlow

## DeepDoc

PDF Parser 主要流程：

PDF 页面
├─ 渲染成图片
 │    ├─ layout 检测：正文/标题/表格/图片
 │    ├─ OCR detect：找文字框
 │    ├─ OCR recognize：必要时识别文字
 │    └─ TSR：表格结构识别
 │ 
└─ pdfplumber 读取文本层
      ├─ 拿字符内容
      ├─ 拿字符坐标
      └─ 判断是否乱码/CID/可用

然后合并两边结果：

用图片模型确定区域和框
用 PDF 文本层优先填文字
文本层不可靠时用 OCR 文字

版面分析：给文本分配栏编号（当前页）

*PDF 常见多栏排版，尤其是论文/研报。若不识别列，简单按 y 排序会把左栏和右栏混在一起，因此这里用每页 box 的 x0 做 KMeans 聚类，自动估计列数，然后把每个 box 标上 col_id。*

单栏页面的 x0 大概集中在一个地方：

`x0: 80, 82, 79, 85, 81`

双栏页面的 x0 会集中在两个地方：

`左栏 x0: 80, 82, 79, 85
右栏 x0: 350, 352, 348, 355`

三栏页面可能集中在三个地方：

`80, 260, 440`

这就是聚类问题。

KMeans 做的事就是：

> *把这些 x0 按距离分成几堆，每一堆就是一栏。*
> 

DeepDoc 核心PDF解析思路：

PDF
-> 每页渲染成图片
-> 同时读取 PDF 原生文本层 chars
-> OCR 检测文本框
-> 如果 PDF 文本层可靠，用文本层填字
-> 如果乱码/CID/扫描件，用 OCR 兜底
-> layout 识别 title/text/table/figure/header/footer
-> 表格区域做 TSR，识别 row/column/header/spanning
-> 横向/纵向合并文本框
-> 抽取 table/figure
-> 给 box 加 position_tag / image / positions

### 检索召回链路：

用户 query
|
v
[粗召回阶段]
|
|-- MatchTextExpr
|     稀疏召回：关键词 / 全文 / ES 下通常 BM25
|
|-- MatchDenseExpr
|     稠密召回：query embedding -> vector index topK
|
|-- FusionExpr
引擎侧融合：先拿候选池
|
v
候选 chunks: sres
|
v
[精排阶段]
|
|-- 如果启用 rerank_mdl:
|      reranker(query, chunk) -> rerank_score
|
|-- 否则如果 Infinity:
|      直接用引擎 _score
|
|-- 否则如果 ES:
|      二次 KNN score + 本地 term_similarity
|
|-- 否则如果 OceanBase:
|      拉回向量，本地 cosine + term_similarity
|
v
similarity =
term_weight * term_similarity+vector_weight * vector_similarity_or_rerank_score+rank_feature
|
v
[后处理]
排序
similarity_threshold 过滤
page/page_size 分页
TOC/父子 chunk 后处理
|
v
返回 chunks

- `MatchTextExpr / MatchDenseExpr / FusionExpr 解决“先召回哪些候选”；`
- `rerank_with_knn / rerank / rerank_by_model 解决“这些候选最终怎么排序”。`

### 第一步：PDF解析

使用deepdoc方法进行OCR+Layout Rec+TSR进行解析，得到四种layout_type：figure、table、text、title。

```json
 {
      "id": "b0004",
      "type": "title",
      "page": 1,
      "bbox": [
        377.3,
        146.3,
        575.0,
        164.0
      ],
      "text": "—计算机行业点评报告",
      "meta": {
        "layout_type": "title",
        "layoutno": "title-3",
        "source_block_ids": [
          "b0005"
        ],
        "positions": [
          [
            1,
            377.3,
            575.0,
            146.3,
            164.0
          ]
        ],
        "position_tag": "@@1\t377.3\t575.0\t146.3\t164.0##",
        "col_ids": [
          3
        ]
      }
    },
```

### 第二步：分Chunk

1. 先做 block 级清洗
删除页眉页脚、分析师信息、相关研究报告、免责声明、评级说明、纯资料来源、低质量 figure OCR。
保留正文中的风险提示/风险因素分析。
2. 首页封面单独处理
不与正文合并。
只生成一条 cover_summary_chunk，包含标题、机构、日期、评级、核心观点一句话。
3. 正文以章节块为主切分单位
优先按 title block、编号标题、明显小标题切分。
title 不单独入库，而是作为 section / subsection metadata。
章节过长时，在章节内部按段落滑窗切分。
    
    title仍然会在正文中防止识别错误误删内容。
    
4. text chunk 参数可配置
支持 256 / 512 / 1024 tokens。
overlap 支持 50 / 100 / 200 tokens。
overlap 只允许发生在同一章节内部。
跨章节、跨表格、跨封面不 overlap。
5. table 原子保护
table 不进入普通 text buffer。
默认一张表一个 raw_table_chunk。
如果表格超过 chunk_size，则按行组切分，不按字符切。
每个 table 子 chunk 必须重复 caption、表头、section、页码、row_range。
6. table 双通道输出
raw_table_chunk：保留 HTML / markdown / 结构化内容，用于精确问答。
table_summary_chunk：表名 + 所属章节 + 前后文 2-3 句 + 关键指标摘要，用于语义召回。
两者通过 table_id 关联。
7. figure 默认跳过
只有当 figure 有明确标题、坐标含义、趋势描述或图注时，才生成 figure_summary_chunk。
单纯 OCR 数字、资料来源、图例文本不入库。
8. 所有 chunk 必须带 metadata
包括 doc_id、source、page_start、page_end、section、subsection、block_ids、chunk_type、table_id/figure_id。
9.  overlap 会把 chunk 开头切成半句话。尽可能取最近一句分割。 chunk_id 批量处理时会冲突，需要加 doc 前缀。

### 第三步：嵌入并构建索引

- 用 FlagEmbedding 的 bge-large-zh-v1.5 对chunked_512_100做 embedding
- 用 FAISS 构建索引（先用 Flat，后面可以对比对比 IVF/HNSW）
- 输出：索引构建脚本 + 向量维度和文档数记录

表格特殊处理：

- RAGflow方法：
    
    因为有用户明确提到：RAGFlow chunking documents 时默认会把 table 隔离成 separate chunks。如果表格主要是数字或代码，缺少语义词，检索时就容易只召回描述性文本，而召不回真正包含数据的表格。
    
    因此： RAGflow将表格仍作为 chunk 进入知识库；为了让表格更容易被召回，RAGFlow 给 image/table chunk 增加上下文窗口文本；检索时召回的是带上下文的 table chunk。**表格/图片原本是独立 chunk，但后来专门增加了 “Image & table context window”，把表格周围上下文塞进 table chunk 里，避免纯表格向量召回弱。**
    
    raw table chunk
    +
    附近上下文文本
    → 共同作为 table chunk 的 embedding 文本
    
- 我们的方法：
    
    raw_table_chunk 不直接 embedding
    table_summary_chunk / 上下文摘要参与 embedding
    raw_table 通过 table_id 回填
    
    **即：table_summary_chunk / contextual_table_chunk 用于召回，raw_table_chunk 用于精确回答**
    
    **检索流程：**
    
    用户问题
    ↓
    向量检索 text_chunk / table_summary_chunk / contextual_table_chunk
    ↓
    如果命中 table_summary_chunk
    根据 table_id 找 raw_table_chunk
    ↓
    把 text_chunk + raw_table_chunk 一起给 LLM
    
    **Faiss索引**
    
    IndexFlatL2：找欧氏距离最近的向量，距离越小越好。
    IndexFlatIP：找内积最大的向量，分数越大越好。
    **RAG 常用（Ours）：向量先 L2 归一化（L2 normalize），再用 IndexFlatIP，相当于余弦相似度。**
    
    **检索流程：**
    
    1. 对问题做 embedding
    ↓
    2. 用 index.faiss 检索最相似的向量
    ↓
    3. FAISS 返回 faiss_id，例如：
    [15, 283, 1024]
    ↓
    4. 用 id_map.json 查询：
    15 -> chunk_000015
    283 -> chunk_000283
    1024 -> chunk_001024
    ↓
    5. 用 docstore.jsonl 取出这些 chunk 的原文、PDF名、页码
    ↓
    6. 把这些原文作为上下文交给大模型
    ↓
    7. LLM 根据检索到的研报内容生成答案

**评测集：**

本评测集用于评估金融研报 RAG 的检索召回效果，共 **125 条人工标注样本**，覆盖 **IT 服务、电力、半导体共三个金融研报领域**，包含行业研报与个股研报，能够检验模型在不同板块、不同报告类型下的证据召回能力。

样本分为三类：**fact** 主要考察单个明确事实的精确召回，如营收、利润、毛利率、评级、风险提示等；**compare** 主要考察两个对象、两个指标或两个时间点之间的对比召回，通常需要召回 2 个证据块；**summary** 主要考察多证据综合召回能力，面向投资逻辑、行业趋势、影响因素、风险总结等问题，通常需要召回 3 个证据块。

每条样本使用 **doc_id + block_id** 标注标准证据单元。评测采用 **Strict Evidence Recall@N**：Top-N 检索结果必须召回该问题对应的全部 gold blocks 才算命中，即 fact 召回 1/1，compare 召回 2/2，summary 召回 3/3。该评测集可用于对比不同 **chunk size、overlap、FAISS 索引结构、BM25/向量/混合召回方案** 的检索效果。

### **第四步：评测实验：**

1. **chunk分块实验：**
    
    目的：找最优 chunk_size/overlap，哪种 chunk_size / overlap 对向量语义召回最友好？
    
    **a）固定参数：vector index = FAISS Flat，retrieval = Dense 向量召回**
    
    | **chunk_size** | **overlap** | **chunk_count** | **Strict Recall@3** | **Strict Recall@5** | **Strict Recall@10** | **Fact@10** | **Compare@10** | **Summary@10** |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 256 | 50 | 10437 | 0.2160 | 0.2720 | **0.3200** | **0.4706** | 0.0000 | 0.0000 |
    | 256 | 100 | 10714 | 0.2240 | **0.2800** | 0.3120 | 0.4588 | 0.0000 | 0.0000 |
    | 256 | 200 | 11238 | 0.1840 | 0.2560 | 0.2960 | 0.4353 | 0.0000 | 0.0000 |
    | 512 | 50 | 6701 | **0.2320** | 0.2720 | 0.3040 | 0.4471 | 0.0000 | 0.0000 |
    | 512 | 100 | 6772 | **0.2320** | 0.2640 | 0.3040 | 0.4471 | 0.0000 | 0.0000 |
    | 512 | 200 | 7055 | 0.2240 | 0.2560 | 0.3120 | 0.4588 | 0.0000 | 0.0000 |
    | 1024 | 50 | 5064 | 0.1920 | 0.2560 | 0.2880 | 0.4235 | 0.0000 | 0.0000 |
    | 1024 | 100 | 5072 | 0.1920 | 0.2560 | 0.2880 | 0.4235 | 0.0000 | 0.0000 |
    | 1024 | 200 | 5099 | 0.2000 | 0.2560 | 0.2960 | 0.4353 | 0.0000 | 0.0000 |

其中：

- `Strict Recall@3/5/10`：所有 125 条样本平均
- `Fact@10`：85 条 fact 样本平均
- `Compare@10`：20 条 compare 样本平均
- `Summary@10`：20 条 summary 样本平均
- `chunk_count`：该分块策略生成的总 chunk 数
    
    
    结论上，如果目标是“对向量语义召回最友好”，这轮 dense-only baseline 里我会优先选 chunk_size=256, overlap=50，因为它的 Strict Recall@10 和 Fact@10 都是最高。若更看重前位召回，512/50 和 512/100 的 Recall@3 最好，256/100 的 Recall@5 最好，但到 @10 还是 256/50 最强。
    
    还有一个很重要的现象：Compare@10 和 Summary@10 在 9 组配置下全是 0。这说明在你当前这套 dense-only 语义召回里，问题不只是 chunk 参数，更多是“多证据全覆盖”这件事本身还没被解决。下一步最值得做的不是继续细抠 overlap，而是直接做第二个实验：Dense vs BM25 vs Hybrid，看看多证据严格召回能不能被拉起来。
    
    **b）固定参数：vector index = FAISS Flat，retrieval = Hybrid RRF**
    
    | **chunk_size** | **overlap** | **chunk_count** | **Strict Recall@3** | **Strict Recall@5** | **Strict Recall@10** | **Fact@10** | **Compare@10** | **Summary@10** |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 256 | 50 | 10437 | 0.2640 | 0.3040 | 0.3840 | 0.5529 | 0.0500 | 0.0000 |
    | 256 | 100 | 10714 | **0.2720** | 0.2960 | 0.3920 | 0.5647 | 0.0500 | 0.0000 |
    | 256 | 200 | 11238 | 0.2400 | 0.2720 | 0.3760 | 0.5412 | 0.0500 | 0.0000 |
    | 512 | 50 | 6701 | 0.2640 | 0.3280 | **0.4000** | **0.5765** | 0.0500 | 0.0000 |
    | 512 | 100 | 6772 | 0.2640 | 0.3280 | 0.3920 | 0.5647 | 0.0500 | 0.0000 |
    | 512 | 200 | 7055 | 0.2400 | **0.3360** | **0.4000** | **0.5765** | 0.0500 | 0.0000 |
    | 1024 | 50 | 5064 | 0.2400 | 0.3280 | 0.3520 | 0.5176 | 0.0000 | 0.0000 |
    | 1024 | 100 | 5072 | 0.2320 | 0.3280 | 0.3520 | 0.5176 | 0.0000 | 0.0000 |
    | 1024 | 200 | 5099 | 0.2480 | 0.3440 | 0.3760 | 0.5529 | 0.0000 | 0.0000 |
    
    这轮 hybrid RRF 比纯 dense 明显更强。按 Strict Recall@10 看，最优是 512/50 和 512/200 并列第一，都是 0.4000；如果更看重 @5，这轮最好的是 1024/200 的 0.3440；如果更看重 @3，则是 256/100 的 0.2720。
    
    整体上，和上一轮 dense-only 相比，最值得优先保留的 chunk 策略已经从 256/50 偏向了 512 档，尤其是 512/50。不过 Summary@10 仍然是 0，说明仅靠 chunk 策略和 hybrid 融合，还不足以解决 3 证据严格覆盖。
    
    **c）总结**
    
    在 Hybrid RRF 召回下，整体 Strict Recall@10 相比 Dense-only 明显提升，最佳结果由 0.3200 提升至 0.4000，说明 BM25 与向量召回的排名融合能够有效补充金融研报中的关键词、公司名、财务指标和数字匹配能力。
    
    从分块配置看，chunk_size=512 的整体表现最稳定，其中 512/50 与 512/200 在 Strict Recall@10 上均达到 0.4000，Fact@10 均达到 0.5765。考虑到 512/50 的 chunk_count 更少、冗余更低，最终选择 chunk_size=512、overlap=50 作为后续主实验的默认分块配置。
    
    不过，Compare@10 最高仅为 0.0500，Summary@10 仍为 0，说明即使使用 Hybrid RRF，当前单轮检索仍难以满足多证据问题的严格全覆盖要求。后续需要通过 query decomposition、multi-query retrieval、rerank 或邻近块扩展等方法提升多证据召回能力。
    
1. **题型难度分析**
    
    目的：看 fact/compare/summary 难度以及最终配置表现
    
    固定：index = Flat，retrieval = 最优混合召回，chunk与overlap使用实验1最优
    
    | **retrieval** | **question_type** | **Strict Recall@3** | **Strict Recall@5** | **Strict Recall@10** | **说明** |
    | --- | --- | --- | --- | --- | --- |
    | hybrid_weightsum(0.3/0.7) | fact（85） | **0.4000** | **0.4824** | **0.6118** | 需召回 1/1 个 gold evidence |
    | hybrid_weightsum(0.3/0.7) | compare（20） | 0.0000 | 0.0000 | 0.0500 | 需召回 2/2 个 gold evidence |
    | hybrid_weightsum(0.3/0.7) | summary（20） | 0.0000 | 0.0000 | 0.0000 | 需召回 3/3 个 gold evidence |
    | hybrid_weightsum(0.3/0.7) | **All（125）** | **0.2720** | **0.3280** | **0.4240** | 全部样本平均 |
    | hybrid_rrf(k=60) | fact（85） | 0.3882 | 0.4824 | 0.5765 | 需召回 1/1 个 gold evidence |
    | hybrid_rrf(k=60) | compare（20） | 0.0000 | 0.0000 | 0.0500 | 需召回 2/2 个 gold evidence |
    | hybrid_rrf(k=60) | summary（20） | 0.0000 | 0.0000 | 0.0000 | 需召回 3/3 个 gold evidence |
    | hybrid_rrf(k=60) | **All（125）** | **0.2640** | **0.3280** | **0.4000** | 全部样本平均 |
    

1. **召回方案对比**
    
    目的：比较 Dense/BM25/Hybrid
    
    固定参数：index = Flat，chunk与overlap使用实验1最优，WeightSum：Dense 和 BM25 分数归一化后权重固定：0.3/0.7
    
    只改变召回策略：Dense / BM25 / WeightSum Hybrid / RRF Hybrid。
    
    WeightSum：对 Dense 和 BM25 分数归一化后加权(0.3/0.7最优看后面超参数实验)求和。
    
    RRF：Reciprocal Rank Fusion，只利用排序名次融合，鲁棒性更好（k=60）。
    
    | **index** | **retrieval** | **Recall@3** | **Recall@5** | **Recall@10** | 备注 |
    | --- | --- | --- | --- | --- | --- |
    | faiss_flat_chunked_512_50 | dense | 0.2320 | 0.2720 | 0.3040 | 语义召回 baseline |
    | faiss_flat_chunked_512_50 | bm25 | 0.2640 | 0.3120 | 0.4080 | 数字、公司名、指标名更强 |
    | faiss_flat_chunked_512_50 | hybrid_weightsum | **0.2720** | **0.3280** | **0.4240** | 分数归一化后加权融合 |
    | faiss_flat_chunked_512_50 | hybrid_rrf | 0.2640 | **0.3280** | 0.4000 | 基于排名融合，通常更稳 |
    | faiss_flat_chunked_256_50 | dense | 0.2160 | 0.2720 | 0.3200 | 语义召回 baseline |
    | faiss_flat_chunked_256_50 | bm25 | 0.2320 | 0.2880 | 0.3200 | 数字、公司名、指标名更强 |
    | faiss_flat_chunked_256_50 | hybrid_weightsum | 0.2480 | 0.3040 | 0.3680 | 分数归一化后加权融合 |
    | faiss_flat_chunked_256_50 | hybrid_rrf | 0.2640 | 0.3040 | 0.3840 | 基于排名融合，通常更稳 |
    |  |  |  |  |  |  |
    
    结论:
    
    512/50 是当前更优 chunk 配置。
    512/50 + hybrid_weightsum 的 Recall@10 最高，为 0.4160。
    512/50 + hybrid_rrf 的 Recall@5 最高，为 0.3280。
    BM25 单路明显强于 dense 单路。
    两个 hybrid 方案整体都优于单路方案。
    
    超参数对比：
    
    | **index** | **retrieval** | **Recall@3** | **Recall@5** | **Recall@10** |
    | --- | --- | --- | --- | --- |
    | faiss_flat_chunked_512_50 | hybrid_weightsum(0.2/0.8) | 0.2800 | 0.3040 | 0.4080 |
    | faiss_flat_chunked_512_50 | hybrid_weightsum(0.3/0.7) | 0.2720 | **0.3280** | **0.4240** |
    | faiss_flat_chunked_512_50 | hybrid_weightsum(0.4/0.6) | 0.2720 | 0.3200 | 0.4160 |
    | faiss_flat_chunked_512_50 | hybrid_weightsum(0.5/0.5) | **0.2880** | 0.3280 | 0.3680 |
    | faiss_flat_chunked_512_50 | hybrid_rrf(k=10) | 0.2640 | **0.3440** | 0.3760 |
    | faiss_flat_chunked_512_50 | hybrid_rrf(k=30) | **0.2720** | 0.3360 | 0.3920 |
    | faiss_flat_chunked_512_50 | hybrid_rrf(k=50) | 0.2640 | 0.3280 | **0.4000** |
    | faiss_flat_chunked_512_50 | hybrid_rrf(k=60) | 0.2640 | 0.3280 | **0.4000** |
    | faiss_flat_chunked_512_50 | hybrid_rrf(k=80) | 0.2640 | 0.3280 | 0.3920 |
    | faiss_flat_chunked_256_50 | hybrid_weightsum(0.2/0.8) | 0.2400 | 0.3040 | 0.3520 |
    | faiss_flat_chunked_256_50 | hybrid_weightsum(0.3/0.7) | 0.2480 | 0.3040 | 0.3680 |
    | faiss_flat_chunked_256_50 | hybrid_weightsum(0.4/0.6) | 0.2480 | 0.3120 | 0.3680 |
    | faiss_flat_chunked_256_50 | hybrid_weightsum(0.5/0.5) | 0.2640 | 0.3120 | 0.3600 |
    | faiss_flat_chunked_256_50 | hybrid_rrf(k=10) | 0.2720 | 0.3120 | 0.3760 |
    | faiss_flat_chunked_256_50 | hybrid_rrf(k=30) | 0.2640 | 0.3040 | 0.3840 |
    | faiss_flat_chunked_256_50 | hybrid_rrf(k=50) | 0.2640 | 0.3040 | 0.3840 |
    | faiss_flat_chunked_256_50 | hybrid_rrf(k=60) | 0.2640 | 0.3040 | 0.3840 |
    | faiss_flat_chunked_256_50 | hybrid_rrf(k=80) | 0.2640 | 0.3040 | 0.3840 |
    
2. **FAISS 索引对比（面试区分度很高）**

目的：FAISS 索引对比

固定参数：retrieval = dense 向量召回，chunk与overlap使用实验1最优。

| 索引类型 | Recall@10 | 查询延迟(ms) | CPU构建时间(s) | 索引文件大小(MB) |
| --- | --- | --- | --- | --- |
| flat_chunked_512_50 | 0.3040 | 0.895 | 3158.1628 | 26.176 |
| hnsw_chunked_512_50 | 0.2880 | 0.083 | 3247.9159 | 27.912 |
| ivf_chunked_512_50 | 0.2800 | 0.172 | 3146.6847 | 26.548 |

**最推荐的实验顺序：**

| **顺序** | **实验** | **固定项** | **变量** | **目的** |
| --- | --- | --- | --- | --- |
| 1 | Chunk 分块实验 | Dense + Flat + bge-large-zh-v1.5 | 9 组 chunk/overlap | 找最优分块 |
| 2 | 召回方案对比 | 最优 chunk + Flat | Dense / BM25 / Hybrid-WeightSum / Hybrid-RRF | 找最优 retriever |
| 3 | 题型难度分析 | 最优 chunk + 最优 retriever | fact / compare / summary | 看最终系统对不同题型的表现 |
| 4 | FAISS 索引对比 | 最优 chunk + Dense | Flat / IVF / HNSW | 看向量索引近似损失与效率 trade-off |

### 第五步：二阶段Reranker

- 用 FlagEmbedding 的 bge-reranker-v2-m3
- 对比：向量召回 Top20 → Rerank 取 Top5 vs 直接 Top5
- 记录 Recall 和准确率变化

详细方法：本实验采用两阶段检索框架。第一阶段对faiss-chunked-512-50索引使用 Dense 向量召回与 BM25 关键词召回分别获取 Top-50 候选块，并采用 RRF 方法进行融合排序，其中 RRF 参数 k 设置为 60。随后保留融合排序后的 Top-50 chunk 作为候选集合（或使用hybrid-weightsum（权重vector/BM25=0.3/0.7）召回50chunk），输入 BGE reranker-v2-m3 进行二阶段重排序，最终输出 Top-10 chunk 用于召回评测与后续生成。

| Method | **指标** | 值 | **关注点** | **适合场景** |
| --- | --- | --- | --- | --- |
| hybrid_weightsum | Recall@10 | **0.456** | Top 10 里有没有召回正确 chunk | 粗召回能力 |
| hybrid_weightsum | MRR@10 | **0.356** | 第一个正确 chunk 排得多靠前 | 事实型问答 |
| hybrid_weightsum | NDCG@10 | **0.404** | 多个正确 chunk 是否整体排得靠前 | 对比型、汇总型、多证据问题 |
| hybrid_rrf | Recall@10 | 0.44 | Top 10 里有没有召回正确 chunk | 粗召回能力 |
| hybrid_rrf | MRR@10 | 0.348 | 第一个正确 chunk 排得多靠前 | 事实型问答 |
| hybrid_rrf | NDCG@10 | 0.394 | 多个正确 chunk 是否整体排得靠前 | 对比型、汇总型、多证据问题 |

| **Route** | **Query Type** | **Count** | **Recall@3** | **Recall@5** | **Recall@10** | **MRR@10** | **NDCG@10** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | fact | 85 | 0.4941 | 0.5529 | **0.6588** | **0.4318** | **0.4859** |
| hybrid_weightsum | compare | 20 | 0.0500 | 0.0500 | 0.0500 | 0.2792 | 0.2925 |
| hybrid_weightsum | summary | 20 | 0.0000 | 0.0000 | 0.0000 | 0.1133 | 0.1646 |
| hybrid_rrf | fact | 85 | 0.4941 | 0.5412 | 0.6353 | 0.4258 | 0.4760 |
| hybrid_rrf | compare | 20 | 0.0500 | 0.0500 | 0.0500 | 0.2597 | 0.2891 |
| hybrid_rrf | summary | 20 | 0.0000 | 0.0000 | 0.0000 | 0.1083 | 0.1501 |

加入 reranker 后，Hybrid-WeightSum 的 Overall Strict Recall@10 从 0.4160 提升至 0.4560，说明 reranker 对候选结果排序具有一定优化作用。其中 fact 类型 Recall@10 达到 0.6588，明显高于 compare 和 summary，表明当前系统对单证据事实型问题具备较好的召回能力。

但 compare 和 summary 的 Strict Recall@10 仍然较低，分别为 0.0500 和 0.0000。结合 MRR@10 和 NDCG@10 可见，系统并非完全无法召回相关证据，而是难以在 Top10 内完整覆盖多条 gold evidence。因此，当前瓶颈主要在多证据完整召回，而非单点事实召回。

此外，部分 fact 失败样本中，Top10 实际召回了同文档内可回答问题的等价证据，但由于 block_id 与 gold 标注不一致，被 strict recall 判为未命中。这说明当前评测标准较严格，后续需要对重复证据、风险提示和投资建议类样本补充等价 gold evidence，以避免低估系统真实召回能力。

### 第六步：评测集改进

为了清理等价证据假阴性，针对重复答案样本补充了gold evidence对，补充后的评测集更接近真实 RAG 场景，也更公平地反映系统是否召回了“可回答问题的证据”，而不是死盯某一个原始 block_id。

对应的recall的计算方式根据query的类型不同而不同：

- fact类型：只需要命中gold_evidence_groups任意一组docid-blockid对即可算为命中召回
- compare类型：需要同时命中gold_evidence_groups1中任意一组docid-blockid对+gold_evidence_groups2中任意一组docid-blockid对，两对docid-blockid对才算命中
- summary类型：需要同时命中gold_evidence_groups1中任意一组docid-blockid对+gold_evidence_groups2中任意一组docid-blockid对++gold_evidence_groups3中任意一组docid-blockid对，三对ocid-blockid对才算命中。
- 即：评测逻辑改成“组内 OR、组间 AND”

**结果：**

**Overall Metrics**

| Route | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | 125 | 0.4400 | 0.4880 | 0.5520 | 0.4086 | 0.5206 |
| hybrid_rrf | 125 | 0.4400 | 0.4800 | 0.5280 | 0.4049 | 0.5088 |

**Metrics By Query Type**

| Route | Type | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | fact | 85 | 0.6118 | 0.6588 | 0.7529 | 0.5838 | 0.6132 |
| hybrid_weightsum | compare | 20 | 0.1000 | 0.2000 | 0.2000 | 0.0558 | 0.4152 |
| hybrid_weightsum | summary | 20 | 0.0500 | 0.0500 | 0.0500 | 0.0167 | 0.2324 |
| hybrid_rrf | fact | 85 | 0.6118 | 0.6588 | 0.7176 | 0.5788 | 0.6013 |
| hybrid_rrf | compare | 20 | 0.1000 | 0.1500 | 0.2000 | 0.0542 | 0.4109 |
| hybrid_rrf | summary | 20 | 0.0500 | 0.0500 | 0.0500 | 0.0167 | 0.2133 |

在补充等价 gold evidence 后，评测结果明显提升。Hybrid-WeightSum + reranker 的 Overall Strict Recall@10 达到 0.5520，较旧版评测的 0.4560 明显提高，说明原评测集中存在一定数量的等价证据假阴性。新版评测通过 gold evidence group 的组内 OR、组间 AND 规则，更合理地反映了系统是否召回可回答问题的证据。

从题型看，fact 类型 Recall@10 达到 0.7529，说明当前系统已具备较好的单事实证据召回能力；compare 类型 Recall@10 提升至 0.2000，但仍明显低于 fact，表明跨证据比较问题仍存在证据覆盖不足；summary 类型 Recall@10 仅为 0.0500，说明多证据汇总问题仍是当前系统的主要瓶颈。

对比两种融合方式，Hybrid-WeightSum 在 Overall Recall@10、MRR@10 和 NDCG@10 上均略优于 Hybrid-RRF，主要优势来自 fact 类型问题。这表明在金融研报场景下，公司名、时间、财务指标和数值等关键词匹配信号较强，BM25 权重较高的分数融合策略更适合当前任务。

### **第七步：Badcase分析：**

**各类型分析：**

"query": "新易盛2026年一季度营收和净利润增长情况如何？"

"query": "桂冠电力2026年预计日常关联交易金额是多少？
这个case是因为原始表格没embedding，导致表格中（非表头）的文字信息无法被检索到，后续可进行改进。
commare类型：多召回为cover/table summary，且召回都以其中一个公司和一个指标为主，后面的公司难召回—>必须引入query改写

summary：关键词太多，且都分散在不同的文档和block，没有同时出现多个关键词，因此需要decomposation.

**现在指标低不是因为系统完全召不回来，而是 badcase 主要集中在“召回到同主题/同行业/同模板内容，但没有命中指定公司、指定指标、指定证据组”**

当前系统在 fact 类型上已经具备较好的单证据召回能力，但 badcase 显示，失败样本主要集中在公司名约束不足、财务预测表高度同质化、风险提示模板泛化以及多证据问题只命中部分组件等场景。compare 和 summary 指标较低，并不代表系统完全无法召回相关内容，而是单轮 hybrid + rerank 难以保证多个目标证据组同时覆盖。

**主要针对fact类型优化：**

fact_0009（新易盛）, fact_0018（表格内文字），fact_0019（表格内文字），fact_0022（表格内文字），fact_0038（风险），fact_0047（彩讯股份增强关键标题），fact_0048（表格召回），fact_0049（云从科技标题都没召回内容），fact_0050（铜牛信息，标题都没召回），fact_0052（标题加强，全是其他公司的归母净利润）,fact_0055,fact_0056，fact_0057，fact_0058，fact_0059,fact_0081(公司doc都没召回)，fact_0061，（召回cover）fact_0072（表格内容）,fact_0077,fact_0082,fact_0084,fact_0085
需要考虑：

- cover summary是不是不能召回？
- 表格是不是需要单独嵌入？
- table summary被召回的话link的表格是不是需要输入rerank？
- **公司名约束不够强**
- **reranker 更擅长“语义相关”，不一定擅长“证据完整”**

**修改**：

- 继续针对badcase中等价证据假阴性证据进行评测集补充，得到新的评测集recall_eval_m.json,之前的作为recall_eval_mO.json
- 只对 text_chunk 和 raw_table_chunk 做 embedding，默认 include_chunk_types 已改成这两类。
- text_chunk 的 embedding 文本变成：标题：{title}\n公司：{company}\n{chunk text}
- raw_table_chunk 的 embedding 文本变成：标题：{title}\n公司：{company}\n前后文：{context}\n表名：{caption}\n{raw table}
- 标题/公司 都从 cover_summary_chunk 抽；公司缺失就不拼。
- raw_table_chunk 会从匹配的 table_summary_chunk 里抽 前后文 和 表名，拼进自己的 embedding 文本。
- BM25 主体现在只用 record["text"]，不再用带前缀的 embedding_text。
- BM25 的 bonus 现在按“文档标题 + 公司名”一起建 doc-level BM25，命中后给该文档下所有 indexed chunk 加分。
- 检索返回会自然只剩 text_chunk / raw_table_chunk，因为只有这两类还会被标记为 indexed=True。

```json
  "build_seconds": 75.653,
  "embedding_text_template": {
    "text_chunk": "标题：{title}\\n公司：{company}\\n{text}",
    "raw_table_chunk": "标题：{title}\\n公司：{company}\\n前后文：{context}\\n表名：{caption}\\n{text}"
  },
  "bm25_body_template": {
    "text_chunk": "{text}",
    "raw_table_chunk": "前后文：{context}\\n表名：{caption}\\n{text}"
  },
```

**针对修改rechunk rebuild reeval结果：**

**Overall Metrics**

| Route | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | 125 | 0.6800 | 0.7840 | 0.8160 | 0.6211 | 0.8110 |
| hybrid_rrf | 125 | 0.6880 | 0.7920 | 0.8240 | 0.6225 | 0.8143 |

**Metrics By Query Type**

| Route | Type | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | fact | 85 | 0.9176 | 0.9882 | 1.0000 | 0.8596 | 0.8719 |
| hybrid_weightsum | compare | 20 | 0.3000 | 0.6000 | 0.7000 | 0.1925 | 0.8012 |
| hybrid_weightsum | summary | 20 | 0.0500 | 0.1000 | 0.1500 | 0.0363 | 0.5618 |
| hybrid_rrf | fact | 85 | 0.9176 | 0.9882 | 1.0000 | 0.8596 | 0.8714 |
| hybrid_rrf | compare | 20 | 0.3000 | 0.6000 | 0.7000 | 0.1842 | 0.7973 |
| hybrid_rrf | summary | 20 | 0.1000 | 0.1500 | 0.2000 | 0.0530 | 0.5884 |

总结：

- 公司名/标题增强 + 只保留 text/raw_table + raw_table 加上下文。
- 这不是小幅调参，而是结构性改善。说明之前 badcase 的判断是对的：主要问题不只是 reranker，而是 **公司名、标题、表格上下文、chunk 类型污染**。

在完成等价 gold evidence 补充后，本文进一步针对前期 badcase 对检索构建方式进行了优化：仅保留 text_chunk 与 raw_table_chunk 参与索引，减少 cover_summary_chunk 与 table_summary_chunk 对前排结果的干扰；同时在 embedding 文本中显式加入报告标题和公司名称，并为 raw_table_chunk 补充表名前后文信息；BM25 侧仍以 chunk 正文为主体，并通过文档级标题/公司名匹配提供额外加权。

优化后，Hybrid-RRF + reranker 在新版评测集上取得 Recall@3 0.6880、Recall@5 0.7920、Recall@10 0.8240、MRR@10 0.6225、NDCG@10 0.8143。相比上一版，整体 Recall@10 由 0.5520 提升至 0.8240，说明公司名与标题增强、表格上下文补充以及检索对象过滤能够显著缓解金融研报中公司错配、表格模板相似和摘要类 chunk 干扰等问题。

分题型看，fact 类型 Recall@10 达到 1.0000，说明系统已基本解决单证据事实型问题；compare 类型 Recall@10 达到 0.7000，表明跨对象比较召回能力明显改善；summary 类型 Recall@10 仍仅为 0.2000，说明多证据汇总问题仍是当前系统主要瓶颈，需要进一步引入 query decomposition、multi-query retrieval 或按实体分组召回等策略。

**这版结果已经比较好，说明“标题/公司名增强 + raw_table 检索 + 去掉 summary 类 chunk 干扰 + rerank”的路线是有效的；当前主要短板不再是 fact，而是 summary 的多实体、多组件覆盖。**

默认方案：512/50 + text/raw_table 索引 + 标题/公司名增强 + Hybrid-RRF + bge-reranker-v2-m3

主指标：Strict Evidence Recall@10 = 0.8240

### 第八步：query decompase：分桶召回 + 分桶 rerank + 直接拼接

纯规则识别:

1. compare/summary 是怎么识别的

先做两步：

- 第一步，抽实体，也就是从 query 里找公司名
- 第二步，看 query 里有没有对应的触发词

当前判定规则是：

- compare
    - 抽到 2 个公司名
    - 且 query 包含这些比较词之一：
    - 谁更高 / 谁更低 / 谁更好 / 谁更强 / 谁更优 / 哪个更高 / 哪个更低 / 哪个更好 / 哪个更强 / 哪个更优
- summary
    - 抽到 3 个公司名
    - 且 query 包含这些总结词之一：
    - 归纳总结 / 总结一下 / 归纳一下 / 总结 / 归纳

否则就不走 decompose，直接回退原来的单 query 检索链路。

1. 具体拆分规则是什么

整体思路是：

- 先把原 query 标准化
- 把识别出的公司名从原句里拿掉
- 再把连接词/任务词拿掉
- 剩下的部分作为 aspect
- 最后拼成 公司名 + aspect

compare 的拆分规则：

- 去掉 2 个公司名
- 再去掉这些词：
- 谁更高 / 谁更低 / 谁更好 / 谁更强 / 谁更优 / 哪个更高 / 哪个更低 / 哪个更好 / 哪个更强 / 哪个更优 / 以及 / 相比 / 对比 / 比较 / 和 / 与 / 及 / 、
- 剩下的内容作为指标槽位
- 生成两个子 query

补充一点，公司名来源不只靠 company 字段，还会从 doc_id 里反推一层候选词，见 _build_company_names (line 674)。所以它对你这批“文件名里带公司简称”的研报比较适配。

#### **Flat index**

**Overall Metrics**

| Route | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | 125 | 0.7440 | 0.9040 | 0.9440 | 0.6585 | 0.8641 |
| hybrid_rrf | 125 | 0.7200 | 0.8720 | 0.9280 | 0.6507 | 0.8539 |

**Metrics By Query Type**

| Route | Type | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | fact | 85 | 0.9176 | 0.9882 | 1.0000 | 0.8596 | 0.8719 |
| hybrid_weightsum | compare | 20 | 0.6000 | 0.9000 | 0.9500 | 0.3008 | 0.8341 |
| hybrid_weightsum | summary | 20 | 0.1500 | 0.5500 | 0.7000 | 0.1614 | 0.8613 |
| hybrid_rrf | fact | 85 | 0.9176 | 0.9882 | 1.0000 | 0.8596 | 0.8714 |
| hybrid_rrf | compare | 20 | 0.5000 | 0.8000 | 0.9500 | 0.2783 | 0.8052 |
| hybrid_rrf | summary | 20 | 0.1000 | 0.4500 | 0.6000 | 0.1351 | 0.8284 |

**总结一下：**

在前一版 Hybrid-RRF/WeightSum + reranker 的基础上，本文进一步针对 compare 与 summary 问题引入规则式 query decomposition。对于 compare 问题，将原问题拆分为两个实体子查询并分别召回；对于 summary 问题，将原问题按实体拆分为多个子查询，各子查询独立经过 hybrid retrieval 与 rerank 后，再按照实体配额合并为最终 Top10。

实验结果显示，query decomposition 显著提升了多证据问题的完整召回能力。以 hybrid_weightsum 为例，Overall Recall@10 从 0.8240 提升至 0.9440；compare Recall@10 从 0.7000 提升至 0.9500；summary Recall@10 达到 0.7000。该结果表明，金融研报中的 compare 和 summary 问题主要瓶颈并非单个证据无法检索，而是单 query 检索难以同时覆盖多个实体或多个证据组。通过实体级子查询召回，可以显著缓解多证据覆盖不足的问题。

**HNSW index**

**Overall Metrics**

| Route | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | 125 | 0.7440 | 0.8960 | 0.9440 | 0.6578 | 0.8672 |
| hybrid_rrf | 125 | 0.7200 | 0.8640 | 0.9280 | 0.6500 | 0.8572 |

**Metrics By Query Type**

| Route | Type | Count | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_weightsum | fact | 74 | 0.9189 | 0.9865 | 1.0000 | 0.8872 | 0.8938 |
| hybrid_weightsum | compare | 31 | 0.7097 | 0.9032 | 0.9355 | 0.4253 | 0.8072 |
| hybrid_weightsum | summary | 20 | 0.1500 | 0.5500 | 0.7500 | 0.1698 | 0.8615 |
| hybrid_rrf | fact | 74 | 0.9189 | 0.9865 | 1.0000 | 0.8872 | 0.8933 |
| hybrid_rrf | compare | 31 | 0.6452 | 0.8387 | 0.9355 | 0.4108 | 0.7886 |
| hybrid_rrf | summary | 20 | 0.1000 | 0.4500 | 0.6500 | 0.1434 | 0.8304 |

### 第九步：检索增强生成

生成约束

- 实现引用溯源：回答中标注证据来源段落
- 实现拒答机制：召回分数全低于阈值时返回"我不确定"而不是硬编
- 这个问题面试百分百会问：`你怎么缓解幻觉？` 你就说"证据绑定+低置信度拒答"

针对耗时长的问题：

- 首先把 query encoder 缓存起来，不要在 _embed_query() 里每次 new FlagModel，
再把把脚本改成改成交互式模式避免每次提问都重新初始化 ThreeWayRetriever 和 reranker：给 ThreeWayRetriever 加 embedding model 缓存，把 answer_query.py 改成常驻模式/本地服务模式
- 针对检索链路时间长的问题，将Flat 检索方式改为检索耗时是其0.1倍的HNSW检索方式，大幅提高检索效率。
- 开始运行脚本就直接加载各个模型和缓存初始化，而不是提问到第一个需要查询数据库的问题时才加载。

生成流程：

- 每次运行脚本，新建一个 QAApp，历史从空开始
- 同一次运行内，最近 N 轮对话会保存在内存里
- 每次用户输入 query：
    
    先结合历史做路由，判断 DIRECT 还是 RETRIEVE
    
    只要 Router 判定为 RETRIEVE，脚本就会：
    
    - 一律先调用 plan_rewrite()
    - 用 rewritten_query 作为检索 query
    - 生成回答时仍然保留 用户原始问题 + rewritten_query + 对话历史 + evidences
- 所有不同Route下的回答都会吃到最近几轮历史。

### 第十步：RAGAS框架评测

context_precision：召回内容排得好不好

context_recall：标准答案需要的信息有没有被召回

faithfulness：答案是否被召回内容支持

answer_relevancy：答案是否切题

factual_correctness：答案和标准答案事实是否一致

 noise_sensitivity：会不会被噪声召回干扰

先拆解任务 → 再调用 LLM/embedding 判断 → 再按公式聚合成指标

1. **Faithfulness（忠实度）**
    
    **含义：**
    
    衡量生成答案是否能够被召回上下文（Context）支持，是否出现幻觉（Hallucination）。
    
    **计算思想：**
    
    1. 将生成答案拆分为多个事实陈述（Statements）。
    2. 判断每个陈述是否能够从召回上下文中得到支持。
    3. 计算支持陈述占比。
2. **Answer Relevancy（答案相关性）**
    
    **含义：**
    
    衡量生成答案是否真正回答了用户问题。
    
    **计算思想：**
    
    RAGAS会利用LLM反向生成问题（Question Generation），比较：
    
    - 用户原始问题
    - 从答案反推出的问题
    
    二者语义越接近，得分越高。
    
3. **Context Precision（上下文精确率）**
    
    **含义：**
    衡量召回内容中有多少是真正有用的。
    
    类似于信息检索中的 Precision。
    
4. **Context Recall（上下文召回率）**
    
    **含义：**
    
    衡量召回内容是否覆盖回答问题所需的信息。
    
    类似于信息检索中的 Recall。
    

#### **Overall结果 & By-Type结果**

| **Question Type** | **Faithfulness** | **Answer Relevancy** | **Context Precision** | **Context Recall** |
| --- | --- | --- | --- | --- |
| Fact | **0.907** | **0.923** | **0.904** | **1.000** |
| Compare | 0.580 | 0.828 | 0.000 | 0.767 |
| Summary | 0.267 | 0.751 | 0.007 | 0.450 |
| **Overall** | **0.827** | **0.880** | **0.614** | **0.874** |

#### 修改改进：

1.如果我将compare和summary召回时按照子query召回chunk分数两两交替（compare）或三三交替（summary）进行排布，不使重复证据扎堆靠前，即可能够极大提升context_precision。

交替排布本质上是在优化：**前排覆盖度、去重性、多子任务均衡曝光。**

这对 RAGAS 的 context_precision 通常很友好。

2. 数字一致但 RAGAS 仍打 faithfulness=0：说明RAGAS评分时没有带上标题（里面含有公司名称）导致得不到充分的chaim对齐，如果将标题和公司添加到chunk前就能提升指标。

3. 单位换算问题：compare和summary太多单位换算对比但是答案没显式换算或上下文没直接写“谁更高”导致这两类faithfulness被严格判低分。

这类题建议：生成前做标准化单位，或在 prompt 中要求：“如单位不同，先换算再比较，并在答案中显式写出换算结果”

4. 是不是faithfulness低也和生成模型qwen-8B模型能力有关系。

**总结：**

compare 类：多数是 答案相关性高，但上下文排序/分布不适合 RAGAS 的 precision 判法
summary 类：多数是 模型把正文里的负面描述、利润承压、业务不确定性，扩写成“风险提示”，这会直接打掉 faithfulness

### 故障处理：

1. **标题语义缺失比较合理的目标：**
    
    背景：由于对公司的研究文档中经常出现“该公司”“公司”等代称，导致chunk语义不包含公司名字，无法进行精确索引。
    
    解决方法：
    
    - **向量检索：**针对向量检索显式为每一个chunk添加meta信息使其包含重要的标题信息。
        
        注意⚠️：将meta信息也能够嵌入到每一个chunk embedding中
        
        ```json
          "embedding_text_template": {
            "text_chunk": "{title}\\n所属章节：{section}\\n所属小节：{subsection}\\n{text}",
            "table_summary_chunk": "所属章节：{section}\\n所属小节：{subsection}\\n{text}",
            "cover_summary_chunk": "{text}",
            "raw_table_chunk": "{text}"
          },
        ```
        
    - **BM25检索：**不能像向量检索一样每个chunk添加标题，因为chunk数多了会增大标题词的document frequency ，IDF 会变小，稀释了区分度。因此使用标题提权或额外bonus：
        
        对每个 chunk：
        
        - BM25 主体仍然只用 chunk 正文
        - 另外，如果 query 命中文档标题词，给该文档下所有 chunk 一个额外 bonus
        
        公式可以很简单：
        
        `final_score = bm25_body_score + alpha * title_match_score`
        
        其中 title_match_score 可以是：
        
        - 标题命中词个数
        - 标题 BM25 分数
        - 标题 token 覆盖率
        
        这样标题能提召回，但不会因为被复制到每个 chunk 而污染 chunk 语料统计。
        
    
    1. **任务难度分析**
        1. **金融研报比普通文本难：**
        - 数字多：收入、利润、同比、环比、毛利率、预测值
        - 表格多：很多答案藏在表格或 OCR 解析结果里
        - 公司名多：不同报告反复出现同一公司
        - 指标相似：营收、归母净利润、扣非净利润、毛利率容易混淆
        - 时间粒度细：2025、2026Q1、2026E、2027E 很容易召错
        - 多文档重复：不同研报会讲相似事件，但数值和口径不同
        
        所以它天然比普通 FAQ、百科、政策文本更难。
        
        **b. block-level strict recall  🆚 document-level recall：**
        
        - 普通RAG只需要召回相关文档即可；
        - 而我的召回需要精确到block level；
        - 且compare 和 summary 是全证据覆盖，属于严格的Recall指标。
        
        c. **金融研报模板化太强**
        
        很多 chunk 都包含：
        
        盈利预测
        投资建议
        风险提示
        2026E
        2027E
        归母净利润
        营业收入
        毛利率
        买入评级
        
        这些词在大量文档中重复出现，BM25 和 reranker 都容易召回“形式相似”的 chunk，而不是“目标公司 + 目标指标”的 chunk。
        
        **比较合理的目标：**
        
        - Fact 是金融 RAG 最基础能力，最终最好能把 Fact@10 提到 **0.70 以上**。
            
            All Strict@10：0.50+
            
            Fact Strict@10：0.70+
            
            Compare Strict@10：0.20+
            
            Summary Strict@10：0.10+
            
        
    2. **表格summary的局限**
        
        "query": "桂冠电力2026年预计日常关联交易金额是多少？
        这个case是因为原始表格没embedding，导致表格中（非表头）的文字信息无法被检索到。
        
        因此，不能因为表格数据太多而不进行嵌入，有些表格是有大量文字信息的，可能会出现表格表头信息覆盖不全文字信息，因此raw表格也需要嵌入索引帮助解决这类badcase。
        
    3.  **strict 假阴性**
        
        例如 `fact_0006`：
        
        问题：空天产业链面临哪些主要风险？
        gold block：b0016
        
        但 Top10 里第 2 名已经召回了同一篇文档里的风险提示内容，文本中包含可回答内容只是不匹配gold block。
        
        这说明：**系统可能已经召回了能回答问题的等价证据，但因为 block_id 不一致，被 strict recall 判成 0。这种不是检索能力问题，而是评测集 gold evidence 标注过窄。**
        
        **解决方案：给 fact 失败样本做人工抽查，把等价block把该 block 加入 gold_evidence 的可接受证据集合。但是不能保证golden答案覆盖全面，因此可能仍有一定程度的标注过窄导致结果指标偏低。**
        
        **补充后还有缺陷：**由于是block级召回，因此答案藏在overlap里的召回无法计入，导致指标偏低，而补充评测集后由于相同的答案可能会在多篇文章中找到，因此指标可能有偏高。综上所述，指标只能起到一定程度的参考价值但不能完全精确地反应召回效果，后续可通过RAGAS评测RAG系统的效果。
        
        以及summary 的 Strict Recall@10 是 0，但 MRR@10 和 NDCG@10 不是 0：
        
        summary Recall@10 = 0
        summary MRR@10 = 0.1133
        summary NDCG@10 = 0.1646
        
        summary 没有完整召回 3/3 个 gold evidence；
        
        但 Top10 里经常命中了其中一部分证据。
        
        按输出文件算了一下辅助指标：
        
        | **类型** | **Strict@10** | **Partial@10** | **Any-hit@10** |
        | --- | --- | --- | --- |
        | fact | 0.6588 | 0.6588 | 0.6588 |
        | compare | 0.0500 | 0.2000 | 0.3500 |
        | summary | 0.0000 | 0.1500 | 0.3000 |
        
        compare：35% 的问题至少召回了一个证据，但只有 5% 召回了全部 2 个。
        
        summary：30% 的问题至少召回了一个证据，但没有一个召回全部 3 个。
        
    4. **chunk语义污染**
    按照1.标题语义缺失比较合理的目标中所述，为了加大chunk中信息的完整度，在chunk中家了标题（标题中没公司名）但是没加公司，分chunk时多加了cover summary和table summary以为可以高度概括文章信息和表格信息，实则对badcase分析后发现这些summary只有相关内容无具体答案，给检索带来了干扰和污染。通过第七步badcase分析和结构性改善指标显著提升，说明之前 badcase 的判断是对的：主要问题不只是 reranker，而是 **公司名、标题、表格上下文、chunk 类型污染**。 
        
        因此改进方法就是只需要将关键的纯标题信息，公司信息拼接到向量嵌入文本/表格开头即可，丢弃到干扰的summary类型chunk，直接将tablesummary中关键的“表格名””前后文“加到raw table chunk后进行直接嵌入即可。针对BM25，延续之前的命中标题和公司名后额外添加bonus提权提高分数即可。
        

### 相关资料

| **模型** | **输出向量维度** | **大致参数量** | **适合场景** |
| --- | --- | --- | --- |
| **bge-small-zh-v1.5** | **512维** | ~24M | 速度快、显存小、轻量检索 |
| **bge-base-zh-v1.5** | **768维** | ~102M | 性能和速度折中 |
| **bge-large-zh-v1.5** | **1024维** | ~326M | 效果更好，但更慢、更占 |

| **指标** | **关注点** | **适合场景** |
| --- | --- | --- |
| Recall@10 | Top 10 里有没有召回正确 chunk | 粗召回能力 |
| MRR@10 | 第一个正确 chunk 排得多靠前 | 事实型问答 |
| NDCG@10 | 多个正确 chunk 是否整体排得靠前 | 对比型、汇总型、多证据问题 |
- MRR 全称是 **Mean Reciprocal Rank**，中文可以叫**平均倒数排名**。它看的是：**第一个正确答案出现在第几名**。
- NDCG 全称是 **Normalized Discounted Cumulative Gain**，中文可以叫**归一化折损累计增益**。它看的是：**多个相关 chunk 是否整体排得靠前**。

**BGE Reranker常见型号：**

| **模型** | **语言** | **特点** | **适合场景** |
| --- | --- | --- | --- |
| bge-reranker-base | 中英 | 较小，速度较快 | 轻量 RAG、资源有限 |
| bge-reranker-large | 中英 | 精度更高，但更慢 | 中文/英文 RAG 精排 |
| bge-reranker-v2-m3 | 多语言 | 新一代轻量多语言 reranker，部署方便 | 中文金融 RAG 很推荐 |
| bge-reranker-v2-gemma | 多语言 | LLM-based reranker，效果强但更重 | 高精度、资源充足 |
| bge-reranker-v2-minicpm-layerwise | 多语言 | 支持选择中间层输出，加速推理 | 需要速度/效果折中 |
| bge-reranker-v2.5-gemma2-lightweight | 多语言 | 支持 token compression 和 layerwise reduction | 更先进的轻量化 reranker |

Langchainchatchat功能：

历史对话轮数/匹配知识条数

知识库管理：新建知识库/上传知识文件/请输入知识库介绍/文件处理配置：单段文本最大长度和相邻文本重合长度：

知识库 samples 中已有文件:包含序号，文档名，文档加载器，分词器，文档数量，源文件，向量库

知识库可进行操作：从向量库删除，从知识库删除，（重新添加至向量库，下载选中文档）

查看一下上下文指代是怎么做的，以及query是怎么改写的：

它的“上下文指代”主要不是靠检索前的 query rewrite 做的，而是靠“把历史消息直接拼进最终回答 prompt”让 LLM 在生成答案时自己理解“它/这个/上一个”等指代。

也就是说：

1. 检索阶段：基本直接用当前轮 query 去查知识库/临时库/搜索引擎。
2. 生成阶段：把 history + 当前问题模板 一起送给 LLM，让模型结合上下文回答。

RAGFlow功能：

可以指代，也可以超长上下文，但是不能同时查找两篇或三篇文章

可以设置开场白，可以参考其system提示词使回答不那么生硬

可以学习一下其直接回答和根据RAG回答的路由是怎么设置的

可以学习一下其上下文历史是如何做到这么长但是不OOM的
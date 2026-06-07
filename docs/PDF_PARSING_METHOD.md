# 金融研报 PDF 解析方法

本阶段目标是为后续 RAG 问答生成可追溯、可检阅、表格不被切断的解析结果。当前保留两个可直接运行的文件：`RAG/pdf_parser/deepdoc_parser.py` 调用 RAGFlow DeepDoc；`RAG/pdf_parser/legacy_parser.py` 是旧版启发式解析器。

## 默认 DeepDoc 解析流程

1. 调用 `RAGFlowPdfParser.parse_into_bboxes()` 完成 PDF 渲染、OCR、版面识别、表格结构识别和坐标回填。
2. 将 DeepDoc 输出的 `layout_type/page_number/positions/text` 归一化为本项目的 `blocks`。
3. 表格和图像作为独立 block 保留，不在文本 chunk 中被切碎。
4. 生成 `*.parsed.json`、`*.review.md`、`*.review.html`。其中 HTML 会嵌入页面底图，并按 `text/title/table/figure` 叠加可点击 bbox，便于肉眼检查版面识别和表格区域是否完整。
5. DeepDoc 依赖和模型需要按 `ragflow/pyproject.toml` 准备；当前 RAGFlow 配置声明 Python `>=3.13,<3.15`。若环境暂未就绪，可用 `--method legacy` 跑旧解析器。

运行示例：

```bash
ragflow/.venv/bin/python RAG/pdf_parser/deepdoc_parser.py RAG/data/it_service_pdfs --max-files 3
```

旧版解析器：

```bash
python RAG/pdf_parser/legacy_parser.py RAG/data/it_service_pdfs --output-dir RAG/outputs/pdf_parse_legacy --max-files 3
```

## Legacy 解析流程

1. 使用 PyMuPDF 提取每页文本行、bbox、字号等几何信息。
2. 使用 pdfplumber 独立检测并抽取表格，保留表格 bbox 和二维行列数据。
3. 表格保护：凡是与表格 bbox 有明显重叠的文本行从正文中移除，表格作为单独 block 保留。
4. 图表保护：识别 `图表/图/表/◆图` 标题到来源之间的区域；如果没有来源，则用图标题到下一块图表标题前的半页区域作为 fallback。图内坐标、刻度、纯数字不进入正文，统一保留为 figure block。
5. 识别重复页眉页脚：统计每页顶部 12% 和底部 10% 的重复文本，重复达到阈值后过滤。
6. 多栏识别：优先用正文行 x0 的真实分布判断左右栏；只有 x0 分布足够分散时，才用中心点间隔兜底，避免把同一栏长短行误判成两栏。
7. 文本合并：同页、同栏、几何距离接近、水平重叠合理的文本行合并为段落 block。
8. 阅读顺序：全宽标题/表格作为版面锚点；两栏内容按左栏从上到下、右栏从上到下排序。
9. 表格保护切块：chunk 时 table block 是原子单元，不拆行、不切断；如果表格过大，允许单独 chunk 超过目标长度。
10. 输出 JSON、Markdown、HTML 三种检阅结果，便于肉眼检查解析质量。

## 与 DeepDoc 思路的关系

DeepDoc 的主链路是：PDF 渲染图片、OCR 检测文字框、PDF 字符层融合、layout 模型识别标题/正文/表格/图片、TSR 表格结构识别、文本框合并、阅读顺序排序、位置标签输出。

本项目第一版不引入 OCR 和 ONNX layout 模型，而是把 DeepDoc 的关键思想转成轻量启发式：

- DeepDoc 用 OCR/text boxes 作为基础几何单元；本项目用 PyMuPDF text lines 作为基础几何单元。
- DeepDoc 用 layout.onnx 判断 `text/title/table/figure/header/footer`；本项目用 pdfplumber 表格 bbox、重复页眉页脚统计、字号/宽度/位置规则近似判断。
- DeepDoc 对 table 区域再跑 Table Structure Recognizer；本项目用 pdfplumber 的表格抽取结果保留二维行列。
- DeepDoc 通过视觉 layout 模型识别图片/图表区域；本项目用标题、来源、数字密度和半页 fallback 区域保护图表，策略更轻但也更依赖研报排版规则。
- DeepDoc 用 `layout_type/layoutno` 和几何距离合并文本；本项目用 `page/col_id/bbox/font_size` 做文本合并。
- DeepDoc 会输出坐标标签用于引用高亮；本项目在每个 block/chunk 中保留页码和 bbox，后续可接引用定位。

## 输出文件

默认输出目录：

```text
RAG/outputs/pdf_parse_deepdoc/
```

每份 PDF 会生成：

- `*.parsed.json`：结构化结果，供后续入库和程序处理。
- `*.review.md`：按 block/chunk 展开的人工检阅文件。
- `*.review.html`：浏览器查看版，嵌入页面底图并叠加可点击 bbox；表格 block 会以绿色边线标出。
- `index.json`：批量解析汇总。

## 当前限制

- DeepDoc 默认依赖 RAGFlow 的 Python 3.13 虚拟环境和本地模型资源；请优先使用 `ragflow/.venv/bin/python` 运行。
- 当前适配层把 DeepDoc box 归一化为本项目通用 `blocks/chunks`，后续入库阶段仍可继续细化表格 HTML、图片 caption、引用截图等字段。
- legacy 解析器仍依赖 PyMuPDF/pdfplumber，适合回归对比和快速兜底。

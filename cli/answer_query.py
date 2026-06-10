#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
from urllib import parse as urllib_parse
from urllib import error as urllib_error
from urllib import request as urllib_request

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from common.paths import DEFAULT_XINFERENCE_ENDPOINT, DEFAULT_XINFERENCE_MODEL, INDEX_ROOT, QA_OUTPUT_ROOT, REPO_ROOT
from retrieval import ThreeWayRetriever
from retrieval.three_way_retriever import RetrievalHit


OUTPUT_JSON_DIR = QA_OUTPUT_ROOT


ROUTER_SYSTEM_PROMPT = """你是金融RAG系统的 Query Router。
任务：结合最近对话历史，判断当前最新用户问题是否需要检索本地金融文档库。
你只能输出严格 JSON，不要输出 Markdown，不要输出多余解释。
判定规则：
1. 如果当前问题或结合历史后的真实问题涉及具体公司、股票、证券、券商研报、行业研究、财务数据、经营情况、业务布局、市场表现、估值、业绩、营收、利润、毛利率、同比/环比、产业链、政策影响、投资观点、风险提示等，输出 RETRIEVE。
2. 如果当前问题是对金融、技术、RAG、编程、算法、常识等通用概念的解释，且不需要依赖本地金融文档库，输出 DIRECT。
3. 如果当前问题是“那它呢”“继续”“和谁相比”“具体说说”等追问，必须结合历史判断其真实意图；只要真实意图属于金融研究问题，输出 RETRIEVE。
4. 如果用户要求“根据文档/研报/资料/数据库/检索结果回答或查询知识库/数据库”，输出 RETRIEVE。
5. 如果不确定，优先输出 RETRIEVE。
输出格式必须为：
{
  "decision": "RETRIEVE" 或 "DIRECT",
  "reason": "一句简短原因"
}
"""


REWRITE_SYSTEM_PROMPT = """你是金融RAG系统的 Query Rewrite Planner。

任务：结合最近对话历史，将当前用户问题改写成一个独立、完整、适合检索的金融研报问题，并判断其问题类型。

注意：
1. 你只负责问题改写和类型判断。
2. 不要进行 query decomposition。
3. 不要输出多个子问题。
4. compare 和 summary 的后续拆分会由规则模块完成。
5. 你的目标是“必要时补全”，不是“强行重写”。

你只能输出严格 JSON，不要输出 Markdown，不要输出多余解释。

核心目标：
1. 将当前问题改写成一个适合检索的独立问题。
2. 仅在当前问题存在指代、省略、上下文依赖时，才结合历史进行补全。
3. 如果当前问题本身已经完整、明确、可检索，应保持原意，只做轻微规范化，甚至可以不改写。
4. 保留用户原始意图，不扩大问题范围，不改变问题类型。
5. 不编造历史中不存在的公司、行业、年份、指标、比较对象或结论。
6. 判断改写后的问题属于 fact、compare、summary 三类之一。

最高优先级规则：避免过度改写
1. 在改写前，必须先判断当前用户问题是否已经是一个独立、完整、可检索的问题。
2. 如果当前问题已经明确包含以下关键信息中的主要部分：
   - 主体：公司、行业、板块、政策、标的、报告对象等；
   - 指标或意图：收入、利润、毛利率、风险、评级、投资建议、业绩表现、政策影响、原因、对比、汇总等；
   - 时间或范围：季度、年份、预测期、报告期等，若问题本身不需要时间则可没有；
   则认为当前问题已经完整。
3. 对于已经完整的问题，不要继承上一轮的 query 模板。
4. 对于已经完整的问题，不要把上一轮的公司、指标、时间、比较关系、总结范围强行带入当前问题。
5. 对于已经完整的问题，rewritten_query 可以与原问题完全相同，或只进行轻微规范化。
6. 只有当前问题明显属于短追问、指代追问、省略追问时，才允许继承历史模板。
7. 不要因为历史中存在稳定模板，就默认当前问题也要套用该模板。

需要继承历史模板的情况：
只有当当前问题存在明显上下文依赖时，才继承历史模板。例如：
1. 短追问：
   - “那它呢”
   - “这家公司呢”
   - “它的风险呢”
   - “那营业收入呢”
   - “和它相比呢”
   - “具体有哪些”
   - “原因是什么”
2. 当前问题缺少关键主体：
   - “它的归母净利润增速是多少？”
   - “风险提示有哪些？”
   - “谁更高？”
3. 当前问题缺少比较对象：
   - “和它相比呢？”
   - “哪个更好？”
4. 当前问题缺少被总结对象：
   - “这些公司呢？”
   - “都有哪些风险？”

不应继承历史模板的情况：
如果当前问题已经是完整问题，即使历史中存在上一轮 effective_query，也不要套用。例如：
1. 当前问题：“神州数码提示了哪些主要风险？”
   - 应保持为：“神州数码提示了哪些主要风险？”
   - 不要改写成上一轮其他公司的收入、利润、评级或盈利预测问题。
2. 当前问题：“紫光股份和神州数码2026年一季度归母净利润增速谁更高？”
   - 应保持为该对比问题。
   - 不要套用上一轮某个 summary 模板。
3. 当前问题：“归纳总结能科科技、并行科技以及卓易信息的风险有哪些。”
   - 应保持为该 summary 问题。
   - 不要改写成上一轮盈利预测、评级或投资建议模板。
4. 当前问题：“寒武纪2026年一季度收入和归母净利润是多少？”
   - 应保持为该 fact 问题。
   - 不要继承历史中的风险、评级、目标价等模板。

处理指代追问的强规则：
1. 如果当前问题属于短追问、指代追问或省略追问，例如“那它呢”“京仪装备的呢”“那这家呢”“它的风险呢”“那营业收入呢”，才可以继承最近一轮有效上下文 query 的问题模板来改写。
2. 这里的“有效上下文 query”优先指最近一轮历史中的 effective_query；如果最近一轮本身也是追问或没有形成稳定模板，就继续向上继承最近的、语义完整的 effective_query 模板。
3. 继承模板时，优先保持上一轮问题的指标、时间、比较关系、总结范围、问题类型不变，只替换当前轮显式提到的新主体、新公司或新对象。
4. 不要因为历史中出现过别的指标，就把当前追问错误改写成新的指标问题。
5. 如果当前问题已经显式给出了新的指标或约束，例如“风险提示呢”“营业收入呢”“谁更高”，则在继承模板时必须以当前轮显式信息为准，不能被更早历史覆盖。
6. 如果当前问题显式给出了完整的新主体和新指标，则不属于追问，不得继承历史模板。

三类 query 定义：

1. fact：事实型问题
- 用于查询单个明确事实，或同一个对象/同一个证据范围内即可支撑的简单事实。
- 通常只需要 1 个核心证据即可回答。
- 适合问题包括：营业收入、归母净利润、毛利率、同比增速、评级、目标价、分红、装机容量、发电量、公司事件、政策时间、项目规模、风险提示、投资建议中的明确条目等。
- 如果问题中包含“分别是多少”“哪个更高”等表达，但所有信息通常位于同一个对象或同一个证据块内，仍可归为 fact。
- 如果问题只问一个公司/一个行业/一个板块的多个风险、多个原因、多个指标，只要答案主要来自单个证据范围，也优先归为 fact。
- 示例：寒武纪2026年一季度收入和归母净利润是多少？
- 示例：神州数码提示了哪些主要风险？

2. compare：对比型问题
- 用于比较两个对象、两个指标、两个时间点、两个行业、两个政策、两个事件或两个投资逻辑。
- 通常需要 2 个核心证据共同支撑。
- 问题应同时涉及双方信息，并需要给出比较结论。
- 适合问题包括：两家公司营收、利润、毛利率、增速对比；两个行业或板块表现对比；两个时间点的数据变化对比；两个政策或投资逻辑差异对比。
- 如果问题中出现“谁更高”“哪个更好”“相比如何”“差异是什么”“对比”“比较”等，并且比较对象为两个，通常归为 compare。
- 如果两个比较对象的信息明显需要分别检索不同证据，则归为 compare。
- 示例：紫光股份和神州数码2026年一季度归母净利润增速谁更高？

3. summary：多对象同类信息汇总型问题
- 用于对三个对象、公司、标的、行业板块或证据来源进行同一类信息的归纳汇总。
- 重点不是比较谁更高或谁更好，而是分别提取多个对象的关键信息，并组织成一个综合答案。
- 通常需要 3 个或更多核心证据共同支撑。
- 适合问题包括：
  1）三个公司的风险提示汇总；
  2）三个公司的盈利预测与评级汇总；
  3）三个公司的投资建议及看好理由汇总；
  4）三个公司的成长逻辑或长期价值支撑汇总；
  5）三个公司的收入、利润、经营表现或业绩表现汇总；
  6）三个标的、板块、政策、产业链机会的多证据归纳。
- 如果问题要求“归纳总结”“汇总”“分别说明”“主要有哪些”，且涉及三个或以上对象或多个证据点，通常归为 summary。
- 如果只有一个对象、一个指标、一个事实，即使问“有哪些”，也优先归为 fact。
- 如果只有两个对象且要求判断谁更高、谁更好、差异是什么，优先归为 compare。
- 示例：归纳总结能科科技、并行科技以及卓易信息的风险有哪些。
- 示例：归纳总结彩讯股份、软通动力和博彦科技的2026-2028年盈利预测与评级。
- 示例：归纳总结华峰测控、炬芯科技和长电科技2026年一季度的业绩表现。

改写规则：
1. 对于进入本模块的问题，一律视为需要为检索生成 rewritten_query，因此 need_rewrite 固定输出 true。
2. 如果当前问题本身已经完整，rewritten_query 可以与原问题相同，或只做轻微规范化改写。
3. 如果当前问题是追问，例如“那它呢”“和它相比呢”“具体有哪些”“原因是什么”“这些公司呢”，必须结合历史上下文补全主体、时间、指标或比较对象。
4. rewritten_query 必须是一个单一问题，不要拆成多个问题。
5. rewritten_query 不要直接包含答案，不要泄漏检索结果。
6. query_type 必须根据 rewritten_query 判断，只能是 fact、compare、summary 三者之一。
7. 如果 fact / compare / summary 难以判断：
   - 单对象、单事实、单证据范围可回答，优先 fact；
   - 两个对象或两个维度需要比较并得出比较结论，优先 compare；
   - 三个或以上对象需要围绕同一类信息进行归纳汇总，优先 summary。

输出格式必须为：
{
  "need_rewrite": true,
  "reason": "一句简短原因，说明是轻微规范化、保持原问题，还是结合历史补全",
  "query_type": "fact" 或 "compare" 或 "summary",
  "rewritten_query": "改写后的独立检索问题"
}
"""


ANSWER_SYSTEM_PROMPT = """你是一个严格依据检索证据回答的金融RAG助手。

任务：根据给定证据回答用户问题。你只能输出严格 JSON，不要输出 Markdown，不要输出多余解释。

基本规则：
1. 依据输入中的证据回答，不允许编造或引入证据之外的事实。
2. 所有涉及公司、年份、财务数据、业务判断、行业观点、政策影响、估值、风险、结论的陈述，都必须绑定至少一个证据ID。
3. 不允许编造数字、页码、公司信息、时间、指标、结论或证据ID。
4. 如果证据可以部分回答问题，应给出可支持的部分，并明确说明哪些部分证据不足。
5. 如果完全没有相关证据，answer 写“很抱歉，我没有检索到相关内容。”，should_refuse=true。
6. 如果证据存在明显矛盾，answer 说明“检索到的证据存在不一致，无法给出确定结论。”，should_refuse=true。
7. 如果问题无法从证据中推出答案，answer 说明“现有证据不足，无法判断。”，should_refuse=true。
8. claims 中的每一条 claim 必须是单条可核验陈述，不能把多个无关结论合并成一条。
9. citations 只能使用输入证据中真实存在的证据ID，例如 E1、E2。
10. answer 必须是面向用户的完整回答，承载主要信息，不能只写一句总括结论然后把细节都留到 claims。
11. claims 只是用于可核验引用，不是 answer 的替代品；answer 必须自然地展开核心细节，claims 只做evidence参考。
12. 如果涉及数字/指标/单位，必须逐字保留证据中的原始数值和原始单位。
13. 禁止自行单位换算、禁止补零、禁止改变小数点位置、禁止把“亿元”换算成“万元/百万元/十亿元”。

数值和单位使用规则：
1. 回答中的所有数值和单位必须来自 evidence。
2. 不得自行进行单位换算、增长率计算、同比计算、差值计算、倍数计算。
3. 如果用户没有要求换算单位，必须保留 evidence 中的原始单位。

confidence 判断：
- high：关键结论有直接证据支持，且证据之间一致。
- medium：证据能支持主要结论，但细节不完整。
- low：只有部分证据支持，或证据较弱，或只能做有限回答。
- 如果 should_refuse=true，confidence 必须为 low。

输出格式必须为：
{
  "answer": "对用户的自然语言回答",
  "claims": [
    {
      "claim": "单条可核验陈述",
      "citations": ["E1", "E2"]
    }
  ],
  "used_evidence": ["E1", "E2"],
  "confidence": "high" 或 "medium" 或 "low",
  "should_refuse": true 或 false
}
"""


DIRECT_SYSTEM_PROMPT = """你是一个有用的助手。

任务：结合最近对话历史，直接回答不需要检索本地金融文档库的通用问题。

适合直接回答的问题包括：
1. 通用概念解释，例如“什么是毛利率”“什么是RAG”“什么是BM25”。
2. 通用方法说明，例如“RAG怎么做多轮对话”“MRR@10怎么计算”。
3. 其他不依赖本地研报、财务数据、公司资料或实时信息的问题。

规则：
1. 如果当前问题依赖对话历史，可以结合历史理解后回答。
2. 如果问题涉及具体公司、股票、行业研报、财务数据、估值、业绩、市场表现、政策影响、券商观点等，应明确说明该问题需要检索本地金融文档库，不能直接回答。
3. 不要编造具体公司数据、研报观点、时间、数值或结论。
4. 不要自称通义千问、Qwen 或其他外部产品身份，统一以“本地金融问答助手”身份回答。
5. 回答风格要自然、友好、耐心，像一个愿意帮忙的真实助手，而不是冷冰冰的系统播报。
6. 在保证准确和保守的前提下，优先使用自然表达，避免过于机械、僵硬、只有一句话的回答。
"""


NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?|\d{4}年|\d{4}Q[1-4]")
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.S)
COMPANY_HINT_PATTERN = re.compile(r"\d{6}\.(?:SH|SZ)|\d{6}|[A-Za-z]{1,5}\.[A-Z]+")


FINANCE_TERMS = {
    "公司", "股票", "个股", "证券", "研报", "券商", "行业", "赛道", "财报", "年报", "季报",
    "营收", "收入", "利润", "归母", "毛利率", "净利率", "估值", "市盈率", "pe", "pb", "roe",
    "同比", "环比", "订单", "出货", "资本开支", "capex", "算力", "半导体", "电力", "新能源",
    "龙头", "受益", "催化", "风险提示", "股价", "业务", "产能", "市占率", "现金流", "分红",
}

DIRECT_TERMS = {
    "什么是", "是什么意思", "原理", "区别", "怎么理解", "介绍一下", "是什么", "如何工作",
    "rag", "llm", "transformer", "attention", "向量数据库", "embedding", "prompt", "agent",
    "faiss", "bm25", "rerank", "xinference",
}

DIRECT_CHAT_PATTERNS = (
    "你是谁",
    "你是什么",
    "你能做什么",
    "你会什么",
    "你叫什么",
    "我是谁",
    "你还记得我是谁",
    "你记得我是谁",
)

STOPWORDS = {
    "的", "了", "和", "与", "及", "是", "在", "对", "把", "将", "并", "或", "一个", "一种",
    "可以", "是否", "进行", "有关", "这个", "那个", "以及", "我们", "你们", "他们", "因为",
    "所以", "问题", "回答", "说明", "认为", "显示", "指出", "提到", "根据",
}

PROMPT_HISTORY_ROUNDS = 3

@dataclass
class RouterDecision:
    decision: str
    reason: str
    source: str


@dataclass
class EvidenceItem:
    evidence_id: str
    doc_id: str
    chunk_id: str
    source_pdf: str
    page_start: Optional[int]
    page_end: Optional[int]
    reranker_score: float
    fused_score: float
    chunk_type: str
    text: str


@dataclass
class TimingRecord:
    stage: str
    seconds: float


@dataclass
class RewriteDecision:
    need_rewrite: bool
    reason: str
    query_type: str
    rewritten_query: str
    source: str


@dataclass
class ConversationTurn:
    user: str
    assistant: str


@dataclass
class QueryRunResult:
    payload: Dict[str, Any]
    saved_path: Path
    router: RouterDecision
    rewrite: Optional[RewriteDecision]
    effective_query: str
    answer_mode: str
    retrieval_hits: List[RetrievalHit]
    final_answer: str
    answer_confidence: str
    should_refuse: bool
    answer_payload: Dict[str, Any]
    evidences: List[Dict[str, Any]]
    timings: List[TimingRecord]


class XinferenceChatClient:
    def __init__(self, endpoint: str, model_uid: str, api_key: Optional[str] = None):
        self.endpoint = self._build_endpoint(endpoint)
        self.model_uid = model_uid
        self.api_key = api_key

    @staticmethod
    def _build_endpoint(raw_endpoint: str) -> str:
        endpoint = raw_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/chat/completions"):
            endpoint = endpoint + "/v1/chat/completions"
        return endpoint

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        text = self.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_json_text(text)

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> str:
        payload = {
            "model": self.model_uid,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib_request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Connection failed: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {json.dumps(body, ensure_ascii=False)}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    @staticmethod
    def _parse_json_text(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = JSON_BLOCK_PATTERN.search(text)
            if match:
                return json.loads(match.group(0))
        raise RuntimeError(f"Model did not return valid JSON: {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end finance QA with query routing, RAG retrieval, refusal, and citation validation.")
    parser.add_argument("--query", help="User query.")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT / "faiss_hnsw_chunked_512_50_bge-large-zh-v1.5")
    parser.add_argument("--endpoint", default=DEFAULT_XINFERENCE_ENDPOINT, help="Xinference endpoint base URL or full chat completions URL.")
    parser.add_argument("--model", default=DEFAULT_XINFERENCE_MODEL, help="Running Xinference model uid.")
    parser.add_argument("--api-key", default=None, help="Optional API key.")
    parser.add_argument("--route", default="hybrid_weightsum", choices=["vector", "bm25", "hybrid", "hybrid_weightsum", "hybrid_rrf"])
    parser.add_argument("--first-stage-top-k", type=int, default=50)
    parser.add_argument("--rerank-top-k", type=int, default=10)
    parser.add_argument("--vector-candidate-k", type=int, default=50)
    parser.add_argument("--bm25-candidate-k", type=int, default=50)
    parser.add_argument("--vector-weight", type=float, default=0.3)
    parser.add_argument("--bm25-weight", type=float, default=0.7)
    parser.add_argument("--title-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rerank-model-path", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-use-fp16", action="store_true")
    parser.add_argument("--rerank-threshold", type=float, default=0.5, help="Reject if all reranker scores are below this threshold.")
    parser.add_argument("--min-valid-claims", type=int, default=1, help="Minimum supported claims required after post-check.")
    parser.add_argument("--max-evidence-chars", type=int, default=900, help="Max chars from each chunk sent to the generator.")
    parser.add_argument("--show-text-chars", type=int, default=9999)
    parser.add_argument("--hide-retrieval-chunks", action="store_true",default=True, help="Do not print retrieved chunks in terminal output.")
    parser.add_argument("--show-used-evidence-text", action="store_true",default=True, help="Print only the text of evidence blocks actually cited as E1/E2/etc.")
    parser.add_argument("--query-decompose", action="store_true", default=True, help="Whether to decompose complex queries for retrieval.")
    parser.add_argument("--show-timings", action="store_true", help="Print per-stage timings.")
    parser.add_argument("--interactive", action="store_true", default=True, help="Keep the process resident and answer queries from stdin.")
    parser.add_argument("--history-rounds", type=int, default=6, help="How many recent conversation rounds to keep in context.")
    parser.add_argument("--output-json-dir", type=Path, default=OUTPUT_JSON_DIR)
    parser.add_argument("--json", action="store_true", help="Print JSON output only.")
    args = parser.parse_args()
    if not args.query and not args.interactive:
        parser.error("either --query or --interactive is required")
    if args.json and args.interactive:
        parser.error("--json cannot be used with --interactive")
    return args


def add_timing(records: List[TimingRecord], stage: str, start_time: float) -> None:
    records.append(TimingRecord(stage=stage, seconds=perf_counter() - start_time))


def heuristic_router(query: str) -> Optional[RouterDecision]:
    normalized = query.strip().lower()
    if any(pattern in query for pattern in DIRECT_CHAT_PATTERNS):
        return RouterDecision(decision="DIRECT", reason="命中助手身份/记忆类通用对话", source="heuristic")
    has_finance = any(term in normalized for term in FINANCE_TERMS) or bool(COMPANY_HINT_PATTERN.search(query))
    has_direct = any(term in normalized for term in DIRECT_TERMS)

    if has_finance and not has_direct:
        return RouterDecision(decision="RETRIEVE", reason="命中金融/公司/行业相关关键词", source="heuristic")
    if has_direct and not has_finance:
        return RouterDecision(decision="DIRECT", reason="更像通用技术/概念问题", source="heuristic")
    return None


def format_history(turns: Sequence[ConversationTurn], max_rounds: int) -> str:
    if not turns or max_rounds <= 0:
        return "无"
    selected = list(turns[-max_rounds:])
    lines: List[str] = []
    for idx, turn in enumerate(selected, start=1):
        lines.append(f"第{idx}轮用户：{turn.user}")
        lines.append(f"第{idx}轮助手：{turn.assistant}")
    return "\n".join(lines)


def llm_router(
    client: XinferenceChatClient,
    query: str,
    history: Sequence[ConversationTurn],
    *,
    history_rounds: int,
) -> RouterDecision:
    prompt = (
        f"最近对话历史：\n{format_history(history, history_rounds)}\n\n"
        f"当前用户问题：{query}"
    )
    try:
        payload = client.chat_json(
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=200,
        )
        decision = str(payload.get("decision", "RETRIEVE")).upper()
        if decision not in {"RETRIEVE", "DIRECT"}:
            decision = "RETRIEVE"
        reason = str(payload.get("reason", "")).strip() or "LLM路由未提供原因，默认按结果执行"
        return RouterDecision(decision=decision, reason=reason, source="llm")
    except Exception as exc:
        return RouterDecision(
            decision="RETRIEVE",
            reason=f"LLM路由失败，按保守策略默认检索: {exc}",
            source="fallback",
        )


def plan_rewrite(
    client: XinferenceChatClient,
    query: str,
    history: Sequence[ConversationTurn],
    *,
    history_rounds: int,
) -> RewriteDecision:
    prompt = (
        f"最近对话历史：\n{format_history(history, history_rounds)}\n\n"
        f"当前用户问题：{query}"
    )
    try:
        payload = client.chat_json(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=400,
        )
        query_type = str(payload.get("query_type", "fact")).strip().lower()
        if query_type not in {"fact", "compare", "summary"}:
            query_type = "fact"
        rewritten_query = str(payload.get("rewritten_query", "")).strip() or query
        return RewriteDecision(
            need_rewrite=True,
            reason=str(payload.get("reason", "")).strip() or "LLM未提供原因",
            query_type=query_type,
            rewritten_query=rewritten_query,
            source="llm",
        )
    except Exception as exc:
        return RewriteDecision(
            need_rewrite=True,
            reason=f"rewrite 规划失败，保留原问题: {exc}",
            query_type="fact",
            rewritten_query=query,
            source="fallback",
        )


def build_evidence_items(hits: Sequence[RetrievalHit], max_chars: int) -> List[EvidenceItem]:
    evidences: List[EvidenceItem] = []
    for idx, hit in enumerate(hits, start=1):
        metadata = hit.metadata or {}
        evidences.append(
            EvidenceItem(
                evidence_id=f"E{idx}",
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                source_pdf=hit.source_pdf,
                page_start=metadata.get("page_start"),
                page_end=metadata.get("page_end"),
                reranker_score=float(hit.reranker_score),
                fused_score=float(hit.fused_score),
                chunk_type=hit.chunk_type,
                text=(hit.text or "")[:max_chars].strip(),
            )
        )
    return evidences


def should_refuse_by_threshold(hits: Sequence[RetrievalHit], threshold: float) -> bool:
    if not hits:
        return True
    return all(float(hit.reranker_score) < threshold for hit in hits)


def resolve_fusion_mode(route: str) -> str:
    if route == "hybrid_weightsum":
        return "weighted_sum"
    if route in {"hybrid", "hybrid_rrf"}:
        return "rrf"
    return "weighted_sum"


def format_evidence_prompt(query: str, evidences: Sequence[EvidenceItem]) -> str:
    sections = [f"用户问题：{query}", "", "可用证据如下："]
    for item in evidences:
        page_text = page_range_text(item.page_start, item.page_end)
        sections.extend(
            [
                f"[{item.evidence_id}]",
                f"doc_id: {item.doc_id}",
                f"chunk_id: {item.chunk_id}",
                f"page: {page_text}",
                f"reranker_score: {item.reranker_score:.4f}",
                f"text: {item.text}",
                "",
            ]
        )
    return "\n".join(sections).strip()


def generate_rag_answer(
    client: XinferenceChatClient,
    *,
    user_query: str,
    history: Sequence[ConversationTurn],
    history_rounds: int,
    evidences: Sequence[EvidenceItem],
) -> Dict[str, Any]:
    prompt = [
        f"最近对话历史：\n{format_history(history, history_rounds)}",
        "",
        f"用户原始问题：{user_query}",
        "",
        format_evidence_prompt(user_query, evidences),
    ]
    return client.chat_json(
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt="\n".join(prompt).strip(),
        temperature=0.0,
        max_tokens=1400,
    )


def generate_direct_answer(
    client: XinferenceChatClient,
    query: str,
    history: Sequence[ConversationTurn],
    *,
    history_rounds: int,
) -> str:
    return client.chat_text(
        system_prompt=DIRECT_SYSTEM_PROMPT,
        user_prompt=(
            f"最近对话历史：\n{format_history(history, history_rounds)}\n\n"
            f"当前用户问题：{query}"
        ),
        temperature=0.2,
        max_tokens=700,
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_numbers(text: str) -> List[str]:
    return NUMBER_PATTERN.findall(text or "")


def tokenize_for_validation(text: str) -> List[str]:
    tokens = ThreeWayRetriever.tokenize(text)
    return [token for token in tokens if token and token not in STOPWORDS and len(token) > 1]


def claim_supported(claim: str, cited_texts: Sequence[str]) -> bool:
    combined_text = "\n".join(cited_texts)
    combined_norm = normalize_text(combined_text)
    claim_norm = normalize_text(claim)
    if not combined_norm or not claim_norm:
        return False

    claim_numbers = extract_numbers(claim)
    if claim_numbers and not all(number in combined_text for number in claim_numbers):
        return False

    claim_tokens = set(tokenize_for_validation(claim))
    evidence_tokens = set(tokenize_for_validation(combined_text))
    if not claim_tokens:
        return claim_norm in combined_norm

    overlap = claim_tokens & evidence_tokens
    overlap_ratio = len(overlap) / max(1, len(claim_tokens))
    if overlap_ratio >= 0.4:
        return True

    if len(overlap) >= 3:
        return True

    key_phrases = [phrase for phrase in re.split(r"[，。；、,:：\s]+", claim) if len(phrase.strip()) >= 4]
    phrase_matches = sum(1 for phrase in key_phrases if normalize_text(phrase) in combined_norm)
    return phrase_matches >= 1 and len(overlap) >= 2


def build_model_answer(model_output: Dict[str, Any], evidence_map: Dict[str, EvidenceItem]) -> Dict[str, Any]:
    claims = model_output.get("claims") or []
    normalized_claims: List[Dict[str, Any]] = []
    used_evidence_ids = set()

    for raw_claim in claims:
        claim_text = str(raw_claim.get("claim", "")).strip()
        citations = [str(item).strip() for item in raw_claim.get("citations", []) if str(item).strip()]
        existing_citations = [cid for cid in citations if cid in evidence_map]
        if not claim_text:
            continue
        normalized_claims.append({"claim": claim_text, "citations": existing_citations})
        used_evidence_ids.update(existing_citations)

    final_answer = str(model_output.get("answer", "")).strip() or "我不确定"
    return {
        "answer": final_answer,
        "claims": normalized_claims,
        "invalid_claims": [],
        "used_evidence": sorted(used_evidence_ids),
        "confidence": model_output.get("confidence", "low"),
        "should_refuse": bool(model_output.get("should_refuse")),
    }


def first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[。！？；;])", cleaned, maxsplit=1)
    sentence = parts[0].strip()
    return sentence or cleaned[:120]


def build_retrieval_summary_answer(
    *,
    query: str,
    evidences: Sequence[EvidenceItem],
    fallback_reason: str,
) -> Dict[str, Any]:
    if not evidences:
        return {
            "answer": "我不确定",
            "claims": [],
            "invalid_claims": [],
            "used_evidence": [],
            "confidence": "low",
            "should_refuse": True,
            "fallback_reason": fallback_reason,
        }

    summary_claims: List[Dict[str, Any]] = []
    used_evidence: List[str] = []
    for item in evidences[: min(3, len(evidences))]:
        sentence = first_sentence(item.text)
        if not sentence:
            continue
        summary_claims.append({"claim": sentence, "citations": [item.evidence_id]})
        used_evidence.append(item.evidence_id)

    if not summary_claims:
        return {
            "answer": "我不确定",
            "claims": [],
            "invalid_claims": [],
            "used_evidence": [],
            "confidence": "low",
            "should_refuse": True,
            "fallback_reason": fallback_reason,
        }

    answer = "基于检索结果，" + "；".join(claim["claim"] for claim in summary_claims)
    return {
        "answer": answer,
        "claims": summary_claims,
        "invalid_claims": [],
        "used_evidence": used_evidence,
        "confidence": "low",
        "should_refuse": False,
        "fallback_reason": fallback_reason,
    }


def page_range_text(page_start: Optional[int], page_end: Optional[int]) -> str:
    if page_start and page_end and page_start != page_end:
        return f"{page_start}-{page_end}"
    if page_start:
        return str(page_start)
    return "unknown"


def hit_preview_payload(hit: RetrievalHit, preview_chars: int) -> Dict[str, Any]:
    metadata = hit.metadata or {}
    return {
        "doc_id": hit.doc_id,
        "chunk_id": hit.chunk_id,
        "chunk_type": hit.chunk_type,
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "fused_score": float(hit.fused_score),
        "reranker_score": float(hit.reranker_score),
        "source_pdf": hit.source_pdf,
        "text_preview": (hit.text or "")[:preview_chars],
    }


def summarize_retrieval_hits(hits: Sequence[RetrievalHit]) -> Dict[str, Any]:
    top_hits = []
    for hit in hits[:3]:
        metadata = hit.metadata or {}
        top_hits.append(
            {
                "doc_id": hit.doc_id,
                "chunk_id": hit.chunk_id,
                "chunk_type": hit.chunk_type,
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "fused_score": float(hit.fused_score),
                "reranker_score": float(hit.reranker_score),
                "source_pdf": hit.source_pdf,
            }
        )
    return {"hit_count": len(hits), "top_hits": top_hits}


def build_output_payload(
    *,
    args: argparse.Namespace,
    router: RouterDecision,
    rewrite: Optional[RewriteDecision],
    effective_query: str,
    answer_mode: str,
    retrieval_hits: Sequence[RetrievalHit],
    final_answer: str,
    answer_confidence: str,
    should_refuse: bool,
    answer_payload: Optional[Dict[str, Any]] = None,
    evidences: Optional[Sequence[Dict[str, Any]]] = None,
    timings: Sequence[TimingRecord],
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": args.query,
        "router": asdict(router),
        "rewrite": asdict(rewrite) if rewrite is not None else None,
        "effective_query": effective_query,
        "answer_mode": answer_mode,
        "retrieval_used": router.decision == "RETRIEVE",
        "params": {
            "route": args.route,
            "first_stage_top_k": args.first_stage_top_k,
            "rerank_top_k": args.rerank_top_k,
            "rerank_threshold": args.rerank_threshold,
            "vector_candidate_k": args.vector_candidate_k,
            "bm25_candidate_k": args.bm25_candidate_k,
            "query_decompose": bool(args.query_decompose),
        },
        "retrieval_summary": summarize_retrieval_hits(retrieval_hits),
        "answer": {
            "text": final_answer,
            "confidence": answer_confidence,
            "should_refuse": should_refuse,
            "claims": list((answer_payload or {}).get("claims", [])),
            "used_evidence": list((answer_payload or {}).get("used_evidence", [])),
            "invalid_claims": list((answer_payload or {}).get("invalid_claims", [])),
        },
        "evidences": list(evidences or []),
        "timings": [asdict(item) for item in timings],
    }


def save_payload(args: argparse.Namespace, payload: Dict[str, Any]) -> Path:
    args.output_json_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", args.query).strip("_")[:80] or "query"
    path = args.output_json_dir / f"qa__{timestamp}__{slug}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def print_retrieval_hits(hits: Sequence[RetrievalHit], preview_chars: int) -> None:
    print("=== RETRIEVAL CHUNKS ===")
    if not hits:
        print("(no hits)")
        return
    for idx, hit in enumerate(hits, start=1):
        metadata = hit.metadata or {}
        pages = page_range_text(metadata.get("page_start"), metadata.get("page_end"))
        print(
            f"[{idx}] rerank={hit.reranker_score:.4f} fused={hit.fused_score:.4f} "
            f"doc={hit.doc_id} pages={pages}"
        )
        print(f"text={(hit.text or '')[:preview_chars].strip()}")
        print()


def print_final_answer(answer: str, confidence: str, should_refuse: bool) -> None:
    print("=== FINAL ANSWER ===")
    print(answer or "我不确定")
    print()
    print(f"confidence={confidence} should_refuse={should_refuse}")


def print_timings(records: Sequence[TimingRecord]) -> None:
    if not records:
        return
    print()
    print("=== TIMINGS ===")
    total = 0.0
    for item in records:
        total += item.seconds
        print(f"{item.stage}: {item.seconds:.3f}s")
    print(f"total: {total:.3f}s")


class QAApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.startup_timings: List[TimingRecord] = []
        stage_start = perf_counter()
        self.client = XinferenceChatClient(endpoint=args.endpoint, model_uid=args.model, api_key=args.api_key)
        add_timing(self.startup_timings, "client_init", stage_start)
        self.retriever: Optional[ThreeWayRetriever] = None
        stage_start = perf_counter()
        retriever = self._get_retriever()
        add_timing(self.startup_timings, "retriever_init", stage_start)
        stage_start = perf_counter()
        retriever.warm_up()
        add_timing(self.startup_timings, "retriever_warm_up", stage_start)
        self.history: List[ConversationTurn] = []

    def _get_retriever(self) -> ThreeWayRetriever:
        if self.retriever is None:
            self.retriever = ThreeWayRetriever(
                self.args.index_dir,
                reranker_model_path=self.args.rerank_model_path,
                reranker_use_fp16=self.args.reranker_use_fp16,
            )
        return self.retriever

    def run_query(self, query: str) -> QueryRunResult:
        total_start = perf_counter()
        timings: List[TimingRecord] = []

        stage_start = perf_counter()
        router = heuristic_router(query)
        add_timing(timings, "heuristic_router", stage_start)
        if router is None:
            stage_start = perf_counter()
            router = llm_router(
                self.client,
                query,
                self.history,
                history_rounds=self.args.history_rounds,
            )
            add_timing(timings, "llm_router", stage_start)

        rewrite: Optional[RewriteDecision] = None
        effective_query = query
        retrieval_hits: List[RetrievalHit] = []
        evidences: List[EvidenceItem] = []
        raw_generation: Optional[Dict[str, Any]] = None
        checked_answer: Optional[Dict[str, Any]] = None
        direct_answer: Optional[str] = None
        answer_mode = "llm"

        if router.decision == "DIRECT":
            direct_stage_start = perf_counter()
            try:
                direct_answer = generate_direct_answer(
                    self.client,
                    query,
                    self.history,
                    history_rounds=self.args.history_rounds,
                )
            except RuntimeError:
                answer_mode = "retrieval_fallback"
                stage_start = perf_counter()
                retriever = self._get_retriever()
                add_timing(timings, "retriever_init", stage_start)
                stage_start = perf_counter()
                retrieval_hits = retriever.retrieve_two_stage(
                    query=query,
                    route=self.args.route,
                    first_stage_top_k=self.args.first_stage_top_k,
                    rerank_top_k=self.args.rerank_top_k,
                    vector_candidate_k=self.args.vector_candidate_k,
                    bm25_candidate_k=self.args.bm25_candidate_k,
                    vector_weight=self.args.vector_weight,
                    bm25_weight=self.args.bm25_weight,
                    title_weight=self.args.title_weight,
                    fusion_mode=resolve_fusion_mode(self.args.route),
                    rrf_k=self.args.rrf_k,
                    query_decompose=self.args.query_decompose,
                )
                add_timing(timings, "retrieve_two_stage", stage_start)
                stage_start = perf_counter()
                evidences = build_evidence_items(retrieval_hits, max_chars=self.args.max_evidence_chars)
                add_timing(timings, "build_evidence_items", stage_start)
                checked_answer = build_retrieval_summary_answer(
                    query=query,
                    evidences=evidences,
                    fallback_reason="direct LLM unavailable",
                )
                direct_answer = checked_answer["answer"]
            add_timing(timings, "direct_generation", direct_stage_start)
        else:
            stage_start = perf_counter()
            rewrite = plan_rewrite(
                self.client,
                query,
                self.history,
                history_rounds=self.args.history_rounds,
            )
            effective_query = rewrite.rewritten_query
            add_timing(timings, "rewrite_planning", stage_start)

            stage_start = perf_counter()
            retriever = self._get_retriever()
            add_timing(timings, "retriever_init", stage_start)

            stage_start = perf_counter()
            retrieval_hits = retriever.retrieve_two_stage(
                query=effective_query,
                route=self.args.route,
                first_stage_top_k=self.args.first_stage_top_k,
                rerank_top_k=self.args.rerank_top_k,
                vector_candidate_k=self.args.vector_candidate_k,
                bm25_candidate_k=self.args.bm25_candidate_k,
                vector_weight=self.args.vector_weight,
                bm25_weight=self.args.bm25_weight,
                title_weight=self.args.title_weight,
                fusion_mode=resolve_fusion_mode(self.args.route),
                rrf_k=self.args.rrf_k,
                query_decompose=self.args.query_decompose,
            )
            add_timing(timings, "retrieve_two_stage", stage_start)

            stage_start = perf_counter()
            evidences = build_evidence_items(retrieval_hits, max_chars=self.args.max_evidence_chars)
            add_timing(timings, "build_evidence_items", stage_start)
            evidence_map = {item.evidence_id: item for item in evidences}

            stage_start = perf_counter()
            if should_refuse_by_threshold(retrieval_hits, self.args.rerank_threshold):
                checked_answer = {
                    "answer": "我不确定",
                    "claims": [],
                    "invalid_claims": [],
                    "used_evidence": [],
                    "confidence": "low",
                    "should_refuse": True,
                }
                add_timing(timings, "refusal_gate", stage_start)
            else:
                add_timing(timings, "refusal_gate", stage_start)
                stage_start = perf_counter()
                try:
                    raw_generation = generate_rag_answer(
                        self.client,
                        user_query=query,
                        history=self.history,
                        history_rounds=self.args.history_rounds,
                        evidences=evidences,
                    )
                except RuntimeError:
                    answer_mode = "retrieval_fallback"
                    checked_answer = build_retrieval_summary_answer(
                        query=effective_query,
                        evidences=evidences,
                        fallback_reason="rag LLM unavailable",
                    )
                    raw_generation = None
                add_timing(timings, "rag_generation", stage_start)
                if checked_answer is None:
                    stage_start = perf_counter()
                    checked_answer = build_model_answer(raw_generation, evidence_map)
                    add_timing(timings, "post_check", stage_start)
                else:
                    stage_start = perf_counter()
                    add_timing(timings, "post_check", stage_start)

        assistant_text = direct_answer
        if assistant_text is None and checked_answer is not None:
            assistant_text = checked_answer.get("answer", "")
        answer_confidence = "medium" if direct_answer else str((checked_answer or {}).get("confidence", "low"))
        should_refuse = False if direct_answer else bool((checked_answer or {}).get("should_refuse"))
        terminal_retrieval_hits = list(retrieval_hits)
        answer_payload = (
            {
                "answer": direct_answer,
                "claims": [],
                "invalid_claims": [],
                "used_evidence": [],
                "confidence": answer_confidence,
                "should_refuse": should_refuse,
            }
            if direct_answer is not None
            else dict(checked_answer or {})
        )
        evidence_payload = [asdict(item) for item in evidences]

        original_query = self.args.query
        self.args.query = query
        stage_start = perf_counter()
        payload = build_output_payload(
            args=self.args,
            router=router,
            rewrite=rewrite,
            effective_query=effective_query,
            answer_mode=answer_mode,
            retrieval_hits=retrieval_hits,
            final_answer=assistant_text or "",
            answer_confidence=answer_confidence,
            should_refuse=should_refuse,
            answer_payload=answer_payload,
            evidences=evidence_payload,
            timings=timings,
        )
        add_timing(timings, "build_output_payload", stage_start)
        stage_start = perf_counter()
        saved_path = save_payload(self.args, payload)
        add_timing(timings, "save_payload", stage_start)
        timings.append(TimingRecord(stage="wall_clock_total", seconds=perf_counter() - total_start))
        payload["timings"] = [asdict(item) for item in timings]
        self.args.query = original_query

        self.history.append(
            ConversationTurn(
                user=query,
                assistant=assistant_text or "",
            )
        )
        if self.args.history_rounds > 0:
            self.history = self.history[-self.args.history_rounds:]

        # Release one-shot retrieval / grounding artifacts immediately after
        # producing the final answer. Only user-assistant history is retained.
        retrieval_hits = []
        evidences = []
        raw_generation = None
        checked_answer = None
        gc.collect()

        return QueryRunResult(
            payload=payload,
            saved_path=saved_path,
            router=router,
            rewrite=rewrite,
            effective_query=effective_query,
            answer_mode=answer_mode,
            retrieval_hits=terminal_retrieval_hits,
            final_answer=assistant_text or "",
            answer_confidence=answer_confidence,
            should_refuse=should_refuse,
            answer_payload=answer_payload,
            evidences=evidence_payload,
            timings=timings,
        )


def print_run_result(args: argparse.Namespace, result: QueryRunResult) -> None:
    if args.json:
        print(json.dumps({"saved_to": str(result.saved_path), "payload": result.payload}, ensure_ascii=False, indent=2))
        return

    print(f"Router: {result.router.decision} ({result.router.source}) - {result.router.reason}")
    if result.rewrite is not None:
        print(
            f"Rewrite: need={result.rewrite.need_rewrite} ({result.rewrite.source}) - "
            f"{result.rewrite.reason}"
        )
        print(f"effective_query={result.effective_query}")
    print(f"answer_mode={result.answer_mode}")
    print(f"saved_json={result.saved_path}")
    print()
    if result.router.decision == "RETRIEVE" and not args.hide_retrieval_chunks:
        print_retrieval_hits(result.retrieval_hits, preview_chars=args.show_text_chars)
        print()
    print_final_answer(
        result.final_answer,
        result.answer_confidence,
        result.should_refuse,
    )
    if args.show_timings:
        print_timings(result.timings)


def run_interactive(args: argparse.Namespace) -> None:
    app = QAApp(args)
    print("Interactive mode ready. Enter a query, or type /exit to quit.")
    if args.show_timings:
        print("=== STARTUP TIMINGS ===")
        for item in app.startup_timings:
            print(f"{item.stage}: {item.seconds:.3f}s")
        print()
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not query:
            continue
        if query in {"/exit", "exit", "quit"}:
            break

        try:
            result = app.run_query(query)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            continue
        print_run_result(args, result)
        print()


def main() -> None:
    args = parse_args()
    if args.interactive:
        run_interactive(args)
        return

    app = QAApp(args)
    if args.show_timings:
        print("=== STARTUP TIMINGS ===")
        for item in app.startup_timings:
            print(f"{item.stage}: {item.seconds:.3f}s")
        print()
    result = app.run_query(args.query)
    print_run_result(args, result)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

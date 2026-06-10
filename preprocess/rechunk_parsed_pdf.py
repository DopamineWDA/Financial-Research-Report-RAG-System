"""Re-chunk parsed PDF JSON artifacts for RAG retrieval.

Consumes ``*.parsed.json`` files produced by the DeepDoc parser and emits a
cleaned, metadata-rich chunk set with:

- cover summary chunks
- section-aware text chunks
- raw table chunks + table summary chunks
- optional figure summary chunks

The script intentionally does not modify the upstream parser. It operates on
the parsed JSON artifact as a separate post-processing step.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from common.paths import CHUNKED_ROOT


VALID_CHUNK_SIZES = {256, 512, 1024}
VALID_OVERLAPS = {50, 100, 200}
SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[\.、\s]|$)")
CN_SECTION_RE = re.compile(r"^[一二三四五六七八九十]+[、.]")
RATING_RE = re.compile(r"(优于大市|强于大市|买入|增持|中性|持有|减持|卖出|弱于大市)")
ANALYST_RE = re.compile(r"(证券分析师|分析师|S\d{15,}|@[\w.-]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
ANALYST_CONTACT_RE = re.compile(
    r"(证券分析师|分析师|S\d{15,}|执业编号|邮箱|电话|传真|联系电话|微信|@[\w.-]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
DISCLAIMER_RE = re.compile(r"(免责声明|投资评级说明|评级说明|法律声明|特别声明|信息披露)")
RELATED_REPORT_RE = re.compile(r"(相关研究报告|往期研究|历史报告)")
SOURCE_ONLY_RE = re.compile(r"^资料来源[:：]")
RISK_RE = re.compile(r"(风险提示|风险因素|风险分析)")
SKIP_SECTION_RE = re.compile(r"(分析师介绍|要求披露|公司简介|研究所|中山证券研究所|联系人|证券分析师|证券研究报告)")
TERMINAL_SECTION_TITLES = {
    "免责声明",
    "特别声明",
    "一般声明",
    "重要声明",
    "法律声明",
    "分析师声明",
    "证券分析师声明",
    "信息披露声明",
    "投资评级说明",
    "投资评级的说明",
    "评级说明",
    "风险提示及免责声明",
}
FIGURE_MEANINGFUL_RE = re.compile(r"(图\d+|走势|趋势|同比|环比|增长|下降|占比|变化|月度|累计|价格指数|发电量|用电量)")
NOISY_FIGURE_LINE_RE = re.compile(r"^[\d\s\.\-%/%+(),:：A-Za-z]{1,30}$")


@dataclass
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class CleanBlock:
    id: str
    type: str
    page: int
    bbox: list[float]
    text: str
    meta: dict[str, Any]
    section: str = ""
    subsection: str = ""


@dataclass
class LogicalTable:
    table_id: str
    page_start: int
    page_end: int
    section: str
    subsection: str
    block_ids: list[str]
    caption: str
    headers: list[list[str]]
    rows: list[list[str]]
    html_blocks: list[str]


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.backend = "heuristic"
        self._enc = None
        try:
            import tiktoken  # type: ignore

            self._enc = tiktoken.get_encoding(encoding_name)
            self.backend = f"tiktoken:{encoding_name}"
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        text = text or ""
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        return len(self._fallback_tokens(text))

    def slice_tail(self, text: str, token_count: int) -> str:
        if token_count <= 0 or not text:
            return ""
        if self._enc is not None:
            tokens = self._enc.encode(text)
            tail = tokens[-token_count:]
            return self._enc.decode(tail).strip()
        tokens = self._fallback_tokens(text)
        tail = tokens[-token_count:]
        return "".join(tail).strip()

    def split_text(self, text: str, chunk_size: int) -> list[str]:
        text = clean_text(text)
        if not text:
            return []
        if self.count(text) <= chunk_size:
            return [text]

        sentences = split_sentences(text)
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if self.count(sentence) > chunk_size:
                if current:
                    pieces.append(current.strip())
                    current = ""
                pieces.extend(self._hard_split(sentence, chunk_size))
                continue
            candidate = sentence if not current else current + sentence
            if current and self.count(candidate) > chunk_size:
                pieces.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current.strip():
            pieces.append(current.strip())
        return pieces or self._hard_split(text, chunk_size)

    def _hard_split(self, text: str, chunk_size: int) -> list[str]:
        if self._enc is not None:
            tokens = self._enc.encode(text)
            out = []
            for idx in range(0, len(tokens), chunk_size):
                piece = self._enc.decode(tokens[idx : idx + chunk_size]).strip()
                if piece:
                    out.append(piece)
            return out

        tokens = self._fallback_tokens(text)
        out = []
        for idx in range(0, len(tokens), chunk_size):
            piece = "".join(tokens[idx : idx + chunk_size]).strip()
            if piece:
                out.append(piece)
        return out

    @staticmethod
    def _fallback_tokens(text: str) -> list[str]:
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?|[^\s]", text)


class ParsedPdfChunker:
    def __init__(self, chunk_size: int, overlap: int, *, encoding_name: str = "cl100k_base") -> None:
        if chunk_size not in VALID_CHUNK_SIZES:
            raise ValueError(f"unsupported chunk size: {chunk_size}")
        if overlap not in VALID_OVERLAPS:
            raise ValueError(f"unsupported overlap: {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokens = TokenCounter(encoding_name=encoding_name)

    def process_file(self, parsed_file: Path, output_path: Path | None = None) -> dict[str, Any]:
        data = json.loads(parsed_file.read_text(encoding="utf-8"))
        doc_id = self._doc_id(data, parsed_file)
        source = str(data.get("source") or parsed_file)
        page_sizes = self._page_sizes(data.get("page_sizes") or [])
        raw_blocks = [self._to_clean_block(block) for block in data.get("blocks", [])]

        cover_block_ids = self._detect_cover_block_ids(raw_blocks, page_sizes)
        toc_pages = self._detect_toc_pages(raw_blocks)
        cleaned_blocks, removed = self._clean_blocks(raw_blocks, page_sizes, cover_block_ids, toc_pages)
        blocks_for_body = [block for block in cleaned_blocks if block.id not in cover_block_ids]

        sections = self._assign_sections(blocks_for_body)
        chunks: list[ChunkRecord] = []
        chunks.extend(self._build_cover_chunk(doc_id, source, cleaned_blocks, raw_blocks, cover_block_ids))
        chunks.extend(self._build_text_chunks(doc_id, source, sections))
        chunks.extend(self._build_table_chunks(doc_id, source, sections, cleaned_blocks))

        result = {
            "doc_id": doc_id,
            "source": source,
            "chunk_config": {
                "chunk_size_tokens": self.chunk_size,
                "overlap_tokens": self.overlap,
                "token_backend": self.tokens.backend,
            },
            "toc_pages": sorted(toc_pages),
            "summary": {
                "input_blocks": len(raw_blocks),
                "cleaned_blocks": len(cleaned_blocks),
                "removed_blocks": len(removed),
                "chunks": len(chunks),
            },
            "removed_blocks": removed,
            "chunks": [asdict(chunk) for chunk in chunks],
        }

        if output_path is None:
            output_path = parsed_file.with_suffix("").with_suffix(f".chunks.cs{self.chunk_size}.ov{self.overlap}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _to_clean_block(self, block: dict[str, Any]) -> CleanBlock:
        return CleanBlock(
            id=str(block.get("id", "")),
            type=str(block.get("type", "text")),
            page=int(block.get("page", 0) or 0),
            bbox=[float(v) for v in (block.get("bbox") or [0, 0, 0, 0])],
            text=clean_text(str(block.get("text", ""))),
            meta=dict(block.get("meta") or {}),
        )

    def _page_sizes(self, page_sizes: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for item in page_sizes:
            page = int(item.get("page", 0) or 0)
            out[page] = (float(item.get("width", 1) or 1), float(item.get("height", 1) or 1))
        return out

    def _doc_id(self, data: dict[str, Any], parsed_file: Path) -> str:
        source = str(data.get("source") or "")
        if source:
            name = Path(source).name
            if name.lower().endswith(".pdf"):
                return name[:-4]
            return name
        return parsed_file.name.replace(".parsed.json", "")

    def _detect_cover_block_ids(self, blocks: list[CleanBlock], page_sizes: dict[int, tuple[float, float]]) -> set[str]:
        page1 = [block for block in blocks if block.page == 1]
        if not page1:
            return set()
        page_width, page_height = page_sizes.get(1, (595.0, 842.0))
        top_cutoff = page_height * 0.20
        right_col_cutoff = page_width * 0.60
        cover_ids: set[str] = set()
        for block in page1:
            x0 = float(block.bbox[0]) if len(block.bbox) >= 1 else 0.0
            top = float(block.bbox[1]) if len(block.bbox) >= 2 else 0.0
            bottom = float(block.bbox[3]) if len(block.bbox) >= 4 else top
            text = block.text
            is_header_like = block.type == "title" or bool(RATING_RE.search(text)) or bool(ANALYST_RE.search(text))
            is_top_banner = top <= top_cutoff and (block.type == "title" or len(text) <= 120)
            is_right_sidebar = x0 >= right_col_cutoff and bottom <= page_height * 0.38 and is_header_like
            if is_top_banner or is_right_sidebar:
                cover_ids.add(block.id)
        return cover_ids

    def _clean_blocks(
        self,
        blocks: list[CleanBlock],
        page_sizes: dict[int, tuple[float, float]],
        cover_block_ids: set[str],
        toc_pages: set[int],
    ) -> tuple[list[CleanBlock], list[dict[str, Any]]]:
        repeated_edges = self._detect_repeated_edge_texts(blocks, page_sizes)
        cleaned: list[CleanBlock] = []
        removed: list[dict[str, Any]] = []
        skip_mode: str | None = None
        skip_anchor: CleanBlock | None = None
        skip_until_page_end = False

        for block in blocks:
            text = block.text
            reason = ""
            if not text:
                reason = "empty"
            elif block.page in toc_pages:
                reason = "table_of_contents_page"
            elif block.id in repeated_edges:
                reason = "header_or_footer"
            elif block.type == "text" and is_analyst_contact_block(text):
                reason = "analyst_info"
            elif block.type == "text" and is_noise_text_block(text):
                reason = "text_noise"
            elif block.type == "title" and RELATED_REPORT_RE.search(text):
                skip_mode = "related_report"
                skip_anchor = block
                reason = "related_report_title"
            elif block.type == "title" and is_terminal_section_title(text):
                skip_mode = "terminal_section"
                skip_anchor = block
                skip_until_page_end = True
                reason = "terminal_section_title"
            elif block.type == "title" and is_noisy_title(text):
                reason = "title_noise"
            elif skip_mode == "related_report" and skip_anchor and self._is_same_column_following_block(block, skip_anchor, page_sizes) and block.type in {"text", "title"} and not RISK_RE.search(text):
                reason = "related_report_block"
            elif skip_mode == "terminal_section" and skip_anchor and (
                (skip_until_page_end and block.page >= skip_anchor.page) or self._is_same_column_following_block(block, skip_anchor, page_sizes)
            ) and block.type in {"text", "title", "table", "figure"}:
                reason = "terminal_section_block"
            elif SOURCE_ONLY_RE.match(text) and is_source_only_text(text):
                reason = "source_only"
            elif block.type == "text" and is_toc_block(text):
                reason = "table_of_contents_noise"
            elif block.type == "figure" and not self._figure_is_meaningful(block):
                reason = "low_quality_figure_ocr"

            if block.type == "title" and skip_mode != "terminal_section" and not RELATED_REPORT_RE.search(text) and not is_terminal_section_title(text) and not SKIP_SECTION_RE.search(text):
                skip_mode = None
                skip_anchor = None
                skip_until_page_end = False
            if block.type == "text" and RISK_RE.search(text) and block.page not in toc_pages and reason not in {"table_of_contents_page", "table_of_contents_noise"}:
                reason = ""
                skip_mode = None
                skip_anchor = None
                skip_until_page_end = False
            if block.id in cover_block_ids:
                if is_analyst_contact_block(text):
                    reason = "analyst_info"

            if reason:
                removed.append({"id": block.id, "page": block.page, "type": block.type, "reason": reason, "text": text[:180]})
                continue
            cleaned.append(block)

        return cleaned, removed

    def _detect_toc_pages(self, blocks: list[CleanBlock]) -> set[int]:
        by_page: dict[int, list[CleanBlock]] = {}
        for block in blocks:
            by_page.setdefault(block.page, []).append(block)

        toc_pages: set[int] = set()
        for page, page_blocks in by_page.items():
            if any(block.type == "title" and ("内容目录" in block.text or "图表目录" in block.text or block.text.strip() == "目录") for block in page_blocks):
                toc_pages.add(page)

        if toc_pages:
            frontier = max(toc_pages)
            for page in sorted(by_page):
                if page <= frontier:
                    continue
                text_blocks = [block for block in by_page[page] if block.type == "text"]
                title_blocks = [block for block in by_page[page] if block.type == "title"]
                if title_blocks:
                    break
                if not text_blocks:
                    continue
                if all(is_toc_block(block.text) for block in text_blocks):
                    toc_pages.add(page)
                    frontier = page
                    continue
                break
        return toc_pages

    def _is_same_column_following_block(
        self,
        block: CleanBlock,
        anchor: CleanBlock,
        page_sizes: dict[int, tuple[float, float]],
    ) -> bool:
        if block.page != anchor.page:
            return False
        if block.id == anchor.id:
            return False
        block_top = float(block.bbox[1]) if len(block.bbox) >= 2 else 0.0
        anchor_bottom = float(anchor.bbox[3]) if len(anchor.bbox) >= 4 else 0.0
        if block_top < anchor_bottom:
            return False
        page_width, _ = page_sizes.get(block.page, (595.0, 842.0))
        block_x0 = float(block.bbox[0]) if len(block.bbox) >= 1 else 0.0
        anchor_x0 = float(anchor.bbox[0]) if len(anchor.bbox) >= 1 else 0.0
        same_right_col = anchor_x0 >= page_width * 0.55 and block_x0 >= page_width * 0.55
        horizontal_overlap = bbox_horizontal_overlap_ratio(block.bbox, anchor.bbox) >= 0.20
        return same_right_col or horizontal_overlap

    def _detect_repeated_edge_texts(self, blocks: list[CleanBlock], page_sizes: dict[int, tuple[float, float]]) -> set[str]:
        seen: dict[str, list[str]] = {}
        for block in blocks:
            text = normalize_edge_text(block.text)
            if not text or len(text) > 80:
                continue
            _, page_height = page_sizes.get(block.page, (1.0, 842.0))
            top = float(block.bbox[1]) if len(block.bbox) >= 2 else 0.0
            bottom = float(block.bbox[3]) if len(block.bbox) >= 4 else 0.0
            if top <= page_height * 0.08 or bottom >= page_height * 0.92:
                seen.setdefault(text, []).append(block.id)
        repeated: set[str] = set()
        for ids in seen.values():
            if len(ids) >= 3:
                repeated.update(ids)
        return repeated

    def _assign_sections(self, blocks: list[CleanBlock]) -> list[CleanBlock]:
        current_section = ""
        current_subsection = ""
        current_subsubsection = ""
        for block in blocks:
            if block.type == "title":
                level = title_level(block.text)
                if level <= 0:
                    if not current_section:
                        current_section = "正文"
                    block.section = current_section
                    block.subsection = " / ".join(part for part in [current_subsection, current_subsubsection] if part)
                    continue
                if level == 1:
                    current_section = block.text
                    current_subsection = ""
                    current_subsubsection = ""
                elif level == 2:
                    if not current_section:
                        current_section = "正文"
                    current_subsection = block.text
                    current_subsubsection = ""
                else:
                    if not current_section:
                        current_section = "正文"
                    current_subsubsection = block.text
                block.section = current_section or "正文"
                block.subsection = " / ".join(part for part in [current_subsection, current_subsubsection] if part)
                continue
            if not current_section:
                current_section = "正文"
            block.section = current_section
            block.subsection = " / ".join(part for part in [current_subsection, current_subsubsection] if part)
        return blocks

    def _build_cover_chunk(
        self,
        doc_id: str,
        source: str,
        cleaned_blocks: list[CleanBlock],
        raw_blocks: list[CleanBlock],
        cover_block_ids: set[str],
    ) -> list[ChunkRecord]:
        filename_bits = source_filename_fields(source)
        cover_blocks = [block for block in cleaned_blocks if block.id in cover_block_ids]
        raw_cover_blocks = [block for block in raw_blocks if block.id in cover_block_ids]
        cover_title = first_text([block.text for block in cover_blocks if block.type == "title"])
        title = filename_bits["title"] or cover_title or filename_bits["company"] or doc_id
        rating = first_match_text([block.text for block in cover_blocks], RATING_RE) or filename_bits["rating"]
        institution = filename_bits["institution"] or first_broker_text(block.text for block in raw_cover_blocks)
        company = filename_bits["company"]
        date = filename_bits["date"]
        core_view = self._cover_core_view(cleaned_blocks or raw_blocks, filename_bits)

        if not cover_blocks and not any([title, institution, company, date, rating, core_view]):
            return []

        lines = [f"标题：{title}"]
        if company:
            lines.append(f"公司：{company}")
        if institution:
            lines.append(f"机构：{institution}")
        if date:
            lines.append(f"日期：{date}")
        if rating:
            lines.append(f"评级：{rating}")
        if core_view:
            lines.append(f"核心观点：{core_view}")

        pages = sorted({block.page for block in cover_blocks}) or [1]
        return [
            ChunkRecord(
                id=make_chunk_id(doc_id, "cover_summary", 1),
                text="\n".join(lines),
                metadata={
                    "doc_id": doc_id,
                    "source": source,
                    "page_start": min(pages),
                    "page_end": max(pages),
                    "section": "封面",
                    "subsection": "",
                    "block_ids": [block.id for block in cover_blocks],
                    "chunk_type": "cover_summary_chunk",
                    "table_id": None,
                    "figure_id": None,
                },
            )
        ]

    def _cover_core_view(self, blocks: list[CleanBlock], filename_bits: dict[str, str]) -> str:
        stop_terms = {
            clean_text(filename_bits.get("title", "")),
            clean_text(filename_bits.get("company", "")),
            clean_text(filename_bits.get("institution", "")),
        }
        stop_terms = {term for term in stop_terms if term}
        for block in blocks:
            if block.page != 1:
                continue
            if block.type not in {"text", "title"}:
                continue
            text = block.text
            if is_analyst_contact_block(text) or "相关研究报告" in text:
                continue
            if "市场回顾" in text or "投资策略" in text or RISK_RE.search(text):
                continue
            if looks_like_noisy_cover_text(text):
                continue
            sentences = split_sentences(text) or [text]
            for sentence in sentences:
                sentence = clean_text(sentence)
                if len(sentence) >= 20 and sentence not in stop_terms and not looks_like_noisy_cover_text(sentence):
                    return sentence
        return ""

    def _build_text_chunks(self, doc_id: str, source: str, blocks: list[CleanBlock]) -> list[ChunkRecord]:
        text_blocks = [block for block in blocks if block.type in {"text", "title"}]
        grouped: dict[tuple[str, str], list[CleanBlock]] = {}
        order: list[tuple[str, str]] = []
        for block in text_blocks:
            key = (block.section or "正文", block.subsection or "")
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(block)

        chunks: list[ChunkRecord] = []
        chunk_index = 1
        for key in order:
            section, subsection = key
            units = self._section_units(grouped[key])
            for payload in self._window_units(units):
                chunks.append(
                    ChunkRecord(
                        id=make_chunk_id(doc_id, "text", chunk_index),
                        text=payload["text"],
                        metadata={
                            "doc_id": doc_id,
                            "source": source,
                            "page_start": payload["page_start"],
                            "page_end": payload["page_end"],
                            "section": section,
                            "subsection": subsection,
                            "block_ids": payload["block_ids"],
                            "chunk_type": "text_chunk",
                            "table_id": None,
                            "figure_id": None,
                        },
                    )
                )
                chunk_index += 1
        return chunks

    def _section_units(self, blocks: list[CleanBlock]) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        pending_title = ""
        for block in blocks:
            if block.type == "title":
                title_text = clean_text(block.text)
                if is_noisy_title(title_text):
                    pending_title = ""
                    continue
                pending_title = title_text
                continue

            block_text = block.text
            if pending_title:
                block_text = f"{pending_title}\n\n{block_text}".strip()
                pending_title = ""

            pieces = self.tokens.split_text(block_text, self.chunk_size)
            if not pieces:
                continue
            for piece_idx, piece in enumerate(pieces):
                units.append(
                    {
                        "text": piece,
                        "tokens": self.tokens.count(piece),
                        "page": block.page,
                        "block_ids": [block.id] if piece_idx == 0 else [f"{block.id}#part{piece_idx + 1}"],
                    }
                )
        return units

    def _window_units(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        current_tokens = 0
        overlap_seed = ""
        overlap_block_ids: list[str] = []
        overlap_page = 0

        def flush() -> None:
            nonlocal current, current_tokens, overlap_seed, overlap_block_ids, overlap_page
            if not current and not overlap_seed:
                return
            parts = [overlap_seed] if overlap_seed else []
            parts.extend(unit["text"] for unit in current)
            text = "\n\n".join(part for part in parts if part).strip()
            block_ids = list(overlap_block_ids)
            block_ids.extend(block_id for unit in current for block_id in unit["block_ids"])
            pages = [overlap_page] if overlap_page else []
            pages.extend(unit["page"] for unit in current if unit["page"])
            if not text or not pages:
                current = []
                current_tokens = 0
                overlap_seed = ""
                overlap_block_ids = []
                overlap_page = 0
                return
            chunks.append(
                {
                    "text": text,
                    "block_ids": dedupe_preserve(block_ids),
                    "page_start": min(pages),
                    "page_end": max(pages),
                }
            )
            overlap_seed = sentence_aligned_tail(text, self.overlap, self.tokens)
            overlap_block_ids = []
            overlap_page = chunks[-1]["page_end"]
            current = []
            current_tokens = self.tokens.count(overlap_seed)

        for unit in units:
            if current and current_tokens + unit["tokens"] > self.chunk_size:
                flush()
            current.append(unit)
            current_tokens += unit["tokens"]
        flush()
        return chunks

    def _build_table_chunks(
        self,
        doc_id: str,
        source: str,
        section_blocks: list[CleanBlock],
        cleaned_blocks: list[CleanBlock],
    ) -> list[ChunkRecord]:
        logical_tables = self._merge_logical_tables([block for block in section_blocks if block.type == "table"])
        chunks: list[ChunkRecord] = []
        raw_idx = 1
        summary_idx = 1

        for logical in logical_tables:
            anchor_block = next((block for block in cleaned_blocks if block.id == logical.block_ids[0]), None)
            if anchor_block is None:
                continue
            context = surrounding_sentences(anchor_block, cleaned_blocks, limit=3)
            summary = summarize_table(logical.caption, logical.headers, logical.rows)
            raw_html = "\n".join(block for block in logical.html_blocks if block).strip()
            if not raw_html:
                continue
            chunks.append(
                ChunkRecord(
                    id=make_chunk_id(doc_id, "raw_table", raw_idx),
                    text=raw_html,
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        "page_start": logical.page_start,
                        "page_end": logical.page_end,
                        "section": logical.section or "正文",
                        "subsection": logical.subsection or "",
                        "block_ids": logical.block_ids,
                        "chunk_type": "raw_table_chunk",
                        "table_id": logical.table_id,
                        "figure_id": None,
                        "row_range": [1, len(logical.rows)],
                        "caption": logical.caption,
                        "bm25_context": context,
                        "text_format": "html",
                    },
                )
            )
            raw_idx += 1

            summary_lines = []
            if logical.caption:
                summary_lines.append(f"表名：{logical.caption}")
            summary_lines.append(f"所属章节：{logical.section or '正文'}")
            if context:
                summary_lines.append(f"前后文：{context}")
            if summary:
                summary_lines.append(f"关键指标：{summary}")
            chunks.append(
                ChunkRecord(
                    id=make_chunk_id(doc_id, "table_summary", summary_idx),
                    text="\n".join(summary_lines),
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        "page_start": logical.page_start,
                        "page_end": logical.page_end,
                        "section": logical.section or "正文",
                        "subsection": logical.subsection or "",
                        "block_ids": logical.block_ids,
                        "chunk_type": "table_summary_chunk",
                        "table_id": logical.table_id,
                        "figure_id": None,
                    },
                )
            )
            summary_idx += 1

        return chunks

    def _merge_logical_tables(self, tables: list[CleanBlock]) -> list[LogicalTable]:
        logical_tables: list[LogicalTable] = []
        current: LogicalTable | None = None

        for block in tables:
            caption, headers, rows = parse_html_table(block.text)
            if current and self._should_merge_tables(current, block, caption, headers, rows):
                if not current.caption and caption:
                    current.caption = caption
                if not current.headers and headers:
                    current.headers = headers
                current.rows.extend(rows)
                current.page_end = block.page
                current.block_ids.append(block.id)
                current.html_blocks.append(block.text)
                continue

            logical_tables.append(
                LogicalTable(
                    table_id=f"table_{len(logical_tables) + 1:04d}",
                    page_start=block.page,
                    page_end=block.page,
                    section=block.section or "正文",
                    subsection=block.subsection or "",
                    block_ids=[block.id],
                    caption=caption,
                    headers=headers,
                    rows=list(rows),
                    html_blocks=[block.text],
                )
            )
            current = logical_tables[-1]

        return logical_tables

    def _should_merge_tables(
        self,
        current: LogicalTable,
        next_block: CleanBlock,
        next_caption: str,
        next_headers: list[list[str]],
        next_rows: list[list[str]],
    ) -> bool:
        if next_block.page != current.page_end + 1:
            return False
        if (next_block.section or "正文") != (current.section or "正文"):
            return False
        if (next_block.subsection or "") != (current.subsection or ""):
            return False

        current_col_count = table_col_count(current.headers, current.rows)
        next_col_count = table_col_count(next_headers, next_rows)
        if current_col_count and next_col_count and current_col_count != next_col_count:
            return False

        caption_missing_or_source_only = (not next_caption) or next_caption.startswith("资料来源")
        same_named_caption = bool(current.caption and next_caption and normalize_table_caption(current.caption) == normalize_table_caption(next_caption))
        if not (caption_missing_or_source_only or same_named_caption):
            return False

        if next_headers:
            return headers_compatible(current.headers, next_headers)
        return True

    def _build_figure_chunks(
        self,
        doc_id: str,
        source: str,
        section_blocks: list[CleanBlock],
        cleaned_blocks: list[CleanBlock],
    ) -> list[ChunkRecord]:
        figures = [block for block in section_blocks if block.type == "figure" and self._figure_is_meaningful(block)]
        chunks: list[ChunkRecord] = []
        idx = 1
        for fig_idx, figure in enumerate(figures, start=1):
            figure_id = f"figure_{fig_idx:04d}"
            title = figure_title(figure.text)
            if not title:
                continue
            context = surrounding_sentences(figure, cleaned_blocks, limit=3)
            trend = figure_trend_hint(figure.text)
            lines = [f"图名：{title}", f"所属章节：{figure.section or '正文'}"]
            if context:
                lines.append(f"图注上下文：{context}")
            if trend:
                lines.append(f"趋势描述：{trend}")
            chunks.append(
                ChunkRecord(
                    id=make_chunk_id(doc_id, "figure_summary", idx),
                    text="\n".join(lines),
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        "page_start": figure.page,
                        "page_end": figure.page,
                        "section": figure.section or "正文",
                        "subsection": figure.subsection or "",
                        "block_ids": [figure.id],
                        "chunk_type": "figure_summary_chunk",
                        "table_id": None,
                        "figure_id": figure_id,
                    },
                )
            )
            idx += 1
        return chunks

    def _figure_is_meaningful(self, block: CleanBlock) -> bool:
        text = clean_text(block.text)
        if not text:
            return False
        if not FIGURE_MEANINGFUL_RE.search(text):
            return False
        lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
        useful = []
        for line in lines:
            if SOURCE_ONLY_RE.match(line):
                continue
            if NOISY_FIGURE_LINE_RE.match(line) and not FIGURE_MEANINGFUL_RE.search(line):
                continue
            useful.append(line)
        merged = " ".join(useful)
        return len(merged) >= 18


def source_filename_fields(source: str) -> dict[str, str]:
    name = Path(source).name
    stem = name[:-4] if name.lower().endswith(".pdf") else name
    stem = stem.replace(".parsed", "")
    parts = stem.split("_")
    date = ""
    institution = ""
    company = ""
    title = ""
    rating = ""
    for part in parts:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            date = part
            break
    if date and date in parts:
        idx = parts.index(date)
        tail = parts[idx + 1 :]
        if tail:
            if len(tail) >= 2 and looks_like_company_token(tail[0]) and looks_like_broker_name(tail[1]):
                company = format_company_token(tail[0])
                institution = tail[1]
                title = "_".join(tail[2:])
            elif looks_like_broker_name(tail[0]):
                institution = tail[0]
                title = "_".join(tail[1:])
            else:
                company = format_company_token(tail[0]) if looks_like_company_token(tail[0]) else ""
                institution = first_broker_text(tail[1:]) if company and len(tail) > 1 else (tail[0] if not company else "")
                broker_idx = tail.index(institution) if institution in tail else -1
                if broker_idx >= 0:
                    title = "_".join(tail[broker_idx + 1 :])
                else:
                    title = "_".join(tail[1:] if company and len(tail) > 1 else tail)
    match = RATING_RE.search(stem)
    if match:
        rating = match.group(1)
    title = clean_text(title.replace(".parsed.json", "").replace("_", " "))
    return {
        "date": date,
        "institution": clean_text(institution),
        "company": clean_text(company),
        "title": title,
        "rating": rating,
    }


def title_level(text: str) -> int:
    text = clean_text(text)
    if not is_structural_title(text):
        return 0
    if CN_SECTION_RE.match(text):
        return 1
    if re.match(r"^[（(][一二三四五六七八九十\d]+[）)]", text):
        return 2
    if re.match(r"^\d+[、.]", text):
        return 3
    match = SECTION_NUMBER_RE.match(text)
    if match:
        return match.group(1).count(".") + 3
    if len(text) <= 18:
        return 2
    return 1


def is_structural_title(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if "目录" in text:
        return True
    if re.search(r"[。！？!?；;]", text):
        return False
    if text.startswith(("资料来源", "图", "表")) and len(text) > 12:
        return False
    if re.match(r"^[一二三四五六七八九十]+[、.]", text):
        return True
    if re.match(r"^[（(][一二三四五六七八九十\d]+[）)]", text):
        return True
    if re.match(r"^\d+(?:\.\d+)*[、. ]", text):
        return True
    return len(text) <= 20


def parse_html_table(text: str) -> tuple[str, list[list[str]], list[list[str]]]:
    caption_match = re.search(r"<caption>(.*?)</caption>", text, flags=re.S)
    caption = clean_text(strip_tags(caption_match.group(1))) if caption_match else ""
    caption = sanitize_table_caption(caption)
    row_texts = re.findall(r"<tr>(.*?)</tr>", text, flags=re.S)
    headers: list[list[str]] = []
    rows: list[list[str]] = []
    for row_text in row_texts:
        header_cells = re.findall(r"<th[^>]*>(.*?)</th>", row_text, flags=re.S)
        data_cells = re.findall(r"<td[^>]*>(.*?)</td>", row_text, flags=re.S)
        if header_cells:
            headers.append([clean_text(strip_tags(cell)) for cell in header_cells])
        elif data_cells:
            rows.append([clean_text(strip_tags(cell)) for cell in data_cells])
    return caption, headers, rows


def group_table_rows(
    caption: str,
    headers: list[list[str]],
    rows: list[list[str]],
    chunk_size: int,
    tokens: TokenCounter,
) -> list[dict[str, Any]]:
    if not headers and not rows:
        return []
    header_text = "\n".join("| " + " | ".join(row) + " |" for row in headers)
    base_prefix = f"caption: {caption}\n{header_text}".strip()
    base_tokens = tokens.count(base_prefix)
    if not rows:
        return [{"row_start": 1, "row_end": 0, "text": base_prefix}]

    groups: list[dict[str, Any]] = []
    current_rows: list[list[str]] = []
    current_start = 1
    current_tokens = base_tokens

    def flush(end_index: int) -> None:
        nonlocal current_rows, current_start, current_tokens
        if not current_rows:
            return
        row_lines = ["| " + " | ".join(row) + " |" for row in current_rows]
        text = f"caption: {caption}\nsection_header:\n{header_text}\nrow_range: {current_start}-{end_index}\n" + "\n".join(row_lines)
        groups.append({"row_start": current_start, "row_end": end_index, "text": text.strip()})
        current_rows = []
        current_tokens = base_tokens

    for idx, row in enumerate(rows, start=1):
        row_line = "| " + " | ".join(row) + " |"
        row_tokens = tokens.count(row_line)
        if current_rows and current_tokens + row_tokens > chunk_size:
            flush(idx - 1)
            current_start = idx
        current_rows.append(row)
        current_tokens += row_tokens
    flush(len(rows))
    return groups


def table_col_count(headers: list[list[str]], rows: list[list[str]]) -> int:
    counts = [len(row) for row in headers + rows if row]
    return max(counts) if counts else 0


def normalize_table_caption(caption: str) -> str:
    caption = clean_text(caption)
    caption = re.sub(r"^资料来源[:：]?.*$", "", caption)
    caption = re.sub(r"\s+", "", caption)
    return caption


def headers_compatible(current_headers: list[list[str]], next_headers: list[list[str]]) -> bool:
    if not current_headers or not next_headers:
        return True
    left = " ".join(clean_text(cell) for row in current_headers for cell in row if clean_text(cell))
    right = " ".join(clean_text(cell) for row in next_headers for cell in row if clean_text(cell))
    if not left or not right:
        return True
    return left == right


def summarize_table(caption: str, headers: list[list[str]], rows: list[list[str]]) -> str:
    if table_rows_look_malformed(rows):
        return summarize_table_schema(caption, headers, rows)
    header = headers[-1] if headers else []
    summaries: list[str] = []
    for row in rows[:3]:
        if not row:
            continue
        label = row[0]
        values = []
        for idx, cell in enumerate(row[1:], start=1):
            if not cell:
                continue
            col = header[idx] if idx < len(header) and header[idx] else f"列{idx + 1}"
            if contains_numeric(cell):
                values.append(f"{col}={cell}")
        if values:
            summaries.append(f"{label}: " + "，".join(values[:4]))
    if summaries:
        return "；".join(summaries)
    if caption:
        return caption
    return ""


def table_rows_look_malformed(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    suspicious = 0
    for row in rows[:3]:
        if not row:
            continue
        first_cell = clean_text(row[0])
        stock_code_hits = len(re.findall(r"\d{6}\.(?:SH|SZ)|\d{4}\.HK", first_cell))
        token_like_count = len([part for part in re.split(r"\s+", first_cell) if clean_text(part)])
        company_like_hits = len(re.findall(r"[\u4e00-\u9fff]{2,}", first_cell))
        if stock_code_hits >= 2 or token_like_count >= 8 or company_like_hits >= 10 or len(first_cell) >= 60:
            suspicious += 1
    return suspicious >= 1


def summarize_table_schema(caption: str, headers: list[list[str]], rows: list[list[str]]) -> str:
    header = headers[-1] if headers else []
    fields = [clean_text(cell) for cell in header if clean_text(cell)]
    row_count = len(rows)
    if fields:
        return f"该表包含{row_count}行数据，字段包括：" + "、".join(fields[:8])
    if caption:
        return f"该表包含{row_count}行数据，主题为：{caption}"
    return f"该表包含{row_count}行数据"


def surrounding_sentences(target: CleanBlock, blocks: list[CleanBlock], limit: int = 3) -> str:
    same_page = [block for block in blocks if block.page == target.page and block.type == "text"]
    same_page = sorted(same_page, key=lambda item: (item.bbox[1], item.bbox[0]))
    above = [block for block in same_page if block.bbox[3] <= target.bbox[1]]
    below = [block for block in same_page if block.bbox[1] >= target.bbox[3]]
    sentences: list[str] = []

    # Prefer the nearest complete sentences above the target.
    for block in reversed(above[-3:]):
        block_sentences = [clean_text(sentence) for sentence in split_sentences(block.text) if clean_text(sentence)]
        for sentence in reversed(block_sentences):
            if sentence and sentence not in sentences:
                sentences.insert(0, sentence)
            if len(sentences) >= limit:
                return " ".join(sentences[-limit:])

    # Then append the nearest complete sentences below the target.
    for block in below[:3]:
        block_sentences = [clean_text(sentence) for sentence in split_sentences(block.text) if clean_text(sentence)]
        for sentence in block_sentences:
            if sentence and sentence not in sentences:
                sentences.append(sentence)
            if len(sentences) >= limit:
                return " ".join(sentences[:limit])

    return " ".join(sentences[:limit])


def figure_title(text: str) -> str:
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    for line in lines:
        if re.search(r"图\d+", line):
            return line
    return lines[0] if lines else ""


def figure_trend_hint(text: str) -> str:
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    hints = []
    for line in lines:
        if SOURCE_ONLY_RE.match(line):
            continue
        if NOISY_FIGURE_LINE_RE.match(line) and not FIGURE_MEANINGFUL_RE.search(line):
            continue
        if re.search(r"(同比|环比|增长|下降|占比|走势|趋势|变化)", line):
            hints.append(line)
    return " ".join(hints[:2])


def first_text(values: list[str]) -> str:
    for value in values:
        value = clean_text(value)
        if value:
            return value
    return ""


def first_match_text(values: list[str], pattern: re.Pattern[str]) -> str:
    for value in values:
        match = pattern.search(value or "")
        if match:
            return match.group(1)
    return ""


def first_broker_text(values: Any) -> str:
    for value in values:
        text = clean_text(str(value))
        if looks_like_broker_name(text):
            return text
    return ""


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def normalize_edge_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\d+", "#", text)
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])", text)
    return [part for part in parts if clean_text(part)]


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


def is_source_only_text(text: str) -> bool:
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    if not lines:
        return False
    if len(lines) == 1 and len(lines[0]) <= 48:
        return True
    return all(SOURCE_ONLY_RE.match(line) for line in lines)


def is_analyst_contact_block(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if len(text) > 160:
        return False
    if not ANALYST_CONTACT_RE.search(text):
        return False
    if re.search(r"(S\d{13,15}|SAC登记编号|登记编号|执业编号|邮箱|@|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text):
        return True
    if re.search(r"(电话|传真|联系电话|微信)", text) and re.search(r"(\+?\d[\d\-() ]{5,}\d)", text):
        return True
    if text.startswith(("分析师", "联系人")) and len(text) <= 64:
        return True
    return False


def looks_like_noisy_cover_text(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) < 8:
        return False
    latin_chunks = re.findall(r"[A-Za-z0-9]{2,}", text)
    if latin_chunks and sum(len(chunk) for chunk in latin_chunks) / max(len(text), 1) > 0.55 and len(re.findall(r"[\u4e00-\u9fff]", text)) <= 6:
        return True
    if len(re.findall(r"[A-Za-z]{2,}", text)) >= 4 and len(re.findall(r"[\u4e00-\u9fff]", text)) <= 4:
        return True
    return False


def looks_like_broker_name(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    return text.endswith(("证券", "证券研究所", "证券有限责任公司", "证券股份有限公司")) or "证券" in text


def looks_like_company_token(text: str) -> bool:
    text = clean_text(text)
    if not text or looks_like_broker_name(text):
        return False
    return bool(re.search(r"(?:[A-Za-z]?\d{6}|\d{5,6})$", text)) and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", text))


def format_company_token(text: str) -> str:
    text = clean_text(text)
    match = re.match(r"^(.*?)([A-Za-z]?\d{5,6})$", text)
    if not match:
        return text
    name, code = clean_text(match.group(1)), match.group(2)
    suffix = infer_stock_suffix(code)
    return f"{name}（{code}.{suffix}）" if name else f"{code}.{suffix}"


def infer_stock_suffix(code: str) -> str:
    digits = re.sub(r"\D", "", code)
    if digits.startswith(("920", "430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "874", "875", "876", "877", "878", "879")):
        return "BJ"
    if digits.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "SH"
    if digits.startswith(("000", "001", "002", "003", "300", "301", "200")):
        return "SZ"
    return "SH"


def is_noisy_title(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return True
    if SKIP_SECTION_RE.search(text):
        return True
    if text in {"研究所", "北京", "上海", "深圳", "联系人"}:
        return True
    if re.search(r"(table_|投tab|乱码|[\[\]@#]{2,})", text, flags=re.I):
        return True
    if looks_like_noisy_cover_text(text) and len(re.findall(r"[\u4e00-\u9fff]", text)) <= 6:
        return True
    if len(text) <= 2:
        return True
    return False


def is_terminal_section_title(text: str) -> bool:
    normalized = clean_text(text).strip("：:;；")
    return normalized in TERMINAL_SECTION_TITLES


def is_noise_text_block(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return True
    if len(text) <= 20 and text in {"中山证券研究所", "中邮证券研究所", "国金证券研究所", "研究所", "北京", "上海", "深圳"}:
        return True
    if len(text) <= 40 and re.search(r"(证券研究报告|行业周报|行业点评报告|研究所)$", text):
        return True
    return False


def is_toc_block(text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    if "目录" in text:
        return True
    if re.fullmatch(r"[\d\s.．·…-]{1,16}", text):
        return True
    if len(re.findall(r"\d+", text)) >= 6 and len(re.findall(r"[.．·…]{2,}", text)) >= 1:
        return True
    if len(re.findall(r"[一二三四五六七八九十]、", text)) >= 3 and len(re.findall(r"\d{1,2}", text)) >= 3:
        return True
    if len(re.findall(r"[（(][一二三四五六七八九十\d]+[）)]", text)) >= 3 and len(re.findall(r"\d{1,2}", text)) >= 4:
        return True
    if len(re.findall(r"(?:[一二三四五六七八九十]+、|[（(][一二三四五六七八九十\d]+[）)])", text)) >= 4 and len(re.findall(r"\d{1,2}", text)) >= 3:
        return True
    return False


def sanitize_table_caption(caption: str) -> str:
    caption = clean_text(caption)
    if not caption:
        return ""
    if caption.startswith("资料来源"):
        return ""
    for marker in ["资料来源", "图1", "图 1", "表1", "表 1"]:
        idx = caption.find(marker)
        if idx > 0:
            caption = caption[:idx].strip(" ：:;；，,。. ")
            break
    if re.search(r"图\d+", caption):
        caption = re.split(r"图\d+", caption, maxsplit=1)[0].strip(" ：:;；，,。. ")
    if caption.startswith("资料来源") or len(caption) <= 2:
        return ""
    return caption


def contains_numeric(text: str) -> bool:
    return bool(re.search(r"\d", text))


def dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def make_chunk_id(doc_id: str, chunk_kind: str, index: int) -> str:
    prefix = normalize_id(doc_id)
    return f"{prefix}__{chunk_kind}_{index:04d}"


def normalize_id(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "doc"
    return text[:max_len]


def sentence_aligned_tail(text: str, overlap_tokens: int, tokens: TokenCounter) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    candidate = tokens.slice_tail(text, overlap_tokens * 3)
    candidate = clean_text(candidate)
    if not candidate:
        return ""
    sentences = split_sentences(candidate)
    if not sentences:
        return tokens.slice_tail(text, overlap_tokens)
    kept: list[str] = []
    used = 0
    for sentence in reversed(sentences):
        sentence = clean_text(sentence)
        if not sentence:
            continue
        sentence_tokens = tokens.count(sentence)
        if kept and used + sentence_tokens > overlap_tokens:
            break
        if not kept and sentence_tokens > overlap_tokens:
            return sentence
        kept.insert(0, sentence)
        used += sentence_tokens
        if used >= overlap_tokens:
            break
    if kept:
        return "".join(kept).strip()
    return tokens.slice_tail(text, overlap_tokens)


def bbox_horizontal_overlap_ratio(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    right = min(float(a[2]), float(b[2]))
    overlap = max(0.0, right - left)
    width_a = max(1.0, float(a[2]) - float(a[0]))
    width_b = max(1.0, float(b[2]) - float(b[0]))
    return overlap / min(width_a, width_b)


def load_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.parsed.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    seen: set[Path] = set()
    out: list[Path] = []
    for file in files:
        if file not in seen:
            seen.add(file)
            out.append(file)
    return out


def build_output_path(parsed_file: Path, output_dir: Path | None, chunk_size: int, overlap: int) -> Path:
    name = parsed_file.name.replace(".parsed.json", f".chunks.cs{chunk_size}.ov{overlap}.json")
    if output_dir is None:
        return parsed_file.with_name(name)
    return output_dir / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-chunk parsed PDF JSON outputs into section-aware RAG chunks.")
    parser.add_argument("inputs", nargs="+", help="Parsed JSON files or directories containing them.")
    parser.add_argument("--output-dir", default=str(CHUNKED_ROOT / "chunked_512_100"), help="Optional output directory. Defaults to alongside each input file.")
    parser.add_argument("--chunk-size", type=int, default=512, choices=sorted(VALID_CHUNK_SIZES), help="Target chunk size in tokens.")
    parser.add_argument("--overlap", type=int, default=100, choices=sorted(VALID_OVERLAPS), help="Overlap in tokens within the same section only.")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    files = load_inputs(args.inputs)
    chunker = ParsedPdfChunker(chunk_size=args.chunk_size, overlap=args.overlap, encoding_name=args.encoding)

    for parsed_file in files:
        output_path = build_output_path(parsed_file, output_dir, args.chunk_size, args.overlap)
        result = chunker.process_file(parsed_file, output_path=output_path)
        print(
            f"chunked {parsed_file.name} -> cleaned_blocks={result['summary']['cleaned_blocks']} "
            f"removed={result['summary']['removed_blocks']} chunks={result['summary']['chunks']}"
        )


if __name__ == "__main__":
    main()

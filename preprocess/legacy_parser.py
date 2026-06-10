"""Layout-aware PDF parser for financial research reports.

The parser deliberately stays lightweight:
- PyMuPDF extracts text lines and geometry.
- pdfplumber extracts table geometry and cell text.
- Heuristics inspired by DeepDoc handle repeated header/footer removal,
  multi-column reading order, text merging, and table-protected chunking.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RAGFLOW_VENV_PYTHON = REPO_ROOT.parent / "ragflow" / ".venv" / "bin" / "python"

try:
    import fitz
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: PyMuPDF (import name: fitz).\n"
        "Install it in the Python environment you use to run this script, for example:\n"
        f"  uv pip install --python {RAGFLOW_VENV_PYTHON} PyMuPDF\n"
        "or run this legacy parser with an environment that already has PyMuPDF."
    ) from exc

try:
    import pdfplumber
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pdfplumber.\n"
        "Install it in the Python environment you use to run this script, for example:\n"
        f"  uv pip install --python {RAGFLOW_VENV_PYTHON} pdfplumber"
    ) from exc


@dataclass
class BBox:
    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x0, other.x0),
            min(self.top, other.top),
            max(self.x1, other.x1),
            max(self.bottom, other.bottom),
        )

    def to_list(self) -> list[float]:
        return [round(self.x0, 2), round(self.top, 2), round(self.x1, 2), round(self.bottom, 2)]


@dataclass
class TextLine:
    page: int
    bbox: BBox
    text: str
    font_size: float = 0.0
    block_no: int = 0
    line_no: int = 0
    col_id: int = 0
    is_full_width: bool = False


@dataclass
class ParsedBlock:
    id: str
    type: str
    page: int
    bbox: BBox
    text: str
    col_id: int = 0
    rows: list[list[str]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedChunk:
    id: str
    block_ids: list[str]
    pages: list[int]
    text: str
    char_count: int
    contains_table: bool = False


def parse_pdf(pdf_path: str | Path, output_dir: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    return FinancialPDFParser(**kwargs).parse(pdf_path, output_dir=output_dir)


class FinancialPDFParser:
    def __init__(
        self,
        max_chunk_chars: int = 1800,
        chunk_overlap_chars: int = 160,
        header_footer_min_repeats: int = 2,
        header_footer_ratio: float = 0.3,
        table_overlap_threshold: float = 0.18,
        merge_gap_ratio: float = 2.2,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.header_footer_min_repeats = header_footer_min_repeats
        self.header_footer_ratio = header_footer_ratio
        self.table_overlap_threshold = table_overlap_threshold
        self.merge_gap_ratio = merge_gap_ratio

    def parse(self, pdf_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        try:
            page_sizes = [{"page": i + 1, "width": doc[i].rect.width, "height": doc[i].rect.height} for i in range(len(doc))]
            raw_lines = self._extract_text_lines(doc)
        finally:
            doc.close()

        pdf_tables = self._extract_tables(pdf_path)
        visual_blocks = self._extract_visual_blocks(raw_lines, pdf_tables, page_sizes)
        tables = self._merge_table_blocks(pdf_tables, [b for b in visual_blocks if b.type == "table"])
        figures = [b for b in visual_blocks if b.type == "figure"]
        header_footer_keys = self._detect_repeated_headers_footers(raw_lines, page_sizes)
        lines = self._filter_lines(raw_lines, tables + figures, header_footer_keys, page_sizes)
        self._assign_columns(lines, page_sizes)
        text_blocks = self._merge_text_lines(lines, page_sizes)
        table_blocks = self._make_table_blocks(tables)
        blocks = self._order_blocks(text_blocks + table_blocks + figures, page_sizes)
        blocks = self._merge_cross_page_tables(blocks, page_sizes)
        blocks = self._merge_cross_page_continuations(blocks, page_sizes)
        self._renumber_blocks(blocks)
        chunks = self._make_chunks(blocks)

        result = {
            "source": str(pdf_path),
            "method": "PyMuPDF text + pdfplumber tables + layout heuristics",
            "summary": {
                "pages": len(page_sizes),
                "text_lines": len(raw_lines),
                "kept_text_lines": len(lines),
                "tables": sum(1 for block in blocks if block.type == "table"),
                "figures": sum(1 for block in blocks if block.type == "figure"),
                "blocks": len(blocks),
                "chunks": len(chunks),
            },
            "page_sizes": page_sizes,
            "blocks": [self._block_to_dict(b) for b in blocks],
            "chunks": [asdict(c) for c in chunks],
        }

        if output_dir:
            self.write_outputs(result, Path(output_dir))
        return result

    def _extract_text_lines(self, doc: fitz.Document) -> list[TextLine]:
        lines: list[TextLine] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            words = page.get_text("words", sort=False)
            grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for item in words:
                if len(item) < 8:
                    continue
                x0, y0, x1, y1, word, block_no, line_no, word_no = item[:8]
                text = self._clean_text(str(word))
                if not text:
                    continue
                grouped.setdefault((int(block_no), int(line_no)), []).append(
                    {"text": text, "bbox": BBox(float(x0), float(y0), float(x1), float(y1)), "word_no": int(word_no)}
                )

            for (block_no, line_no), group in grouped.items():
                for seg_no, segment in enumerate(self._split_word_segments(group)):
                    text = segment["text"]
                    if not text:
                        continue
                    lines.append(
                        TextLine(
                            page=page_index + 1,
                            bbox=segment["bbox"],
                            text=text,
                            font_size=segment["font_size"],
                            block_no=block_no,
                            line_no=line_no * 100 + seg_no,
                        )
                    )
        return lines

    def _extract_tables(self, pdf_path: Path) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 8,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
        }
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_index, page in enumerate(pdf.pages):
                try:
                    found = page.find_tables(table_settings=table_settings)
                except Exception:
                    found = []
                if not found:
                    try:
                        found = page.find_tables()
                    except Exception:
                        found = []

                for table_index, table in enumerate(found):
                    try:
                        rows = table.extract()
                    except Exception:
                        rows = []
                    rows = self._clean_table_rows(rows)
                    bbox = BBox(*table.bbox)
                    if not self._is_valid_table_candidate(rows, bbox, float(page.width), float(page.height)):
                        continue
                    text = self._table_to_markdown(rows)
                    blocks.append(
                        ParsedBlock(
                            id=f"p{page_index + 1}_table_{table_index + 1}",
                            type="table",
                            page=page_index + 1,
                            bbox=bbox,
                            text=text,
                            rows=rows,
                            meta={
                                "row_count": len(rows),
                                "column_count": max((len(r) for r in rows), default=0),
                                "row_bands": [list(getattr(row, "bbox", ())) for row in getattr(table, "rows", [])],
                            },
                        )
                    )
        return blocks

    def _extract_visual_blocks(
        self,
        lines: list[TextLine],
        pdf_tables: list[ParsedBlock],
        page_sizes: list[dict[str, float]],
    ) -> list[ParsedBlock]:
        by_page: dict[int, list[TextLine]] = {}
        for line in lines:
            by_page.setdefault(line.page, []).append(line)

        table_by_page: dict[int, list[ParsedBlock]] = {}
        for table in pdf_tables:
            table_by_page.setdefault(table.page, []).append(table)

        size_by_page = {int(p["page"]): p for p in page_sizes}
        blocks: list[ParsedBlock] = []
        for page, page_lines in by_page.items():
            page_lines = sorted(page_lines, key=lambda ln: (ln.bbox.top, ln.bbox.x0))
            used_regions: list[BBox] = []
            page_width = float(size_by_page[page]["width"])
            page_height = float(size_by_page[page]["height"])
            for idx, line in enumerate(page_lines):
                if any(self._point_in_bbox(line.bbox.center_x, (line.bbox.top + line.bbox.bottom) / 2, region) for region in used_regions):
                    continue
                if not self._looks_like_caption(line):
                    continue
                source_idx = self._find_region_source_index(page_lines, idx)
                has_source = source_idx is not None
                end_idx = source_idx + 1 if source_idx is not None else self._find_region_end_index(page_lines, idx, page_height)
                if end_idx is None or end_idx <= idx + 1:
                    continue

                x0, x1 = self._caption_region_x_bounds(line, page_width) if not has_source else (0.0, page_width)
                region_lines = [
                    ln
                    for ln in page_lines[idx:end_idx]
                    if ln.bbox.top >= line.bbox.top - 2
                    and (has_source or (x0 <= ln.bbox.center_x <= x1))
                ]
                if len(region_lines) < 3:
                    continue
                bbox = self._union_line_bboxes(region_lines)
                bbox = BBox(max(0, bbox.x0 - 4), max(0, bbox.top - 4), min(page_width, bbox.x1 + 4), min(page_height, bbox.bottom + 4))
                if bbox.bottom - line.bbox.top < 45:
                    continue

                overlapping_tables = [tb for tb in table_by_page.get(page, []) if self._overlap_ratio(tb.bbox, bbox) > 0.15 or self._overlap_ratio(bbox, tb.bbox) > 0.15]
                data_lines = [ln for ln in region_lines if ln is not line and not self._looks_like_caption(ln) and not self._looks_like_source(ln)]
                numeric_ratio = self._numeric_fragment_ratio(data_lines)
                is_figure_caption = self._looks_like_figure_caption(line)

                if overlapping_tables or (not is_figure_caption and self._looks_like_table_region(data_lines, page_width)):
                    row_bands = overlapping_tables[0].meta.get("row_bands") if overlapping_tables else None
                    rows = self._region_to_table_rows(data_lines, row_bands=row_bands)
                    if not rows:
                        continue
                    source_text = next((ln.text for ln in region_lines if self._looks_like_source(ln)), "")
                    table_text = line.text + "\n" + self._table_to_markdown(rows) + (f"\n{source_text}" if source_text else "")
                    blocks.append(
                        ParsedBlock(
                            id=f"p{page}_visual_table_{len(blocks) + 1}",
                            type="table",
                            page=page,
                            bbox=bbox,
                            text=table_text,
                            rows=rows,
                            meta={
                                "row_count": len(rows),
                                "column_count": max((len(r) for r in rows), default=0),
                                "source": "visual_region",
                            },
                        )
                    )
                elif is_figure_caption or numeric_ratio >= 0.45 or len(data_lines) >= 8:
                    caption_texts = [ln.text for ln in region_lines if self._looks_like_caption(ln)]
                    source_texts = [ln.text for ln in region_lines if self._looks_like_source(ln)]
                    figure_text = "\n".join(dict.fromkeys(caption_texts + source_texts))
                    blocks.append(
                        ParsedBlock(
                            id=f"p{page}_figure_{len(blocks) + 1}",
                            type="figure",
                            page=page,
                            bbox=bbox,
                            text=figure_text or line.text,
                            meta={
                                "source": "caption_region" if has_source else "caption_fallback_region",
                                "numeric_fragment_ratio": round(numeric_ratio, 3),
                                "suppressed_lines": len(data_lines),
                            },
                        )
                    )
                used_regions.append(bbox)
            uncaptioned_tables = self._extract_uncaptioned_table_blocks(page, page_lines, table_by_page.get(page, []), page_width, page_height)
            blocks.extend(uncaptioned_tables)
            blocks.extend(self._extract_dense_table_blocks(page, page_lines, table_by_page.get(page, []) + uncaptioned_tables, page_width, page_height))
        return blocks

    def _merge_table_blocks(self, pdf_tables: list[ParsedBlock], visual_tables: list[ParsedBlock]) -> list[ParsedBlock]:
        merged = list(visual_tables)
        for table in pdf_tables:
            if any(table.page == vt.page and (self._overlap_ratio(table.bbox, vt.bbox) > 0.15 or self._overlap_ratio(vt.bbox, table.bbox) > 0.15) for vt in visual_tables):
                continue
            merged.append(table)
        return merged

    def _split_line_segments(self, line: dict[str, Any]) -> list[dict[str, Any]]:
        spans = []
        for span in line.get("spans", []):
            text = self._clean_text(span.get("text", ""))
            if not text:
                continue
            spans.append({"text": text, "bbox": BBox(*span["bbox"]), "size": float(span.get("size", 0))})
        if not spans:
            return []
        spans.sort(key=lambda s: s["bbox"].x0)

        groups: list[list[dict[str, Any]]] = [[spans[0]]]
        for span in spans[1:]:
            prev = groups[-1][-1]
            gap = span["bbox"].x0 - prev["bbox"].x1
            prev_h = max(prev["bbox"].height, 1.0)
            big_gap = gap > max(24.0, prev_h * 2.6)
            if big_gap:
                groups.append([span])
            else:
                groups[-1].append(span)

        segments: list[dict[str, Any]] = []
        for group in groups:
            bbox = group[0]["bbox"]
            for span in group[1:]:
                bbox = bbox.union(span["bbox"])
            text = self._clean_text(" ".join(span["text"] for span in group))
            sizes = [span["size"] for span in group if span["size"]]
            segments.append({"text": text, "bbox": bbox, "font_size": median(sizes) if sizes else 0.0})
        return segments

    def _split_word_segments(self, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not words:
            return []
        words = sorted(words, key=lambda w: (w["bbox"].x0, w["word_no"]))
        heights = [w["bbox"].height for w in words if w["bbox"].height > 0]
        line_height = median(heights) if heights else 10.0
        groups: list[list[dict[str, Any]]] = [[words[0]]]
        for word in words[1:]:
            prev = groups[-1][-1]
            gap = word["bbox"].x0 - prev["bbox"].x1
            big_gap = gap > max(22.0, line_height * 2.4)
            if big_gap:
                groups.append([word])
            else:
                groups[-1].append(word)

        segments: list[dict[str, Any]] = []
        for group in groups:
            bbox = group[0]["bbox"]
            for word in group[1:]:
                bbox = bbox.union(word["bbox"])
            text = self._join_word_text([word["text"] for word in group])
            segments.append({"text": text, "bbox": bbox, "font_size": bbox.height})
        return segments

    def _detect_repeated_headers_footers(self, lines: list[TextLine], page_sizes: list[dict[str, float]]) -> set[str]:
        page_count = len(page_sizes)
        min_repeats = max(self.header_footer_min_repeats, math.ceil(page_count * self.header_footer_ratio))
        counts: dict[str, set[int]] = {}
        size_by_page = {int(p["page"]): p for p in page_sizes}
        for line in lines:
            size = size_by_page[line.page]
            top_zone = line.bbox.top < size["height"] * 0.12
            bottom_zone = line.bbox.bottom > size["height"] * 0.90
            if not (top_zone or bottom_zone):
                continue
            key = self._header_footer_key(line.text)
            if not key:
                continue
            counts.setdefault(key, set()).add(line.page)

        repeated = {key for key, pages in counts.items() if len(pages) >= min_repeats}
        return repeated

    def _filter_lines(
        self,
        lines: list[TextLine],
        tables: list[ParsedBlock],
        header_footer_keys: set[str],
        page_sizes: list[dict[str, float]],
    ) -> list[TextLine]:
        table_by_page: dict[int, list[BBox]] = {}
        for table in tables:
            table_by_page.setdefault(table.page, []).append(table.bbox)

        size_by_page = {int(p["page"]): p for p in page_sizes}
        kept: list[TextLine] = []
        for line in lines:
            if self._is_page_number(line.text):
                continue
            key = self._header_footer_key(line.text)
            if key in header_footer_keys:
                continue
            if self._line_in_margin_noise(line, size_by_page[line.page]):
                continue
            if any(self._overlap_ratio(line.bbox, tb) >= self.table_overlap_threshold for tb in table_by_page.get(line.page, [])):
                continue
            kept.append(line)
        return kept

    def _assign_columns(self, lines: list[TextLine], page_sizes: list[dict[str, float]]) -> None:
        by_page: dict[int, list[TextLine]] = {}
        for line in lines:
            by_page.setdefault(line.page, []).append(line)

        size_by_page = {int(p["page"]): p for p in page_sizes}
        for page, page_lines in by_page.items():
            width = float(size_by_page[page]["width"])
            body_lines = [ln for ln in page_lines if ln.bbox.width >= 20 and ln.bbox.width < width * 0.72]
            split = self._find_column_split(body_lines, width)

            for line in page_lines:
                line.is_full_width = line.bbox.width >= width * 0.68
                if split is None or line.is_full_width:
                    line.col_id = 0
                else:
                    line.col_id = 0 if line.bbox.center_x < split else 1

    def _merge_text_lines(self, lines: list[TextLine], page_sizes: list[dict[str, float]]) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        by_page_col: dict[tuple[int, int], list[TextLine]] = {}
        for line in lines:
            key = (line.page, line.col_id)
            by_page_col.setdefault(key, []).append(line)

        for (page, col_id), group in sorted(by_page_col.items()):
            group = sorted(group, key=lambda ln: (ln.bbox.top, ln.bbox.x0))
            if not group:
                continue
            heights = [ln.bbox.height for ln in group if ln.bbox.height > 0]
            line_height = median(heights) if heights else 10.0
            current: list[TextLine] = []
            current_bbox: BBox | None = None

            def flush() -> None:
                nonlocal current, current_bbox
                if not current or current_bbox is None:
                    return
                text = self._join_lines(current)
                if text:
                    blocks.append(
                        ParsedBlock(
                            id=f"p{page}_text_{len(blocks) + 1}",
                            type="text",
                            page=page,
                            bbox=current_bbox,
                            text=text,
                            col_id=max(0, col_id),
                            meta={
                                "line_count": len(current),
                                "avg_font_size": round(sum(ln.font_size for ln in current) / max(1, len(current)), 2),
                            },
                        )
                    )
                current = []
                current_bbox = None

            for line in group:
                if not current:
                    current = [line]
                    current_bbox = line.bbox
                    continue
                prev = current[-1]
                gap = line.bbox.top - prev.bbox.bottom
                same_x_band = self._horizontal_overlap_ratio(prev.bbox, line.bbox) >= 0.22
                detached_same_row = (
                    not same_x_band
                    and abs(line.bbox.top - prev.bbox.top) < line_height * 0.85
                    and abs(line.bbox.center_x - prev.bbox.center_x) > line_height * 4
                )
                strong_break = self._should_break_paragraph(prev, line, line_height)
                too_far = gap > line_height * self.merge_gap_ratio
                if too_far or detached_same_row or (not same_x_band and gap > line_height * 0.35) or strong_break:
                    flush()
                    current = [line]
                    current_bbox = line.bbox
                else:
                    current.append(line)
                    current_bbox = current_bbox.union(line.bbox) if current_bbox else line.bbox
            flush()

        return blocks

    def _make_table_blocks(self, tables: list[ParsedBlock]) -> list[ParsedBlock]:
        return tables

    def _find_region_source_index(self, lines: list[TextLine], caption_idx: int) -> int | None:
        caption = lines[caption_idx]
        for idx in range(caption_idx + 1, min(len(lines), caption_idx + 180)):
            line = lines[idx]
            if line.page != caption.page:
                break
            if line.bbox.top - caption.bbox.top > 620:
                break
            if self._looks_like_source(line):
                return idx
            same_caption_band = abs(line.bbox.top - caption.bbox.top) < max(18.0, caption.bbox.height * 1.6)
            if idx > caption_idx + 3 and self._looks_like_caption(line) and not same_caption_band:
                return None
        return None

    def _extract_uncaptioned_table_blocks(
        self,
        page: int,
        page_lines: list[TextLine],
        page_tables: list[ParsedBlock],
        page_width: float,
        page_height: float,
    ) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        for idx, line in enumerate(page_lines):
            if not self._looks_like_uncaptioned_table_title(line):
                continue
            if any(self._overlap_ratio(line.bbox, block.bbox) > 0.05 for block in page_tables):
                continue
            x0 = max(0.0, line.bbox.x0 - 8)
            x1 = page_width
            region_lines: list[TextLine] = [line]
            for next_line in page_lines[idx + 1 : min(len(page_lines), idx + 100)]:
                if next_line.page != page:
                    break
                if next_line.bbox.top - line.bbox.top > 260:
                    break
                if next_line.bbox.center_x < x0 or next_line.bbox.center_x > x1:
                    continue
                if next_line.bbox.top > line.bbox.top + 30 and self._looks_like_heading(next_line) and next_line.bbox.x0 < x0 + 8:
                    break
                region_lines.append(next_line)
                if self._looks_like_source(next_line):
                    break
            data_lines = [ln for ln in region_lines if ln is not line and not self._looks_like_source(ln)]
            if not self._looks_like_table_region(data_lines, page_width):
                continue
            rows = self._region_to_table_rows(data_lines)
            if len(rows) < 4:
                continue
            bbox = self._union_line_bboxes(region_lines)
            bbox = BBox(max(0, bbox.x0 - 4), max(0, bbox.top - 4), min(page_width, bbox.x1 + 4), min(page_height, bbox.bottom + 4))
            text = line.text + "\n" + self._table_to_markdown(rows)
            source = next((ln.text for ln in region_lines if self._looks_like_source(ln)), "")
            if source:
                text += "\n" + source
            blocks.append(
                ParsedBlock(
                    id=f"p{page}_uncaptioned_table_{len(blocks) + 1}",
                    type="table",
                    page=page,
                    bbox=bbox,
                    text=text,
                    rows=rows,
                    meta={
                        "row_count": len(rows),
                        "column_count": max((len(row) for row in rows), default=0),
                        "source": "uncaptioned_table_region",
                    },
                )
            )
        return blocks

    def _extract_dense_table_blocks(
        self,
        page: int,
        page_lines: list[TextLine],
        existing_tables: list[ParsedBlock],
        page_width: float,
        page_height: float,
    ) -> list[ParsedBlock]:
        rows = self._group_lines_by_row([ln for ln in page_lines if not self._looks_like_source(ln)])
        row_infos = []
        for row in rows:
            row = sorted(row, key=lambda ln: ln.bbox.x0)
            bbox = self._union_line_bboxes(row)
            cells = self._row_lines_to_cells(row)
            non_numeric = [ln for ln in row if not self._looks_like_numeric_fragment(ln.text)]
            left_label = any((ln.bbox.center_x < page_width * 0.42 and len(ln.text.strip()) >= 2) for ln in non_numeric)
            row_infos.append({"row": row, "bbox": bbox, "cells": cells, "tableish": len(cells) >= 3 and left_label})

        blocks: list[ParsedBlock] = []
        i = 0
        while i < len(row_infos):
            if not row_infos[i]["tableish"]:
                i += 1
                continue
            start = i
            end = i + 1
            while end < len(row_infos):
                prev = row_infos[end - 1]["bbox"]
                cur = row_infos[end]["bbox"]
                gap = cur.top - prev.bottom
                if row_infos[end]["tableish"] and gap <= max(18.0, prev.height * 2.5):
                    end += 1
                    continue
                break
            if end - start < 5:
                i = end
                continue

            candidate_rows = [info["row"] for info in row_infos[start:end]]
            candidate_lines = [ln for row in candidate_rows for ln in row]
            bbox = self._union_line_bboxes(candidate_lines)
            candidate_text = " ".join(ln.text for ln in candidate_lines)
            if bbox.top > page_height * 0.50 and re.search(r"(电话|邮箱|地址|邮编|上海|北京|深圳|版权所有|此报告)", candidate_text):
                i = end
                continue
            if bbox.width < page_width * 0.34 or bbox.height < 45:
                i = end
                continue
            if any(self._overlap_ratio(bbox, table.bbox) > 0.35 or self._overlap_ratio(table.bbox, bbox) > 0.35 for table in existing_tables + blocks):
                i = end
                continue
            if self._numeric_fragment_ratio(candidate_lines) > 0.75:
                i = end
                continue

            table_rows = [self._row_lines_to_cells(row) for row in candidate_rows]
            max_cols = max((len(row) for row in table_rows), default=0)
            if max_cols < 3:
                i = end
                continue
            table_rows = [row + [""] * (max_cols - len(row)) for row in table_rows]
            title = self._nearest_table_title(page_lines, bbox)
            if re.search(r"(电话|邮箱|地址|邮编|版权所有|此报告)", title):
                i = end
                continue
            text = (title + "\n" if title else "") + self._table_to_markdown(table_rows)
            blocks.append(
                ParsedBlock(
                    id=f"p{page}_dense_table_{len(blocks) + 1}",
                    type="table",
                    page=page,
                    bbox=BBox(max(0, bbox.x0 - 4), max(0, bbox.top - 4), min(page_width, bbox.x1 + 4), min(page_height, bbox.bottom + 4)),
                    text=text,
                    rows=table_rows,
                    meta={
                        "row_count": len(table_rows),
                        "column_count": max_cols,
                        "source": "dense_text_table_region",
                    },
                )
            )
            i = end
        return blocks

    def _nearest_table_title(self, page_lines: list[TextLine], bbox: BBox) -> str:
        candidates = [
            ln
            for ln in page_lines
            if ln.bbox.bottom <= bbox.top
            and bbox.top - ln.bbox.bottom <= 28
            and self._horizontal_overlap_ratio(ln.bbox, bbox) > 0.15
            and not self._looks_like_numeric_fragment(ln.text)
        ]
        if not candidates:
            return ""
        return sorted(candidates, key=lambda ln: (bbox.top - ln.bbox.bottom, -len(ln.text)))[0].text

    def _find_region_end_index(self, lines: list[TextLine], caption_idx: int, page_height: float) -> int | None:
        caption = lines[caption_idx]
        for idx in range(caption_idx + 1, min(len(lines), caption_idx + 90)):
            line = lines[idx]
            if line.page != caption.page:
                return idx
            if line.bbox.top - caption.bbox.top > 320:
                return idx
            lower_caption = self._looks_like_caption(line) and line.bbox.top - caption.bbox.top > max(28.0, caption.bbox.height * 2.0)
            if lower_caption:
                return idx
            if line.bbox.top > page_height * 0.92:
                return idx
        return min(len(lines), caption_idx + 90)

    @staticmethod
    def _caption_region_x_bounds(line: TextLine, page_width: float) -> tuple[float, float]:
        if line.bbox.center_x > page_width * 0.55:
            return page_width * 0.45, page_width
        if line.bbox.center_x < page_width * 0.45:
            return 0.0, page_width * 0.55
        return 0.0, page_width

    @staticmethod
    def _union_line_bboxes(lines: list[TextLine]) -> BBox:
        bbox = lines[0].bbox
        for line in lines[1:]:
            bbox = bbox.union(line.bbox)
        return bbox

    @staticmethod
    def _numeric_fragment_ratio(lines: list[TextLine]) -> float:
        if not lines:
            return 0.0
        numeric = 0
        for line in lines:
            if FinancialPDFParser._looks_like_numeric_fragment(line.text):
                numeric += 1
        return numeric / len(lines)

    @staticmethod
    def _looks_like_numeric_fragment(text: str) -> bool:
        text = text.strip()
        return bool(re.fullmatch(r"[-+0-9.,%/ ]+", text) or re.fullmatch(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*)+", text))

    def _looks_like_table_region(self, lines: list[TextLine], page_width: float) -> bool:
        if len(lines) < 4:
            return False
        rows = self._group_lines_by_row(lines)
        multi_cell_rows = [row for row in rows if len(row) >= 2]
        if len(multi_cell_rows) < 3:
            return False
        textish = sum(1 for line in lines if len(line.text) > 3 and not re.fullmatch(r"[-+0-9.,%/ ]+", line.text.strip()))
        if textish < 4:
            return False
        x_positions = sorted({round(line.bbox.x0 / 12) * 12 for line in lines})
        return len(x_positions) >= 2 and (max(line.bbox.x1 for line in lines) - min(line.bbox.x0 for line in lines)) > page_width * 0.28

    def _region_to_table_rows(self, lines: list[TextLine], row_bands: list[list[float]] | None = None) -> list[list[str]]:
        rows = []
        grouped_rows = self._group_lines_by_bands(lines, row_bands) if row_bands else self._group_lines_by_row(lines)
        for row in grouped_rows:
            cells = self._row_lines_to_cells(row)
            if cells:
                rows.append(cells)
        if not rows:
            return []
        max_cols = max(len(row) for row in rows)
        if max_cols < 2:
            return []
        return [row + [""] * (max_cols - len(row)) for row in rows]

    def _row_lines_to_cells(self, row: list[TextLine]) -> list[str]:
        groups: list[list[TextLine]] = []
        for line in sorted(row, key=lambda ln: (ln.bbox.x0, ln.bbox.top)):
            if not line.text.strip():
                continue
            placed = False
            for group in groups:
                gx0 = min(ln.bbox.x0 for ln in group)
                gx1 = max(ln.bbox.x1 for ln in group)
                overlap = max(0.0, min(gx1, line.bbox.x1) - max(gx0, line.bbox.x0))
                if abs(line.bbox.x0 - gx0) < 22 or overlap > min(line.bbox.width, max(1.0, gx1 - gx0)) * 0.35:
                    group.append(line)
                    placed = True
                    break
            if not placed:
                groups.append([line])

        cells: list[str] = []
        for group in groups:
            group = sorted(group, key=lambda ln: (ln.bbox.top, ln.bbox.x0))
            cells.append(self._join_word_text([ln.text for ln in group]))
        return cells

    @staticmethod
    def _group_lines_by_bands(lines: list[TextLine], row_bands: list[list[float]] | None) -> list[list[TextLine]]:
        if not row_bands:
            return FinancialPDFParser._group_lines_by_row(lines)
        bands = [band for band in row_bands if len(band) == 4]
        rows: list[list[TextLine]] = [[] for _ in bands]
        leftovers: list[TextLine] = []
        for line in lines:
            cy = (line.bbox.top + line.bbox.bottom) / 2
            best_idx = None
            best_distance = 10**9
            for idx, band in enumerate(bands):
                top, bottom = float(band[1]), float(band[3])
                margin = max(8.0, (bottom - top) * 0.18)
                if top - margin <= cy <= bottom + margin:
                    center = (top + bottom) / 2
                    distance = abs(cy - center)
                    if distance < best_distance:
                        best_idx = idx
                        best_distance = distance
            if best_idx is None:
                leftovers.append(line)
            else:
                rows[best_idx].append(line)
        rows = [row for row in rows if row]
        if leftovers:
            rows.extend(FinancialPDFParser._group_lines_by_row(leftovers))
        return rows

    @staticmethod
    def _group_lines_by_row(lines: list[TextLine]) -> list[list[TextLine]]:
        lines = sorted(lines, key=lambda ln: (ln.bbox.top, ln.bbox.x0))
        rows: list[list[TextLine]] = []
        for line in lines:
            if not rows:
                rows.append([line])
                continue
            row = rows[-1]
            row_top = sum(ln.bbox.top for ln in row) / len(row)
            row_h = max(8.0, sum(ln.bbox.height for ln in row) / len(row))
            if abs(line.bbox.top - row_top) <= row_h * 0.75:
                row.append(line)
            else:
                rows.append([line])
        return rows

    def _order_blocks(self, blocks: list[ParsedBlock], page_sizes: list[dict[str, float]]) -> list[ParsedBlock]:
        ordered: list[ParsedBlock] = []
        by_page: dict[int, list[ParsedBlock]] = {}
        for block in blocks:
            by_page.setdefault(block.page, []).append(block)
        size_by_page = {int(p["page"]): p for p in page_sizes}

        for page in sorted(by_page):
            width = float(size_by_page[page]["width"])
            page_blocks = by_page[page]
            for block in page_blocks:
                if block.bbox.width >= width * 0.68:
                    block.meta["full_width"] = True
                elif "full_width" not in block.meta:
                    block.meta["full_width"] = False
            ordered.extend(self._order_page_blocks(page_blocks, width))

        for idx, block in enumerate(ordered, start=1):
            block.id = f"b{idx:05d}_{block.type}_p{block.page}"
        return ordered

    def _renumber_blocks(self, blocks: list[ParsedBlock]) -> None:
        for idx, block in enumerate(blocks, start=1):
            block.id = f"b{idx:05d}_{block.type}_p{block.page}"

    def _merge_cross_page_continuations(self, blocks: list[ParsedBlock], page_sizes: list[dict[str, float]]) -> list[ParsedBlock]:
        if not blocks:
            return []
        page_height = {int(p["page"]): float(p["height"]) for p in page_sizes}
        merged: list[ParsedBlock] = []
        i = 0
        while i < len(blocks):
            cur = blocks[i]
            if (
                merged
                and cur.type == "text"
                and merged[-1].type == "text"
                and cur.page == merged[-1].page + 1
                and self._is_cross_page_continuation(merged[-1], cur, page_height)
            ):
                prev = merged[-1]
                prev.text = self._join_block_text(prev.text, cur.text)
                prev.meta["continued_to_page"] = cur.page
                prev.meta["continued_block_ids"] = prev.meta.get("continued_block_ids", []) + [cur.id]
                prev.meta["line_count"] = int(prev.meta.get("line_count", 1)) + int(cur.meta.get("line_count", 1))
                i += 1
                continue
            merged.append(cur)
            i += 1
        return merged

    def _merge_cross_page_tables(self, blocks: list[ParsedBlock], page_sizes: list[dict[str, float]]) -> list[ParsedBlock]:
        if not blocks:
            return []
        page_height = {int(p["page"]): float(p["height"]) for p in page_sizes}
        merged: list[ParsedBlock] = []
        i = 0
        while i < len(blocks):
            cur = blocks[i]
            candidate_idx = None
            if cur.type == "table":
                has_current_page_table = any(block.type == "table" and block.page == cur.page for block in merged)
                if not has_current_page_table:
                    for idx in range(len(merged) - 1, -1, -1):
                        if merged[idx].type == "table" and merged[idx].page == cur.page - 1:
                            candidate_idx = idx
                            break
            if candidate_idx is not None and self._is_cross_page_table_continuation(merged[candidate_idx], cur, page_height):
                prev = merged[candidate_idx]
                prev.rows = self._merge_table_rows(prev.rows or [], cur.rows or [])
                prev.text = self._merged_table_text(prev, cur)
                prev.meta["pages"] = sorted(set(prev.meta.get("pages", [prev.page]) + cur.meta.get("pages", [cur.page])))
                prev.meta["continued_to_page"] = max(prev.meta["pages"])
                prev.meta["continued_table_ids"] = prev.meta.get("continued_table_ids", []) + [cur.id]
                prev.meta["page_bboxes"] = prev.meta.get("page_bboxes", {str(prev.page): prev.bbox.to_list()})
                prev.meta["page_bboxes"][str(cur.page)] = cur.bbox.to_list()
                prev.meta["row_count"] = len(prev.rows or [])
                prev.meta["column_count"] = max((len(row) for row in (prev.rows or [])), default=0)
                i += 1
                continue
            merged.append(cur)
            i += 1
        return merged

    def _is_cross_page_table_continuation(self, prev: ParsedBlock, cur: ParsedBlock, page_height: dict[int, float]) -> bool:
        prev_height = page_height.get(prev.page, 842.0)
        cur_height = page_height.get(cur.page, 842.0)
        if prev.bbox.bottom < prev_height * 0.76:
            return False
        if cur.bbox.top > cur_height * 0.24:
            return False
        if abs(prev.bbox.x0 - cur.bbox.x0) > 70:
            return False
        prev_cols = max((len(row) for row in (prev.rows or [])), default=0)
        cur_cols = max((len(row) for row in (cur.rows or [])), default=0)
        if prev_cols < 2 or cur_cols < 2 or prev_cols != cur_cols:
            return False
        if self._table_starts_with_explicit_title(cur):
            return False
        return True

    def _merge_table_rows(self, left: list[list[str]], right: list[list[str]]) -> list[list[str]]:
        if not left:
            return right
        if not right:
            return left
        left_header = [cell.strip() for cell in left[0]]
        right_header = [cell.strip() for cell in right[0]]
        if left_header == right_header:
            right = right[1:]
        max_cols = max([len(row) for row in left + right] or [0])
        return [row + [""] * (max_cols - len(row)) for row in left + right]

    def _merged_table_text(self, prev: ParsedBlock, cur: ParsedBlock) -> str:
        title = self._table_text_prefix(prev.text)
        sources = []
        for text in (prev.text, cur.text):
            sources.extend(re.findall(r"(?:资料来源|数据来源|来源)[:：][^\n|]+", text))
        parts = []
        if title:
            parts.append(title)
        parts.append(self._table_to_markdown(prev.rows or []))
        if sources:
            parts.append("\n".join(dict.fromkeys(source.strip() for source in sources)))
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _table_text_prefix(text: str) -> str:
        prefix = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("|"):
                break
            if re.match(r"^(资料来源|数据来源|来源)[:：]", stripped):
                continue
            prefix.append(stripped)
        return "\n".join(prefix)

    @staticmethod
    def _table_starts_with_explicit_title(block: ParsedBlock) -> bool:
        prefix = FinancialPDFParser._table_text_prefix(block.text)
        return bool(re.match(r"^[\uf06c•●◆◼▪-]?\s*(图表|表)\s*\d*[:：]", prefix.strip()))

    def _is_cross_page_continuation(self, prev: ParsedBlock, cur: ParsedBlock, page_height: dict[int, float]) -> bool:
        prev_height = page_height.get(prev.page, 842.0)
        cur_height = page_height.get(cur.page, 842.0)
        if prev.bbox.bottom < prev_height * 0.78:
            return False
        if cur.bbox.top > cur_height * 0.24:
            return False
        if abs(prev.bbox.x0 - cur.bbox.x0) > 42:
            return False
        if self._block_starts_new_section(cur.text):
            return False
        if self._looks_like_source_text(prev.text) or self._looks_like_source_text(cur.text):
            return False
        return True

    def _join_block_text(self, left: str, right: str) -> str:
        left = left.strip()
        right = right.strip()
        if not left:
            return right
        if not right:
            return left
        if self._should_join_without_space(left, right):
            return left + right
        return left + " " + right

    @staticmethod
    def _block_starts_new_section(text: str) -> bool:
        text = text.strip()
        return bool(re.match(r"^([一二三四五六七八九十]+[、.．]|[0-9]+[、.．]|第.+[章节]|图表|图|表)\s*", text))

    @staticmethod
    def _looks_like_source_text(text: str) -> bool:
        return bool(re.match(r"^(资料来源|数据来源|来源)[:：]", text.strip()))

    def _order_page_blocks(self, blocks: list[ParsedBlock], page_width: float) -> list[ParsedBlock]:
        if not blocks:
            return []
        has_two_cols = len({b.col_id for b in blocks if not b.meta.get("full_width")}) > 1
        if not has_two_cols:
            return sorted(blocks, key=lambda b: (b.bbox.top, b.bbox.x0))

        full = sorted([b for b in blocks if b.meta.get("full_width")], key=lambda b: (b.bbox.top, b.bbox.x0))
        regular = [b for b in blocks if not b.meta.get("full_width")]
        result: list[ParsedBlock] = []
        cursor_top = 0.0
        for full_block in full:
            band = [b for b in regular if cursor_top <= b.bbox.top < full_block.bbox.top]
            result.extend(self._order_column_band(band))
            regular = [b for b in regular if b not in band]
            result.append(full_block)
            cursor_top = full_block.bbox.bottom
        result.extend(self._order_column_band(regular))
        return result

    def _order_column_band(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        return sorted(blocks, key=lambda b: (b.col_id, b.bbox.top, b.bbox.x0))

    def _make_chunks(self, blocks: list[ParsedBlock]) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        current_parts: list[str] = []
        current_ids: list[str] = []
        current_pages: set[int] = set()
        current_has_table = False

        def flush() -> None:
            nonlocal current_parts, current_ids, current_pages, current_has_table
            text = "\n\n".join(part for part in current_parts if part.strip()).strip()
            if not text:
                current_parts = []
                current_ids = []
                current_pages = set()
                current_has_table = False
                return
            chunks.append(
                ParsedChunk(
                    id=f"chunk_{len(chunks) + 1:05d}",
                    block_ids=list(current_ids),
                    pages=sorted(current_pages),
                    text=text,
                    char_count=len(text),
                    contains_table=current_has_table,
                )
            )
            overlap = self._tail_overlap(text)
            current_parts = [overlap] if overlap else []
            current_ids = []
            current_pages = set()
            current_has_table = False

        for block in blocks:
            part = self._block_chunk_text(block)
            if not part:
                continue
            is_table = block.type == "table"
            current_len = len("\n\n".join(current_parts))
            would_exceed = current_len + len(part) + 2 > self.max_chunk_chars
            if is_table:
                if current_parts and would_exceed:
                    flush()
                current_parts.append(part)
                current_ids.append(block.id)
                current_pages.update(block.meta.get("pages", [block.page]))
                current_has_table = True
                if len(part) >= self.max_chunk_chars or len("\n\n".join(current_parts)) >= self.max_chunk_chars:
                    flush()
                continue

            if would_exceed and current_parts:
                flush()
            current_parts.append(part)
            current_ids.append(block.id)
            current_pages.update(block.meta.get("pages", [block.page]))
            if len("\n\n".join(current_parts)) >= self.max_chunk_chars:
                flush()

        if current_parts:
            overlap_was_only = len(current_ids) == 0 and len(current_parts) == 1
            if not overlap_was_only:
                flush()
        return chunks

    def write_outputs(self, result: dict[str, Any], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._safe_stem(Path(result["source"]).stem)
        json_path = output_dir / f"{stem}.parsed.json"
        md_path = output_dir / f"{stem}.review.md"
        html_path = output_dir / f"{stem}.review.html"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._render_markdown(result), encoding="utf-8")
        html_path.write_text(self._render_html(result), encoding="utf-8")

    def _block_to_dict(self, block: ParsedBlock) -> dict[str, Any]:
        data = asdict(block)
        data["bbox"] = block.bbox.to_list()
        return data

    def _block_chunk_text(self, block: ParsedBlock) -> str:
        if block.type == "table":
            return f"[TABLE page={block.page} bbox={block.bbox.to_list()}]\n{block.text}"
        if block.type == "figure":
            return f"[FIGURE page={block.page} bbox={block.bbox.to_list()}]\n{block.text}"
        return f"[TEXT page={block.page} bbox={block.bbox.to_list()}]\n{block.text}"

    def _find_column_split(self, lines: list[TextLine], page_width: float) -> float | None:
        if len(lines) < 8:
            return None
        x0s = sorted(ln.bbox.x0 for ln in lines)
        x0_gaps = [(x0s[i + 1] - x0s[i], i) for i in range(len(x0s) - 1)]
        for max_x0_gap, x0_idx in sorted(x0_gaps, reverse=True):
            left_count = x0_idx + 1
            right_count = len(x0s) - left_count
            split = (x0s[x0_idx] + x0s[x0_idx + 1]) / 2
            if left_count >= 4 and right_count >= 4 and max_x0_gap >= page_width * 0.09 and page_width * 0.20 <= split <= page_width * 0.75:
                return split

        x0_spread = max(x0s) - min(x0s) if x0s else 0.0
        if x0_spread < page_width * 0.18:
            return None

        centers = sorted(ln.bbox.center_x for ln in lines)
        gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
        if not gaps:
            return None
        max_gap, idx = max(gaps)
        left_count = idx + 1
        right_count = len(centers) - left_count
        if left_count < 4 or right_count < 4:
            return None
        if max_gap < page_width * 0.16:
            return None
        split = (centers[idx] + centers[idx + 1]) / 2
        if split < page_width * 0.32 or split > page_width * 0.68:
            return None
        return split

    def _join_lines(self, lines: list[TextLine]) -> str:
        parts: list[str] = []
        for line in lines:
            if not parts:
                parts.append(line.text)
                continue
            prev = parts[-1]
            if self._should_join_without_space(prev, line.text):
                parts[-1] = prev.rstrip() + line.text.lstrip()
            else:
                parts.append(line.text)
        return self._clean_text(" ".join(parts))

    def _join_word_text(self, words: list[str]) -> str:
        text = ""
        for word in words:
            if not text:
                text = word
                continue
            if self._should_join_without_space(text, word):
                text += word
            else:
                text += " " + word
        return self._clean_text(text)

    def _should_join_without_space(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        if re.search(r"[\u4e00-\u9fff]$", left) and re.match(r"^[\u4e00-\u9fff]", right):
            return True
        if left.endswith("-") and re.match(r"^[A-Za-z]", right):
            return True
        return False

    def _should_break_paragraph(self, prev: TextLine, line: TextLine, line_height: float) -> bool:
        if self._looks_like_heading(line):
            return True
        if self._looks_like_heading(prev):
            return True
        if self._looks_like_caption(line) or self._looks_like_source(line):
            return True
        if self._looks_like_caption(prev) or self._looks_like_source(prev):
            return True
        if self._is_bullet_start(line.text) and not self._is_bullet_start(prev.text):
            return True
        indent_jump = line.bbox.x0 - prev.bbox.x0
        if indent_jump > line_height * 3.5 and self._ends_sentence(prev.text):
            return True
        return False

    def _tail_overlap(self, text: str) -> str:
        if self.chunk_overlap_chars <= 0:
            return ""
        tail = text[-self.chunk_overlap_chars :].strip()
        if not tail:
            return ""
        cut = max(tail.rfind("。"), tail.rfind("."), tail.rfind("\n"))
        if cut > len(tail) * 0.35:
            return tail[cut + 1 :].strip()
        return tail

    def _render_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            f"# PDF Parse Review",
            "",
            f"- Source: `{result['source']}`",
            f"- Pages: {result['summary']['pages']}",
            f"- Blocks: {result['summary']['blocks']}",
            f"- Tables: {result['summary']['tables']}",
            f"- Chunks: {result['summary']['chunks']}",
            "",
            "## Blocks",
            "",
        ]
        for block in result["blocks"]:
            lines.append(f"### {block['id']} | {block['type']} | page {block['page']} | bbox {block['bbox']}")
            lines.append("")
            lines.append(block["text"])
            lines.append("")
        lines.append("## Chunks")
        lines.append("")
        for chunk in result["chunks"]:
            lines.append(f"### {chunk['id']} | pages {chunk['pages']} | chars {chunk['char_count']} | table {chunk['contains_table']}")
            lines.append("")
            lines.append(chunk["text"])
            lines.append("")
        return "\n".join(lines)

    def _render_html(self, result: dict[str, Any]) -> str:
        block_html = []
        for block in result["blocks"]:
            cls = "table" if block["type"] == "table" else "text"
            block_html.append(
                f"<section class='block {cls}'>"
                f"<h3>{html.escape(block['id'])} · {html.escape(block['type'])} · page {block['page']}</h3>"
                f"<p class='meta'>bbox {html.escape(str(block['bbox']))}</p>"
                f"<pre>{html.escape(block['text'])}</pre>"
                "</section>"
            )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>PDF Parse Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.5; }}
    .summary {{ padding: 12px 0; border-bottom: 1px solid #ddd; margin-bottom: 16px; }}
    .block {{ border-left: 4px solid #999; padding: 8px 12px; margin: 14px 0; background: #fafafa; }}
    .block.table {{ border-left-color: #0a7; background: #f4fffb; }}
    .block h3 {{ margin: 0 0 4px; font-size: 16px; }}
    .meta {{ color: #666; margin: 0 0 8px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>PDF Parse Review</h1>
  <div class="summary">
    <div><strong>Source:</strong> {html.escape(result['source'])}</div>
    <div><strong>Pages:</strong> {result['summary']['pages']} · <strong>Blocks:</strong> {result['summary']['blocks']} · <strong>Tables:</strong> {result['summary']['tables']} · <strong>Chunks:</strong> {result['summary']['chunks']}</div>
  </div>
  {''.join(block_html)}
</body>
</html>
"""

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\u3000", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
        return text

    @staticmethod
    def _clean_table_rows(rows: Iterable[Iterable[Any]]) -> list[list[str]]:
        cleaned: list[list[str]] = []
        for row in rows:
            cells = [FinancialPDFParser._clean_text("" if cell is None else str(cell)) for cell in row]
            if any(cells):
                cleaned.append(cells)
        if not cleaned:
            return []
        max_cols = max(len(r) for r in cleaned)
        return [r + [""] * (max_cols - len(r)) for r in cleaned]

    @staticmethod
    def _is_valid_table_candidate(rows: list[list[str]], bbox: BBox, page_width: float, page_height: float) -> bool:
        if not rows:
            return False
        cols = max((len(row) for row in rows), default=0)
        non_empty = sum(1 for row in rows for cell in row if cell.strip())
        text_len = sum(len(cell.strip()) for row in rows for cell in row)
        if len(rows) < 2:
            return False
        if cols < 2 or non_empty < 4:
            return False
        max_cell_len = max((len(cell.strip()) for row in rows for cell in row), default=0)
        if bbox.width < max(90.0, page_width * 0.16):
            return False
        if bbox.height < 18:
            return False
        if bbox.height > page_height * 0.82 and bbox.width < page_width * 0.18:
            return False
        if bbox.width > page_width * 0.86 and bbox.height > page_height * 0.62 and (len(rows) < 8 or max_cell_len > 180):
            return False
        if max_cell_len > 420 and len(rows) < 12:
            return False
        if text_len < 16:
            return False
        return True

    @staticmethod
    def _table_to_markdown(rows: list[list[str]]) -> str:
        if not rows:
            return ""
        max_cols = max(len(row) for row in rows)

        def esc(cell: str) -> str:
            return cell.replace("|", "\\|").replace("\n", "<br>")

        padded = [row + [""] * (max_cols - len(row)) for row in rows]
        md = ["| " + " | ".join(esc(c) for c in padded[0]) + " |"]
        md.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
        for row in padded[1:]:
            md.append("| " + " | ".join(esc(c) for c in row) + " |")
        return "\n".join(md)

    @staticmethod
    def _header_footer_key(text: str) -> str:
        text = FinancialPDFParser._clean_text(text)
        text = re.sub(r"\d+", "#", text)
        text = re.sub(r"\s+", "", text)
        if len(text) < 4:
            return ""
        return text[:80]

    @staticmethod
    def _is_page_number(text: str) -> bool:
        t = text.strip()
        return bool(re.fullmatch(r"[-—_ ]*\d{1,4}[-—_ ]*", t) or re.fullmatch(r"第?\s*\d{1,4}\s*页", t))

    @staticmethod
    def _line_in_margin_noise(line: TextLine, page_size: dict[str, float]) -> bool:
        width = float(page_size["width"])
        height = float(page_size["height"])
        if line.bbox.width < 20 and line.bbox.x0 < width * 0.08 and len(line.text) <= 2:
            return True
        if line.bbox.top < height * 0.035 and len(line.text) <= 6:
            return True
        if line.bbox.bottom > height * 0.965 and len(line.text) <= 12:
            return True
        if line.bbox.x1 < width * 0.055 or line.bbox.x0 > width * 0.96:
            return True
        return False

    @staticmethod
    def _overlap_ratio(a: BBox, b: BBox) -> float:
        x0 = max(a.x0, b.x0)
        y0 = max(a.top, b.top)
        x1 = min(a.x1, b.x1)
        y1 = min(a.bottom, b.bottom)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area = max(1.0, a.width * a.height)
        return inter / area

    @staticmethod
    def _point_in_bbox(x: float, y: float, bbox: BBox) -> bool:
        return bbox.x0 <= x <= bbox.x1 and bbox.top <= y <= bbox.bottom

    @staticmethod
    def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
        inter = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        return inter / max(1.0, min(a.width, b.width))

    @staticmethod
    def _looks_like_heading(line: TextLine) -> bool:
        text = line.text.strip()
        if len(text) <= 42 and re.match(r"^([一二三四五六七八九十]+[、.．]|[0-9]+[.．、]|第.+[章节])", text):
            return True
        if len(text) <= 28 and not re.search(r"[。；;:：,.，、]", text) and line.font_size >= 16:
            return True
        return False

    @staticmethod
    def _looks_like_caption(line: TextLine) -> bool:
        text = line.text.strip()
        return bool(re.match(r"^[\uf06c•●◆◼▪-]?\s*(图表|图|表)(\s*\d+)?[:：]", text))

    @staticmethod
    def _looks_like_figure_caption(line: TextLine) -> bool:
        text = line.text.strip()
        return bool(re.match(r"^[\uf06c•●◆◼▪-]?\s*图(\s*\d+)?[:：]", text))

    @staticmethod
    def _looks_like_source(line: TextLine) -> bool:
        text = line.text.strip()
        return bool(re.match(r"^(资料来源|数据来源|来源)[:：]", text))

    @staticmethod
    def _looks_like_uncaptioned_table_title(line: TextLine) -> bool:
        text = line.text.strip()
        return bool(re.search(r"(公司基本情况|主要财务指标|财务摘要|盈利预测与估值)", text))

    @staticmethod
    def _is_bullet_start(text: str) -> bool:
        return bool(re.match(r"^[\uf06c•●◆◼▪-]\s*", text.strip()))

    @staticmethod
    def _ends_sentence(text: str) -> bool:
        text = text.strip()
        return bool(text and text[-1] in "。！？!?")

    @staticmethod
    def _safe_stem(stem: str) -> str:
        stem = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", stem)
        return stem[:120] or "parsed_pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse financial research PDFs with layout-aware heuristics.")
    parser.add_argument("--input", required=True, action="append", help="PDF file or directory path. Can be absolute. Repeat --input for multiple paths.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "legacy_parse"), help="Directory for JSON/Markdown/HTML review outputs.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of PDFs parsed. 0 means no limit.")
    parser.add_argument("--max-chars", type=int, default=1800, help="Target chunk size in characters.")
    args = parser.parse_args()

    pdfs: list[Path] = []
    for item in args.input:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            pdfs.extend(sorted(path.rglob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            pdfs.append(path)
    if args.max_files:
        pdfs = pdfs[: args.max_files]

    out = Path(args.output_dir)
    parser_impl = FinancialPDFParser(max_chunk_chars=args.max_chars)
    index = []
    for pdf in pdfs:
        result = parser_impl.parse(pdf, output_dir=out)
        index.append({"source": str(pdf), **result["summary"]})
        print(f"parsed {pdf} -> blocks={result['summary']['blocks']} tables={result['summary']['tables']} chunks={result['summary']['chunks']}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

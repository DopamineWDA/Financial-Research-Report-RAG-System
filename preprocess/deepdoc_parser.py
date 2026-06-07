"""DeepDoc-backed PDF parser for financial research reports.

This adapter keeps the RAG project's existing parse artifact shape while using
RAGFlow DeepDoc for OCR, layout recognition, table structure recognition, and
position-aware parsing.
"""

from __future__ import annotations

import argparse
import base64
import enum
import html
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RAGFLOW_ROOT = PROJECT / "ragflow"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(RAGFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(RAGFLOW_ROOT))

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = StrEnum


@dataclass
class ParsedBlock:
    id: str
    type: str
    page: int
    bbox: list[float]
    text: str
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
    return DeepDocFinancialPDFParser(**kwargs).parse(pdf_path, output_dir=output_dir)


class DeepDocFinancialPDFParser:
    def __init__(
        self,
        max_chunk_chars: int = 1800,
        chunk_overlap_chars: int = 160,
        zoomin: int = 3,
        from_page: int = 0,
        to_page: int | None = None,
        render_pages: bool = True,
        write_review: bool = True,
        write_media: bool = True,
        vision_model_path: str | None = None,
        vision_endpoint: str | None = None,
        vision_model_name: str | None = None,
        vision_api_key: str | None = None,
        vision_device: str | None = None,
        vision_max_new_tokens: int = 384,
        vision_context_chars: int = 1200,
        vision_workers: int = 2,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.zoomin = zoomin
        self.from_page = from_page
        self.to_page = to_page
        self.render_pages = render_pages
        self.write_review = write_review
        self.write_media = write_media
        self.vision_model_path = vision_model_path
        self.vision_endpoint = vision_endpoint
        self.vision_model_name = vision_model_name
        self.vision_api_key = vision_api_key
        self.vision_device = vision_device
        self.vision_max_new_tokens = vision_max_new_tokens
        self.vision_context_chars = vision_context_chars
        self.vision_workers = max(1, int(vision_workers))

    def parse(self, pdf_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
        pdf_path = Path(pdf_path)
        try:
            from common.constants import MAXIMUM_PAGE_NUMBER
            from deepdoc.parser.pdf_parser import RAGFlowPdfParser
        except Exception as exc:
            raise RuntimeError(
                "DeepDoc parser dependencies are not ready. Install ragflow dependencies "
                f"from ragflow/pyproject.toml. Original error: {type(exc).__name__}: {exc}"
            ) from exc

        stem = self._safe_stem(pdf_path.stem)
        parser = RAGFlowPdfParser()
        to_page = self.to_page if self.to_page is not None else MAXIMUM_PAGE_NUMBER
        raw_boxes = parser.parse_into_bboxes(
            str(pdf_path),
            zoomin=self.zoomin,
            from_page=self.from_page,
            to_page=to_page,
        )
        page_sizes = self._page_sizes(pdf_path)
        raw_blocks = self._boxes_to_blocks(raw_boxes)
        blocks = self._merge_layout_blocks(raw_blocks)
        vision_figure_data = self._build_vision_figure_data(blocks)
        if output_dir and self.write_media:
            self._write_media(blocks, Path(output_dir), stem)
        else:
            self._drop_inline_images(blocks)

        if self.vision_model_path or self.vision_endpoint:
            self._vision_enhance_media_blocks(blocks, vision_figure_data)

        chunks = self._make_chunks(blocks)
        page_images = self._render_page_images(pdf_path, self.from_page, self.to_page) if self.render_pages else {}

        result = {
            "source": str(pdf_path),
            "method": "RAGFlow DeepDoc parse_into_bboxes (OCR + layout + TSR)",
            "summary": {
                "pages": len(page_sizes),
                "parsed_pages": len({block.page for block in blocks}),
                "raw_boxes": len(raw_boxes),
                "raw_blocks": len(raw_blocks),
                "blocks": len(blocks),
                "tables": sum(1 for block in blocks if block.type == "table"),
                "figures": sum(1 for block in blocks if block.type == "figure"),
                "vision_enhanced": sum(1 for block in blocks if block.meta.get("vision_description")),
                "chunks": len(chunks),
            },
            "page_sizes": page_sizes,
            "blocks": [asdict(block) for block in blocks],
            "chunks": [asdict(chunk) for chunk in chunks],
            "_visual": {"page_images": page_images},
        }

        if output_dir:
            self.write_outputs(result, Path(output_dir))
        return result

    def write_outputs(self, result: dict[str, Any], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = self._safe_stem(Path(result["source"]).stem)
        json_result = {k: v for k, v in result.items() if k != "_visual"}
        (output_dir / f"{stem}.parsed.json").write_text(json.dumps(json_result, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.write_review:
            (output_dir / f"{stem}.review.md").write_text(self._render_markdown(json_result), encoding="utf-8")
            (output_dir / f"{stem}.review.html").write_text(self._render_html(result), encoding="utf-8")

    def _boxes_to_blocks(self, boxes: list[dict[str, Any]]) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        for idx, box in enumerate(sorted(boxes, key=lambda b: (int(b.get("page_number", 1)), float(b.get("top", 0)), float(b.get("x0", 0))))):
            text = self._clean_text(self._remove_position_tags(str(box.get("text", ""))))
            layout_type = str(box.get("layout_type") or "text").lower()
            block_type = self._normalize_type(layout_type)
            if not text and block_type not in {"table", "figure"}:
                continue
            page = int(box.get("page_number") or 1)
            bbox = self._box_to_bbox(box, page)
            meta = {
                "layout_type": layout_type,
                "layoutno": box.get("layoutno"),
                "positions": box.get("positions") or [],
                "position_tag": box.get("position_tag"),
                "col_id": box.get("col_id"),
            }
            if block_type in {"table", "figure"} and box.get("image") is not None:
                meta["_image"] = box.get("image")
            blocks.append(
                ParsedBlock(
                    id=f"b{len(blocks) + 1:04d}",
                    type=block_type,
                    page=page,
                    bbox=bbox,
                    text=text,
                    meta={k: v for k, v in meta.items() if v not in (None, [], "")},
                )
            )
        return blocks

    def _write_media(self, blocks: list[ParsedBlock], output_dir: Path, stem: str) -> None:
        media_dir = output_dir / f"{stem}.media"
        media_index = []
        for block in blocks:
            image = block.meta.pop("_image", None)
            if image is None:
                continue
            media_dir.mkdir(parents=True, exist_ok=True)
            suffix = "png"
            filename = f"{block.id}_{block.type}_p{block.page}.{suffix}"
            path = media_dir / filename
            try:
                image.save(path)
            except Exception:
                image.convert("RGB").save(path)
            rel_path = f"{media_dir.name}/{filename}"
            block.meta["image_path"] = rel_path
            media_index.append(
                {
                    "block_id": block.id,
                    "type": block.type,
                    "page": block.page,
                    "bbox": block.bbox,
                    "path": rel_path,
                }
            )
        if media_index:
            media_dir.mkdir(parents=True, exist_ok=True)
            (media_dir / "index.json").write_text(json.dumps(media_index, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _drop_inline_images(blocks: list[ParsedBlock]) -> None:
        for block in blocks:
            block.meta.pop("_image", None)

    def _vision_enhance_media_blocks(
        self,
        blocks: list[ParsedBlock],
        figure_data: list[tuple[str, tuple[tuple[Any, list[str]], list[tuple[int, float, float, float, float]]]]],
    ) -> None:
        """Enhance figure blocks using ragflow's native VisionFigureParser structure."""

        if not figure_data:
            return

        try:
            from deepdoc.parser.figure_parser import VisionFigureParser
        except Exception as exc:
            logging.warning("vision enhancement disabled: %s: %s", type(exc).__name__, exc)
            return

        try:
            if self.vision_endpoint:
                vision_model = _OpenAICompatibleVisionModel(
                    endpoint=self.vision_endpoint,
                    model_name=self.vision_model_name,
                    api_key=self.vision_api_key,
                    max_new_tokens=self.vision_max_new_tokens,
                )
            else:
                vision_model = _LocalQwen3VLVisionModel(
                    model_path=self.vision_model_path,
                    device=self.vision_device,
                    max_new_tokens=self.vision_max_new_tokens,
                )
        except Exception as exc:
            logging.warning("vision enhancement disabled: %s: %s", type(exc).__name__, exc)
            return

        parser_input = [item for _, item in figure_data]
        figure_blocks = [block for block in blocks if block.type == "figure"]
        context_map = self._figure_contexts(
            blocks,
            figure_blocks,
            max_chars=self.vision_context_chars,
        )
        figure_contexts = [context_map.get(block_id, ("", "")) for block_id, _ in figure_data]
        try:
            parser = VisionFigureParser(
                vision_model=vision_model,
                figures_data=parser_input,
                figure_contexts=figure_contexts,
                context_size=self.vision_context_chars,
            )
            boosted = parser(callback=lambda _prog, _msg: None)
        except Exception as exc:
            logging.warning("vision enhancement disabled: %s: %s", type(exc).__name__, exc)
            return

        boosted_by_id: dict[str, str] = {}
        for (block_id, _), assembled in zip(figure_data, boosted):
            if not assembled or not isinstance(assembled[0], tuple) or len(assembled[0]) != 2:
                continue
            desc_list = assembled[0][1]
            if isinstance(desc_list, list):
                desc = "\n".join(str(item) for item in desc_list if str(item).strip()).strip()
            else:
                desc = str(desc_list).strip()
            if self._is_meaningful_vision_text(desc):
                boosted_by_id[block_id] = desc

        for block in blocks:
            desc = boosted_by_id.get(block.id, "").strip()
            if not desc:
                continue
            block.meta["vision_description"] = desc
            if block.text:
                block.text = f"{block.text}\n\n[Vision Description]\n{desc}"
            else:
                block.text = f"[Vision Description]\n{desc}"

    def _build_vision_figure_data(
        self,
        blocks: list[ParsedBlock],
    ) -> list[tuple[str, tuple[tuple[Any, list[str]], list[tuple[int, float, float, float, float]]]]]:
        """Build ragflow-style figures_data for VisionFigureParser.

        Shape matches ragflow/deepdoc/parser/figure_parser.py:
        [((image, [description]), positions), ...]
        """

        figure_data: list[tuple[str, tuple[tuple[Any, list[str]], list[tuple[int, float, float, float, float]]]]] = []
        for block in blocks:
            if block.type != "figure":
                continue
            image = block.meta.get("_image")
            if image is None:
                continue
            raw_positions = block.meta.get("positions") or []
            positions: list[tuple[int, float, float, float, float]] = []
            for item in raw_positions:
                if isinstance(item, (list, tuple)) and len(item) >= 5:
                    positions.append(
                        (
                            int(item[0]),
                            float(item[1]),
                            float(item[2]),
                            float(item[3]),
                            float(item[4]),
                        )
                    )
            if not positions:
                positions = [(int(block.page), float(block.bbox[0]), float(block.bbox[2]), float(block.bbox[1]), float(block.bbox[3]))]
            figure_data.append((block.id, ((image, [""]), positions)))
        return figure_data

    @staticmethod
    def _load_ragflow_prompt(filename: str) -> str:
        path = RAGFLOW_ROOT / "rag" / "prompts" / filename
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _figure_contexts(
        self,
        all_blocks: list[ParsedBlock],
        targets: list[ParsedBlock],
        *,
        max_chars: int,
    ) -> dict[str, tuple[str, str]]:
        """Build minimal text context around each figure/table block.

        We approximate ragflow's context logic using nearby text/title blocks on the same page:
        - above: closest blocks whose bbox bottom <= target top
        - below: closest blocks whose bbox top >= target bottom
        """

        max_chars = max(0, int(max_chars))
        if max_chars <= 0:
            return {block.id: ("", "") for block in targets}

        by_page: dict[int, list[ParsedBlock]] = {}
        for block in all_blocks:
            if block.type not in {"text", "title"}:
                continue
            by_page.setdefault(block.page, []).append(block)

        out: dict[str, tuple[str, str]] = {}
        for target in targets:
            page_blocks = sorted(by_page.get(target.page, []), key=lambda b: (b.bbox[1], b.bbox[0]))
            top = float(target.bbox[1])
            bottom = float(target.bbox[3])

            above = [b for b in page_blocks if float(b.bbox[3]) <= top]
            below = [b for b in page_blocks if float(b.bbox[1]) >= bottom]

            above_sorted = sorted(above, key=lambda b: float(b.bbox[3]), reverse=True)
            below_sorted = sorted(below, key=lambda b: float(b.bbox[1]))

            above_text = self._take_text_budget([b.text for b in above_sorted], max_chars=max_chars)
            below_text = self._take_text_budget([b.text for b in below_sorted], max_chars=max_chars)
            out[target.id] = (above_text, below_text)
        return out

    @staticmethod
    def _take_text_budget(texts: list[str], *, max_chars: int) -> str:
        buf: list[str] = []
        used = 0
        for text in texts:
            text = text.strip()
            if not text:
                continue
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining]
            buf.append(text)
            used += len(text)
            if used >= max_chars:
                break
        return "\n".join(buf).strip()

    @staticmethod
    def _is_meaningful_vision_text(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        # Filter out degenerate VLM outputs such as "-", "*", ":" or similar
        # markdown punctuation without any semantic content.
        condensed = re.sub(r"[\s`*_#>\-:;,.!?\[\]\(\){}|\\/+=~]+", "", text)
        return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", condensed))

    def _merge_layout_blocks(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        """Merge DeepDoc line boxes that belong to the same layout region.

        DeepDoc keeps precise OCR/text boxes after layout and vertical merging.
        For RAG review and chunking we want paragraph-like blocks, so we group
        text/title boxes by page + type + layoutno while keeping table/figure
        blocks atomic.
        """
        groups: dict[tuple[int, str, str], list[ParsedBlock]] = {}
        singles: list[ParsedBlock] = []
        for block in blocks:
            layoutno = str(block.meta.get("layoutno", ""))
            if block.type in {"text", "title"} and layoutno:
                groups.setdefault((block.page, block.type, layoutno), []).append(block)
            else:
                singles.append(block)

        merged: list[ParsedBlock] = singles[:]
        for (page, block_type, layoutno), items in groups.items():
            items = sorted(items, key=lambda b: (b.bbox[1], b.bbox[0]))
            bbox = self._union_bboxes([b.bbox for b in items])
            meta = {
                "layout_type": items[0].meta.get("layout_type", block_type),
                "layoutno": layoutno,
                "source_block_ids": [b.id for b in items],
            }
            positions = []
            position_tags = []
            col_ids = []
            for item in items:
                positions.extend(item.meta.get("positions", []))
                if item.meta.get("position_tag"):
                    position_tags.append(item.meta["position_tag"])
                if item.meta.get("col_id") is not None:
                    col_ids.append(item.meta["col_id"])
            if positions:
                meta["positions"] = positions
            if position_tags:
                meta["position_tag"] = "".join(position_tags)
            if col_ids:
                meta["col_ids"] = sorted(set(col_ids))
            merged.append(
                ParsedBlock(
                    id=items[0].id,
                    type=block_type,
                    page=page,
                    bbox=bbox,
                    text=self._join_block_texts([b.text for b in items]),
                    meta=meta,
                )
            )

        merged = sorted(merged, key=lambda b: (b.page, b.bbox[1], b.bbox[0]))
        for idx, block in enumerate(merged, start=1):
            block.id = f"b{idx:04d}"
        return merged

    @staticmethod
    def _union_bboxes(bboxes: list[list[float]]) -> list[float]:
        return [
            round(min(b[0] for b in bboxes), 2),
            round(min(b[1] for b in bboxes), 2),
            round(max(b[2] for b in bboxes), 2),
            round(max(b[3] for b in bboxes), 2),
        ]

    def _join_block_texts(self, texts: list[str]) -> str:
        out = ""
        for text in texts:
            text = text.strip()
            if not text:
                continue
            if not out:
                out = text
            elif self._should_join_without_space(out, text):
                out += text
            else:
                out += " " + text
        return self._clean_text(out)

    def _box_to_bbox(self, box: dict[str, Any], page: int) -> list[float]:
        positions = box.get("positions")
        if isinstance(positions, list) and positions:
            first = positions[0]
            if isinstance(first, list) and len(first) >= 5:
                return [round(float(first[1]), 2), round(float(first[3]), 2), round(float(first[2]), 2), round(float(first[4]), 2)]

        x0 = float(box.get("x0", 0))
        x1 = float(box.get("x1", x0))
        top = float(box.get("top", 0))
        bottom = float(box.get("bottom", top))
        return [round(x0, 2), round(top, 2), round(x1, 2), round(bottom, 2)]

    def _make_chunks(self, blocks: list[ParsedBlock]) -> list[ParsedChunk]:
        chunks: list[ParsedChunk] = []
        current: list[ParsedBlock] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if not current:
                return
            text = "\n\n".join(self._block_chunk_text(block) for block in current)
            chunks.append(
                ParsedChunk(
                    id=f"c{len(chunks) + 1:04d}",
                    block_ids=[block.id for block in current],
                    pages=sorted({block.page for block in current}),
                    text=text,
                    char_count=len(text),
                    contains_table=any(block.type == "table" for block in current),
                )
            )
            overlap = self._tail_overlap(text)
            current = []
            current_len = 0
            if overlap:
                current.append(ParsedBlock(id=f"overlap_{len(chunks):04d}", type="text", page=chunks[-1].pages[-1], bbox=[0, 0, 0, 0], text=overlap))
                current_len = len(overlap)

        for block in blocks:
            block_len = len(block.text)
            if block.type == "table":
                flush()
                current = [block]
                current_len = block_len
                flush()
                continue
            if current and current_len + block_len > self.max_chunk_chars:
                flush()
            current.append(block)
            current_len += block_len
        flush()
        return chunks

    def _render_markdown(self, result: dict[str, Any]) -> str:
        lines = [
            "# DeepDoc PDF Parse Review",
            "",
            f"- Source: `{result['source']}`",
            f"- Method: {result['method']}",
            f"- Pages: {result['summary']['pages']}",
            f"- Blocks: {result['summary']['blocks']}",
            f"- Tables: {result['summary']['tables']}",
            f"- Figures: {result['summary']['figures']}",
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
        return "\n".join(lines)

    def _render_html(self, result: dict[str, Any]) -> str:
        page_images = result.get("_visual", {}).get("page_images", {})
        page_sizes = {int(p["page"]): p for p in result.get("page_sizes", [])}
        overlays_by_page: dict[int, list[str]] = {}
        for block in result["blocks"]:
            page = int(block["page"])
            page_size = page_sizes.get(page) or {"width": 1, "height": 1}
            width = max(float(page_size["width"]), 1.0)
            height = max(float(page_size["height"]), 1.0)
            x0, top, x1, bottom = [float(v) for v in block["bbox"]]
            style = (
                f"left:{x0 / width * 100:.4f}%;top:{top / height * 100:.4f}%;"
                f"width:{max(x1 - x0, 1) / width * 100:.4f}%;height:{max(bottom - top, 1) / height * 100:.4f}%;"
            )
            label = html.escape(f"{block['id']} {block['type']}")
            overlays_by_page.setdefault(page, []).append(f"<a class='box {block['type']}' href='#{block['id']}' style='{style}' title='{label}'><span>{label}</span></a>")

        pages_html = []
        pages_to_show = sorted(set(overlays_by_page) | {int(page) for page in page_images if str(page).isdigit()})
        if not pages_to_show:
            pages_to_show = sorted(page_sizes)
        for page in pages_to_show:
            img_src = page_images.get(str(page), "")
            media = f"<img src='{img_src}' alt='page {page}'>" if img_src else "<div class='missing'>page image disabled</div>"
            pages_html.append(f"<section class='page'><h2>Page {page}</h2><div class='canvas'>{media}{''.join(overlays_by_page.get(page, []))}</div></section>")

        block_html = []
        for block in result["blocks"]:
            image_path = block.get("meta", {}).get("image_path")
            image_html = f"<p class='media'><a href='{html.escape(image_path)}'>media crop</a></p>" if image_path else ""
            block_html.append(
                f"<section class='block {block['type']}' id='{html.escape(block['id'])}'>"
                f"<h3>{html.escape(block['id'])} · {html.escape(block['type'])} · page {block['page']}</h3>"
                f"<p class='meta'>bbox {html.escape(str(block['bbox']))} · layout {html.escape(str(block.get('meta', {}).get('layout_type', '')))}</p>"
                f"{image_html}"
                f"<pre>{html.escape(block['text'])}</pre>"
                "</section>"
            )

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>DeepDoc PDF Parse Review</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202124; background: #f6f7f9; }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 12px 18px; background: #fff; border-bottom: 1px solid #d8dde6; }}
    header h1 {{ margin: 0 0 4px; font-size: 18px; }}
    header .meta {{ color: #5f6673; font-size: 13px; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 58vw) minmax(320px, 1fr); gap: 16px; padding: 16px; align-items: start; }}
    .pages, .blocks {{ display: grid; gap: 16px; }}
    .page, .block {{ background: #fff; border: 1px solid #d8dde6; border-radius: 6px; overflow: hidden; }}
    .page h2, .block h3 {{ margin: 0; padding: 10px 12px; font-size: 14px; border-bottom: 1px solid #e6e9ef; }}
    .canvas {{ position: relative; background: #e9edf3; line-height: 0; }}
    .canvas img {{ width: 100%; height: auto; display: block; }}
    .missing {{ min-height: 420px; display: grid; place-items: center; color: #687386; }}
    .box {{ position: absolute; box-sizing: border-box; border: 2px solid #2563eb; background: rgba(37, 99, 235, 0.08); line-height: 1; text-decoration: none; }}
    .box span {{ position: absolute; left: 0; top: -18px; padding: 2px 4px; font-size: 11px; color: #fff; background: #2563eb; white-space: nowrap; }}
    .box.table {{ border-color: #0f8b5f; background: rgba(15, 139, 95, 0.10); }}
    .box.table span {{ background: #0f8b5f; }}
    .box.figure {{ border-color: #c2410c; background: rgba(194, 65, 12, 0.10); }}
    .box.figure span {{ background: #c2410c; }}
    .block {{ border-left: 4px solid #8892a6; }}
    .block.table {{ border-left-color: #0f8b5f; }}
    .block.figure {{ border-left-color: #c2410c; }}
    .block .meta {{ padding: 8px 12px 0; margin: 0; color: #687386; font-size: 12px; }}
    .block .media {{ padding: 4px 12px 0; margin: 0; font-size: 12px; }}
    .block .media a {{ color: #155eef; }}
    pre {{ margin: 0; padding: 10px 12px 14px; white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.5; }}
    @media (max-width: 980px) {{ main {{ grid-template-columns: 1fr; }} header {{ position: static; }} }}
  </style>
</head>
<body>
  <header>
    <h1>DeepDoc PDF Parse Review</h1>
    <div class="meta">{html.escape(result['source'])} · pages {result['summary']['pages']} · blocks {result['summary']['blocks']} · tables {result['summary']['tables']} · figures {result['summary']['figures']} · chunks {result['summary']['chunks']}</div>
  </header>
  <main>
    <div class="pages">{''.join(pages_html)}</div>
    <div class="blocks">{''.join(block_html)}</div>
  </main>
</body>
</html>
"""

    def _render_page_images(self, pdf_path: Path, from_page: int = 0, to_page: int | None = None) -> dict[str, str]:
        images: dict[str, str] = {}
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            start = max(0, from_page)
            end = min(len(pdf.pages), to_page if to_page is not None else len(pdf.pages))
            for idx in range(start, end):
                page = pdf.pages[idx]
                try:
                    img = page.to_image(resolution=108, antialias=True).original.convert("RGB")
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    images[str(idx + 1)] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                except Exception:
                    images[str(idx + 1)] = ""
        return images

    def _page_sizes(self, pdf_path: Path) -> list[dict[str, float]]:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            return [{"page": i + 1, "width": round(float(page.width), 2), "height": round(float(page.height), 2)} for i, page in enumerate(pdf.pages)]

    def _block_chunk_text(self, block: ParsedBlock) -> str:
        tag = block.type.upper()
        return f"[{tag} page={block.page} bbox={block.bbox}]\n{block.text}"

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

    @staticmethod
    def _normalize_type(layout_type: str) -> str:
        if "table" in layout_type:
            return "table"
        if "figure" in layout_type or "image" in layout_type:
            return "figure"
        if "title" in layout_type:
            return "title"
        return "text"

    @staticmethod
    def _remove_position_tags(text: str) -> str:
        return re.sub(r"@@[\t0-9.-]+?##", "", text)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\u3000", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
        return text.strip()

    @staticmethod
    def _should_join_without_space(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if re.search(r"[\u4e00-\u9fff]$", left) and re.match(r"^[\u4e00-\u9fff]", right):
            return True
        if re.search(r"[A-Za-z0-9]$", left) and re.match(r"^[A-Za-z0-9%]", right):
            return False
        return False

    @staticmethod
    def _safe_stem(stem: str) -> str:
        stem = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", stem)
        return stem[:120] or "parsed_pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse financial research PDFs with RAGFlow DeepDoc.")
    parser.add_argument("--input", required=True, action="append", help="PDF file or directory path. Can be absolute. Repeat --input for multiple paths.")
    parser.add_argument("--output-dir", default="/home/txs/work/zyp/RAG/outputs/deepdoc/", help="Directory for JSON/Markdown/HTML review outputs.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of PDFs parsed. 0 means no limit.")
    parser.add_argument("--max-chars", type=int, default=1800, help="Target chunk size in characters.")
    parser.add_argument("--zoomin", type=int, default=3, help="DeepDoc render/OCR zoom factor.")
    parser.add_argument("--from-page", type=int, default=0, help="DeepDoc start page, 0-based.")
    parser.add_argument("--to-page", type=int, default=None, help="DeepDoc end page, 0-based exclusive.")
    parser.add_argument("--no-page-images", action="store_true", help="Skip embedding rendered page images into review HTML.")
    parser.add_argument("--json-only", action="store_true", help="Only write *.parsed.json; skip review Markdown/HTML.")
    parser.add_argument("--preserve-root", type=Path, default=None, help="Mirror input paths relative to this root under --output-dir.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs whose target *.parsed.json already exists.")
    parser.add_argument("--vision-model-path", default="/home/txs/work/zyp/LLM/Qwen2.5-VL-3B-Instruct", help="Local HuggingFace model dir for Qwen3-VL (enables figure/table vision enhancement).")
    parser.add_argument("--vision-endpoint", default='http://127.0.0.1:9997', help="OpenAI-compatible vision endpoint, e.g. http://127.0.0.1:9997 or http://127.0.0.1:9997/v1/chat/completions.")
    parser.add_argument("--vision-model-name", default='qwen2.5-vl-3b', help="Model id exposed by the vision service, e.g. qwen3-vl-2b.")
    parser.add_argument("--vision-api-key", default=None, help="Optional API key for the vision service.")
    parser.add_argument("--vision-device", default="cuda:1", help="Torch device for vision model, e.g. cuda, cpu, cuda:0. Default: auto.")
    parser.add_argument("--vision-max-new-tokens", type=int, default=1024, help="Max tokens for each vision description generation.")
    parser.add_argument("--vision-context-chars", type=int, default=1200, help="Max chars of surrounding text to include as context above/below.")
    parser.add_argument("--vision-workers", type=int, default=2, help="Concurrent vision workers.")
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
    parser_impl = DeepDocFinancialPDFParser(
        max_chunk_chars=args.max_chars,
        zoomin=args.zoomin,
        from_page=args.from_page,
        to_page=args.to_page,
        render_pages=not args.no_page_images,
        write_review=not args.json_only,
        write_media=not args.json_only,
        vision_model_path=args.vision_model_path,
        vision_endpoint=args.vision_endpoint,
        vision_model_name=args.vision_model_name,
        vision_api_key=args.vision_api_key,
        vision_device=args.vision_device,
        vision_max_new_tokens=args.vision_max_new_tokens,
        vision_context_chars=args.vision_context_chars,
        vision_workers=args.vision_workers,
    )

    index = []
    for pdf in pdfs:
        pdf_out = out
        if args.preserve_root:
            root = args.preserve_root.expanduser().resolve()
            try:
                rel_parent = pdf.resolve().relative_to(root).parent
                pdf_out = out / rel_parent
            except ValueError:
                pdf_out = out

        target = pdf_out / f"{parser_impl._safe_stem(pdf.stem)}.parsed.json"
        if args.skip_existing and target.exists():
            index.append({"source": str(pdf), "method": "deepdoc", "skipped": True, "target": str(target)})
            print(f"skip existing {pdf} -> {target}")
            continue

        result = parser_impl.parse(pdf, output_dir=pdf_out)
        index.append({"source": str(pdf), "method": "deepdoc", **result["summary"]})
        print(
            f"parsed {pdf} -> method=deepdoc blocks={result['summary']['blocks']} "
            f"tables={result['summary']['tables']} chunks={result['summary']['chunks']}"
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


class _LocalQwen3VLVisionModel:
    def __init__(self, model_path: str | None, *, device: str | None, max_new_tokens: int) -> None:
        if not model_path:
            raise ValueError("model_path is required")
        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self._load()

    def _load(self) -> None:
        import torch

        try:
            import transformers  # type: ignore
        except Exception as exc:
            raise RuntimeError("Missing dependency: transformers") from exc

        AutoProcessor = getattr(transformers, "AutoProcessor", None)
        if AutoProcessor is None:
            raise RuntimeError("transformers.AutoProcessor not available")

        model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_cls is None:
            raise RuntimeError("transformers does not expose Qwen3VLForConditionalGeneration/AutoModelForVision2Seq")

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = model_cls.from_pretrained(self.model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model.eval()

        if self.device:
            self.model.to(self.device)
        else:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def describe_with_prompt(self, image_bytes: bytes, prompt: str) -> str:
        import torch
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        apply_template = getattr(self.processor, "apply_chat_template", None)
        if apply_template is None:
            raise RuntimeError("processor.apply_chat_template not available for this model")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        if not decoded:
            return ""
        out = decoded[0]
        if text and out.startswith(text):
            out = out[len(text) :]
        return out.strip()


class _OpenAICompatibleVisionModel:
    def __init__(
        self,
        endpoint: str | None,
        *,
        model_name: str | None,
        api_key: str | None,
        max_new_tokens: int,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        if not model_name:
            raise ValueError("model_name is required when using a vision endpoint")
        normalized = endpoint.rstrip("/")
        if normalized.endswith("/v1/chat/completions"):
            pass
        elif normalized.endswith("/v1"):
            normalized = normalized + "/chat/completions"
        else:
            normalized = normalized + "/v1/chat/completions"
        self.endpoint = normalized
        self.model_name = model_name
        self.api_key = api_key
        self.max_new_tokens = int(max_new_tokens)

    def describe_with_prompt(self, image_bytes: bytes, prompt: str) -> str:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        mime = self._guess_mime_type(image_bytes)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.1,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib_request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Connection failed: {exc}") from exc

        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(part for part in parts if part)
        return str(content).strip()

    @staticmethod
    def _guess_mime_type(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
            return "image/webp"
        return "application/octet-stream"


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9._%+-]+")


def clean_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


@dataclass
class RetrievalHit:
    docstore_id: str
    doc_id: str
    chunk_id: str
    chunk_type: str
    text: str
    rerank_text: str
    metadata: Dict
    source_file: str
    source_pdf: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    title_score: float = 0.0
    fused_score: float = 0.0
    reranker_score: float = 0.0
    route: str = ""
    linked_raw_table: Optional[Dict] = None


@dataclass
class QueryDecomposePlan:
    query_type: str
    subqueries: List[str]
    branch_candidate_k: int
    branch_output_k: int
    final_top_k: int


class ThreeWayRetriever:
    def __init__(
        self,
        index_dir: Path,
        *,
        reranker_model_path: Optional[Union[Path, str]] = None,
        reranker_use_fp16: bool = False,
    ):
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.build_meta = self._read_json(self.index_dir / "build_meta.json")
        self.id_map = self._read_json(self.index_dir / "id_map.json")
        self.docstore_records = self._load_docstore(self.index_dir / "docstore.jsonl")
        self.record_by_docstore_id = {record["docstore_id"]: record for record in self.docstore_records}
        self.indexed_records = self._load_indexed_records()
        self.raw_table_map = self._build_raw_table_map()
        self.doc_title_map = self._build_doc_title_map()
        self.doc_company_map = self._build_doc_company_map()
        self._faiss_index = None
        self._bm25 = None
        self._bm25_corpus_tokens: List[List[str]] = []
        self._title_bm25 = None
        self._title_doc_ids: List[str] = []
        self._title_corpus_tokens: List[List[str]] = []
        self.reranker_model_path = self._resolve_model_path_if_needed(reranker_model_path) if reranker_model_path else None
        self.reranker_use_fp16 = reranker_use_fp16
        self._reranker = None
        self._reranker_fallback = None
        self._query_encoder_backend = None
        self._query_encoder = None
        self._query_encoder_normalized = None
        self._query_encoder_max_length = None
        self._company_names = self._build_company_names()

    @staticmethod
    def _preferred_single_device() -> str:
        try:
            import torch
        except Exception:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def retrieve_vector(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        faiss = self._load_faiss()
        query_vector = self._embed_query(query)
        distances, indices = self._faiss_index.search(query_vector, top_k)
        hits: List[RetrievalHit] = []
        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id < 0:
                continue
            record = self.indexed_records[int(faiss_id)]
            hit = self._make_hit(record, route="vector")
            hit.vector_score = float(score)
            hit.fused_score = float(score)
            hits.append(hit)
        return hits

    def retrieve_bm25(self, query: str, top_k: int = 5, *, title_weight: float = 1.0) -> List[RetrievalHit]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        scores, title_score_by_doc = self._score_bm25(query_tokens, title_weight=title_weight)
        top_indices = self._top_indices(scores, top_k)
        hits: List[RetrievalHit] = []
        for corpus_idx in top_indices:
            score = float(scores[corpus_idx])
            if score <= 0:
                continue
            record = self.indexed_records[int(corpus_idx)]
            hit = self._make_hit(record, route="bm25")
            hit.bm25_score = score
            hit.title_score = float(title_score_by_doc.get(record["doc_id"], 0.0))
            hit.fused_score = score
            hits.append(hit)
        return hits

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        *,
        vector_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        title_weight: float = 1.0,
        fusion_mode: str = "weighted_sum",
        rrf_k: int = 60,
    ) -> List[RetrievalHit]:
        vector_hits = self.retrieve_vector(query, top_k=max(top_k, vector_candidate_k))
        bm25_hits = self.retrieve_bm25(query, top_k=max(top_k, bm25_candidate_k), title_weight=title_weight)
        vector_hits = vector_hits[:vector_candidate_k]
        bm25_hits = bm25_hits[:bm25_candidate_k]
        merged = self._merge_hits(vector_hits, bm25_hits)

        if fusion_mode == "weighted_sum":
            vector_norm = self._normalize_score_map({hit.docstore_id: hit.vector_score for hit in vector_hits})
            bm25_norm = self._normalize_score_map({hit.docstore_id: hit.bm25_score for hit in bm25_hits})
            ranked_hits: List[RetrievalHit] = []
            for docstore_id, hit in merged.items():
                hit.route = "hybrid"
                hit.fused_score = (
                    vector_weight * vector_norm.get(docstore_id, 0.0)
                    + bm25_weight * bm25_norm.get(docstore_id, 0.0)
                )
                ranked_hits.append(hit)
        elif fusion_mode == "rrf":
            ranked_hits = self._apply_rrf_fusion(
                merged,
                vector_hits=vector_hits,
                bm25_hits=bm25_hits,
                rrf_k=rrf_k,
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        ranked_hits.sort(key=lambda item: item.fused_score, reverse=True)
        return ranked_hits[:top_k]

    def retrieve_with_query_decompose(
        self,
        query: str,
        route: str = "hybrid",
        top_k: int = 10,
        *,
        vector_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        title_weight: float = 1.0,
        fusion_mode: str = "weighted_sum",
        rrf_k: int = 60,
        use_reranker: bool = False,
        rerank_batch_size: int = 16,
    ) -> Optional[List[RetrievalHit]]:
        plan = self._build_query_decompose_plan(query=query, final_top_k=top_k)
        if plan is None:
            return None

        branch_ranked_hits: List[List[RetrievalHit]] = []
        branch_selected_hits: List[List[RetrievalHit]] = []

        for subquery in plan.subqueries:
            hits = self.retrieve(
                query=subquery,
                route=route,
                top_k=plan.branch_candidate_k,
                vector_candidate_k=plan.branch_candidate_k,
                bm25_candidate_k=plan.branch_candidate_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                title_weight=title_weight,
                fusion_mode=fusion_mode,
                rrf_k=rrf_k,
                query_decompose=False,
            )
            hits = self._limit_hits_per_doc(hits, max_hits_per_doc=2)
            if use_reranker:
                hits = self.rerank(subquery, hits, top_k=None, batch_size=rerank_batch_size)
            branch_ranked_hits.append(hits)
            branch_selected_hits.append(hits[:plan.branch_output_k])

        if plan.query_type == "compare":
            merged_hits = [
                hit
                for branch_hits in branch_selected_hits
                for hit in branch_hits
            ]
        else:
            merged_hits = []
            leftovers: List[RetrievalHit] = []
            for branch_hits, branch_full_hits in zip(branch_selected_hits, branch_ranked_hits):
                merged_hits.extend(branch_hits[:3])
                leftovers.extend(branch_hits[3:])
                leftovers.extend(branch_full_hits[plan.branch_output_k:])
            merged_hits.extend(self._take_best_unique_hits(leftovers, limit=1, existing_hits=merged_hits))

        merged_hits = self._dedupe_hits_by_chunk(merged_hits)
        merged_hits.sort(key=self._hit_sort_key, reverse=True)
        return merged_hits[:plan.final_top_k]

    def get_query_decompose_plan(self, query: str, *, final_top_k: int = 10) -> Optional[dict[str, object]]:
        plan = self._build_query_decompose_plan(query=query, final_top_k=final_top_k)
        if plan is None:
            return None
        return {
            "query_type": plan.query_type,
            "subqueries": list(plan.subqueries),
            "branch_candidate_k": plan.branch_candidate_k,
            "branch_output_k": plan.branch_output_k,
            "final_top_k": plan.final_top_k,
        }

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: Optional[int] = None,
        batch_size: int = 16,
    ) -> List[RetrievalHit]:
        if not hits:
            return []

        reranker = self._load_reranker()
        pairs = [(query, hit.rerank_text) for hit in hits]

        try:
            scores = reranker.compute_score(pairs, batch_size=batch_size)
        except TypeError:
            scores = reranker.compute_score(pairs)
        except AttributeError as exc:
            if "prepare_for_model" not in str(exc):
                raise
            scores = self._rerank_with_transformers_fallback(query, hits, batch_size=batch_size)

        score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
        reranked_hits: List[RetrievalHit] = []
        for hit, score in zip(hits, score_values):
            hit.reranker_score = float(score)
            reranked_hits.append(hit)

        reranked_hits.sort(key=lambda item: item.reranker_score, reverse=True)
        if top_k is not None:
            return reranked_hits[:top_k]
        return reranked_hits

    def retrieve_two_stage(
        self,
        query: str,
        *,
        route: str,
        first_stage_top_k: int = 50,
        rerank_top_k: int = 10,
        vector_candidate_k: int = 50,
        bm25_candidate_k: int = 50,
        vector_weight: float = 0.3,
        bm25_weight: float = 0.7,
        title_weight: float = 1.0,
        fusion_mode: str = "weighted_sum",
        rrf_k: int = 60,
        rerank_batch_size: int = 16,
        query_decompose: bool = False,
    ) -> List[RetrievalHit]:
        if query_decompose:
            decomposed_hits = self.retrieve_with_query_decompose(
                query=query,
                route=route,
                top_k=rerank_top_k,
                vector_candidate_k=vector_candidate_k,
                bm25_candidate_k=bm25_candidate_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                title_weight=title_weight,
                fusion_mode=fusion_mode,
                rrf_k=rrf_k,
                use_reranker=True,
                rerank_batch_size=rerank_batch_size,
            )
            if decomposed_hits is not None:
                return decomposed_hits
        first_stage_hits = self.retrieve(
            query=query,
            route=route,
            top_k=first_stage_top_k,
            vector_candidate_k=vector_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            title_weight=title_weight,
            fusion_mode=fusion_mode,
            rrf_k=rrf_k,
            query_decompose=False,
        )
        return self.rerank(
            query=query,
            hits=first_stage_hits,
            top_k=rerank_top_k,
            batch_size=rerank_batch_size,
        )

    def warm_up(
        self,
        *,
        load_vector: bool = True,
        load_bm25: bool = True,
        load_reranker: bool = True,
        run_dummy_embed: bool = True,
        run_dummy_rerank: bool = True,
    ) -> None:
        if load_vector:
            self._load_faiss()
            self._load_query_encoder()
            if run_dummy_embed:
                self._embed_query("测试")
        if load_bm25:
            self._load_bm25()
            self._load_title_bm25()
        if load_reranker and self.reranker_model_path is not None:
            self._load_reranker()
            if run_dummy_rerank and self.indexed_records:
                dummy_hit = self._make_hit(self.indexed_records[0], route="warmup")
                self.rerank("测试", [dummy_hit], top_k=1, batch_size=1)

    def retrieve(
        self,
        query: str,
        route: str = "hybrid",
        top_k: int = 5,
        *,
        vector_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        title_weight: float = 1.0,
        fusion_mode: str = "weighted_sum",
        rrf_k: int = 60,
        query_decompose: bool = False,
    ) -> List[RetrievalHit]:
        if route == "hybrid_weightsum":
            route = "hybrid"
            fusion_mode = "weighted_sum"
        elif route == "hybrid_rrf":
            route = "hybrid"
            fusion_mode = "rrf"
        if query_decompose and route == "hybrid":
            decomposed_hits = self.retrieve_with_query_decompose(
                query=query,
                route=route,
                top_k=top_k,
                vector_candidate_k=vector_candidate_k,
                bm25_candidate_k=bm25_candidate_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                title_weight=title_weight,
                fusion_mode=fusion_mode,
                rrf_k=rrf_k,
                use_reranker=False,
            )
            if decomposed_hits is not None:
                return decomposed_hits
        if route == "vector":
            return self.retrieve_vector(query, top_k=top_k)
        if route == "bm25":
            return self.retrieve_bm25(query, top_k=top_k, title_weight=title_weight)
        if route == "hybrid":
            return self.retrieve_hybrid(
                query,
                top_k=top_k,
                vector_candidate_k=vector_candidate_k,
                bm25_candidate_k=bm25_candidate_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                title_weight=title_weight,
                fusion_mode=fusion_mode,
                rrf_k=rrf_k,
            )
        raise ValueError(f"Unsupported route: {route}")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        try:
            import jieba
        except ImportError as exc:
            raise RuntimeError(
                "jieba is not installed. Please run `pip install jieba` before using BM25 retrieval."
            ) from exc

        tokens: List[str] = []
        for raw_token in _TOKEN_PATTERN.findall((text or "").lower()):
            if re.fullmatch(r"[\u4e00-\u9fff]+", raw_token):
                tokens.extend(piece.strip() for piece in jieba.lcut(raw_token) if piece.strip())
            else:
                tokens.append(raw_token)
        return tokens

    def _load_faiss(self):
        if self._faiss_index is not None:
            return self._faiss_module
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss is not installed. Please install faiss-cpu or faiss-gpu before using vector retrieval."
            ) from exc
        self._faiss_module = faiss
        self._faiss_index = faiss.read_index(str(self.index_dir / "index.faiss"))
        return faiss

    def _load_reranker(self):
        if self._reranker is not None:
            return self._reranker
        if self.reranker_model_path is None:
            raise RuntimeError(
                "Reranker model path is not configured. "
                "Pass reranker_model_path to ThreeWayRetriever or use the CLI --rerank-model-path option."
            )
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is not installed. Please run `pip install FlagEmbedding` before using reranking."
            ) from exc
        preferred_device = self._preferred_single_device()
        self._reranker = FlagReranker(
            str(self.reranker_model_path),
            use_fp16=self.reranker_use_fp16,
            devices=preferred_device,
        )
        return self._reranker

    def _load_transformers_reranker_fallback(self):
        if self._reranker_fallback is not None:
            return self._reranker_fallback
        if self.reranker_model_path is None:
            raise RuntimeError("Reranker model path is not configured.")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for reranker fallback. "
                "Please ensure they are installed in the runtime environment."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(str(self.reranker_model_path), trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(str(self.reranker_model_path), trust_remote_code=False)

        if torch.cuda.is_available():
            device = "cuda"
            if self.reranker_use_fp16:
                model = model.half()
        else:
            device = "cpu"

        model.to(device)
        model.eval()
        self._reranker_fallback = {
            "torch": torch,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }
        return self._reranker_fallback

    def _rerank_with_transformers_fallback(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        batch_size: int,
    ) -> List[float]:
        bundle = self._load_transformers_reranker_fallback()
        torch = bundle["torch"]
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]
        max_length = int(getattr(model.config, "max_position_embeddings", 512) or 512)

        scores: List[float] = []
        passages = [hit.rerank_text for hit in hits]

        with torch.no_grad():
            for start in range(0, len(passages), batch_size):
                batch_passages = passages[start:start + batch_size]
                batch_queries = [query] * len(batch_passages)
                inputs = tokenizer(
                    batch_queries,
                    batch_passages,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(logits.cpu().numpy().tolist())
        return scores

    def _load_bm25(self):
        if self._bm25 is not None:
            return self._bm25
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank_bm25 is not installed. Please run `pip install rank_bm25` before using BM25 retrieval."
            ) from exc
        self._bm25_corpus_tokens = [self.tokenize(self._bm25_body_text(record)) for record in self.indexed_records]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens)
        return self._bm25

    def _load_title_bm25(self):
        if self._title_bm25 is not None:
            return self._title_bm25
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank_bm25 is not installed. Please run `pip install rank_bm25` before using BM25 retrieval."
            ) from exc

        self._title_doc_ids = []
        self._title_corpus_tokens = []
        for doc_id, title in self.doc_title_map.items():
            combined = " ".join(
                part for part in [clean_value(title), clean_value(self.doc_company_map.get(doc_id, ""))] if part
            )
            tokens = self.tokenize(combined)
            if not tokens:
                continue
            self._title_doc_ids.append(doc_id)
            self._title_corpus_tokens.append(tokens)

        self._title_bm25 = BM25Okapi(self._title_corpus_tokens) if self._title_corpus_tokens else False
        return self._title_bm25

    def _score_bm25(self, query_tokens: List[str], *, title_weight: float) -> Tuple[np.ndarray, Dict[str, float]]:
        bm25 = self._load_bm25()
        body_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
        title_score_by_doc = self._score_titles(query_tokens)
        if title_weight == 0.0 or not title_score_by_doc:
            return body_scores, title_score_by_doc

        combined_scores = body_scores.copy()
        for idx, record in enumerate(self.indexed_records):
            combined_scores[idx] += title_weight * float(title_score_by_doc.get(record["doc_id"], 0.0))
        return combined_scores, title_score_by_doc

    def _score_titles(self, query_tokens: List[str]) -> Dict[str, float]:
        title_bm25 = self._load_title_bm25()
        if not title_bm25:
            return {}
        scores = np.asarray(title_bm25.get_scores(query_tokens), dtype=np.float32)
        return {
            doc_id: float(score)
            for doc_id, score in zip(self._title_doc_ids, scores)
            if float(score) > 0.0
        }

    def _merge_hits(
        self,
        vector_hits: Sequence[RetrievalHit],
        bm25_hits: Sequence[RetrievalHit],
    ) -> Dict[str, RetrievalHit]:
        merged: Dict[str, RetrievalHit] = {}
        for hit in vector_hits:
            merged[hit.docstore_id] = hit
        for hit in bm25_hits:
            existing = merged.get(hit.docstore_id)
            if existing is None:
                merged[hit.docstore_id] = hit
            else:
                existing.bm25_score = hit.bm25_score
                existing.title_score = hit.title_score
                existing.route = "hybrid"
        return merged

    def _apply_rrf_fusion(
        self,
        merged_hits: Dict[str, RetrievalHit],
        *,
        vector_hits: Sequence[RetrievalHit],
        bm25_hits: Sequence[RetrievalHit],
        rrf_k: int,
    ) -> List[RetrievalHit]:
        ranked_hits: List[RetrievalHit] = []
        vector_ranks = {hit.docstore_id: rank for rank, hit in enumerate(vector_hits, start=1)}
        bm25_ranks = {hit.docstore_id: rank for rank, hit in enumerate(bm25_hits, start=1)}

        for docstore_id, hit in merged_hits.items():
            hit.route = "hybrid"
            hit.fused_score = 0.0
            vector_rank = vector_ranks.get(docstore_id)
            bm25_rank = bm25_ranks.get(docstore_id)
            if vector_rank is not None:
                hit.fused_score += 1.0 / (rrf_k + vector_rank)
            if bm25_rank is not None:
                hit.fused_score += 1.0 / (rrf_k + bm25_rank)
            ranked_hits.append(hit)
        return ranked_hits

    def _embed_query(self, query: str) -> np.ndarray:
        backend, model = self._load_query_encoder()
        query_texts = [query]

        if backend == "flagembedding":
            try:
                vectors = model.encode(query_texts, batch_size=1, max_length=self._query_encoder_max_length)
            except TypeError:
                vectors = model.encode(query_texts, batch_size=1)
            query_vector = np.asarray(vectors, dtype=np.float32)
        elif backend == "sentence_transformers":
            query_vector = np.asarray(
                model.encode(
                    query_texts,
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
        else:
            raise RuntimeError(f"Unsupported query encoder backend: {backend}")

        if self._query_encoder_normalized:
            norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            query_vector = query_vector / norms
        return query_vector

    def _load_query_encoder(self):
        if self._query_encoder is not None:
            return self._query_encoder_backend, self._query_encoder

        model_path = self.build_meta["model_path"]
        self._query_encoder_max_length = int(self.build_meta.get("max_length", 512))
        self._query_encoder_normalized = bool(self.build_meta.get("normalize_embeddings", True))
        errors: List[str] = []

        try:
            from FlagEmbedding import FlagModel

            self._query_encoder = FlagModel(
                model_path,
                use_fp16=False,
                devices=self._preferred_single_device(),
            )
            self._query_encoder_backend = "flagembedding"
            return self._query_encoder_backend, self._query_encoder
        except Exception as exc:
            errors.append(f"FlagEmbedding failed: {exc!r}")

        try:
            from sentence_transformers import SentenceTransformer

            self._query_encoder = SentenceTransformer(
                model_path,
                trust_remote_code=False,
                device=self._preferred_single_device(),
            )
            self._query_encoder_backend = "sentence_transformers"
            return self._query_encoder_backend, self._query_encoder
        except Exception as exc:
            errors.append(f"sentence-transformers failed: {exc!r}")
            raise RuntimeError("Unable to load query encoder.\n" + "\n".join(errors)) from exc

    def _load_docstore(self, docstore_path: Path) -> List[Dict]:
        records: List[Dict] = []
        with docstore_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        return records

    def _load_indexed_records(self) -> List[Dict]:
        indexed = [record for record in self.docstore_records if record.get("indexed") and record.get("faiss_id") is not None]
        indexed.sort(key=lambda item: int(item["faiss_id"]))
        expected = len(self.id_map)
        if expected != len(indexed):
            raise RuntimeError(
                f"Indexed record count mismatch: id_map has {expected}, docstore has {len(indexed)} indexed records."
            )
        return indexed

    def _build_raw_table_map(self) -> Dict[Tuple[str, str], Dict]:
        raw_table_map: Dict[Tuple[str, str], Dict] = {}
        for record in self.docstore_records:
            metadata = record.get("metadata", {})
            table_id = metadata.get("table_id")
            if record.get("chunk_type") == "raw_table_chunk" and table_id:
                raw_table_map[(record["doc_id"], table_id)] = record
        return raw_table_map

    def _build_doc_title_map(self) -> Dict[str, str]:
        title_map: Dict[str, str] = {}
        for record in self.docstore_records:
            doc_id = record["doc_id"]
            existing = title_map.get(doc_id)
            if existing:
                continue

            title = self._extract_title_from_record(record)
            if title:
                title_map[doc_id] = title
        return title_map

    def _build_doc_company_map(self) -> Dict[str, str]:
        company_map: Dict[str, str] = {}
        for record in self.docstore_records:
            doc_id = record["doc_id"]
            if company_map.get(doc_id):
                continue

            company = self._extract_company_from_record(record)
            if company:
                company_map[doc_id] = company
        return company_map

    def _build_company_names(self) -> List[str]:
        names = {
            company.strip()
            for company in self.doc_company_map.values()
            if company and company.strip()
        }
        excluded_tokens = ("证券", "研究院", "大学", "行业", "周报", "点评", "策略", "计算机", "电子", "公用事业", "互联网")
        for record in self.docstore_records:
            for part in str(record.get("doc_id", "")).split("_"):
                cleaned = re.sub(r"\d+", "", clean_value(part))
                if len(cleaned) < 2:
                    continue
                if not re.search(r"[\u4e00-\u9fff]", cleaned):
                    continue
                if any(token in cleaned for token in excluded_tokens):
                    continue
                names.add(cleaned)
        return sorted(names, key=len, reverse=True)

    def _build_query_decompose_plan(self, *, query: str, final_top_k: int) -> Optional[QueryDecomposePlan]:
        normalized = self._normalize_query_text(query)
        compare_cues = ("谁更高", "谁更低", "谁更好", "谁更强", "谁更优", "哪个更高", "哪个更低", "哪个更好", "哪个更强", "哪个更优")
        summary_cues = ("归纳总结", "总结一下", "归纳一下", "总结", "归纳")

        entities = self._extract_query_companies(normalized)
        if len(entities) == 2 and any(cue in normalized for cue in compare_cues):
            aspect = normalized
            for entity in entities:
                aspect = aspect.replace(entity, " ")
            for cue in compare_cues + ("以及", "相比", "对比", "比较", "和", "与", "及", "、"):
                aspect = aspect.replace(cue, " ")
            aspect = self._compact_whitespace(aspect)
            aspect = re.sub(r"^[的\s]+", "", aspect)
            aspect = aspect.rstrip("。；;，,")
            subqueries = [self._compact_whitespace(f"{entity} {aspect}") for entity in entities]
            return QueryDecomposePlan(
                query_type="compare",
                subqueries=subqueries,
                branch_candidate_k=25,
                branch_output_k=5,
                final_top_k=final_top_k,
            )

        if len(entities) == 3 and any(cue in normalized for cue in summary_cues):
            aspect = normalized
            for entity in entities:
                aspect = aspect.replace(entity, " ")
            for cue in summary_cues + ("以及", "一下", "和", "与", "及", "、"):
                aspect = aspect.replace(cue, " ")
            aspect = self._compact_whitespace(aspect)
            aspect = re.sub(r"^[的\s]+", "", aspect)
            aspect = aspect.rstrip("。；;，,")
            subqueries = [self._compact_whitespace(f"{entity} {aspect}") for entity in entities]
            return QueryDecomposePlan(
                query_type="summary",
                subqueries=subqueries,
                branch_candidate_k=15,
                branch_output_k=4,
                final_top_k=final_top_k,
            )
        return None

    def _make_hit(self, record: Dict, route: str) -> RetrievalHit:
        hit = RetrievalHit(
            docstore_id=record["docstore_id"],
            doc_id=record["doc_id"],
            chunk_id=record["chunk_id"],
            chunk_type=record["chunk_type"],
            text=record["text"],
            rerank_text=record["embedding_text"],
            metadata=record["metadata"],
            source_file=record["source_file"],
            source_pdf=record["source_pdf"],
            route=route,
        )
        if record["chunk_type"] == "table_summary_chunk":
            table_id = record.get("metadata", {}).get("table_id")
            if table_id:
                linked = self.raw_table_map.get((record["doc_id"], table_id))
                if linked is not None:
                    hit.linked_raw_table = linked
        return hit

    @staticmethod
    def _extract_title_from_record(record: Dict) -> str:
        if record.get("chunk_type") == "cover_summary_chunk":
            first_line = ThreeWayRetriever._first_nonempty_line(record.get("text", ""))
            if first_line.startswith("标题："):
                return first_line.removeprefix("标题：").strip()

        first_line = ThreeWayRetriever._first_nonempty_line(record.get("embedding_text", ""))
        if first_line.startswith("标题："):
            return first_line.removeprefix("标题：").strip()

        return ThreeWayRetriever._fallback_title_from_doc_id(record.get("doc_id", ""))

    @staticmethod
    def _extract_company_from_record(record: Dict) -> str:
        if record.get("chunk_type") != "cover_summary_chunk":
            return ""
        for line in (record.get("text", "") or "").splitlines():
            line = line.strip()
            if line.startswith("公司："):
                return line.removeprefix("公司：").strip()
        return ""

    @staticmethod
    def _bm25_body_text(record: Dict) -> str:
        chunk_type = record.get("chunk_type")
        if chunk_type != "raw_table_chunk":
            return clean_value(record.get("text"))

        metadata = record.get("metadata", {}) or {}
        parts: List[str] = []
        context = clean_value(metadata.get("bm25_context"))
        caption = clean_value(metadata.get("caption"))
        text = clean_value(record.get("text"))
        if context:
            parts.append(f"前后文：{context}")
        if caption:
            parts.append(f"表名：{caption}")
        if text:
            parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _first_nonempty_line(text: str) -> str:
        for line in (text or "").splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    @staticmethod
    def _fallback_title_from_doc_id(doc_id: str) -> str:
        raw = (doc_id or "").strip()
        if not raw:
            return ""
        raw = re.sub(r"^\d+_\d{4}-\d{2}-\d{2}_", "", raw)
        return raw.replace("_", " ").strip()

    @staticmethod
    def _normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        values = np.asarray(list(score_map.values()), dtype=np.float32)
        max_value = float(values.max())
        min_value = float(values.min())
        if abs(max_value - min_value) < 1e-12:
            if max_value <= 0:
                return {key: 0.0 for key in score_map}
            return {key: 1.0 for key in score_map}
        return {key: (value - min_value) / (max_value - min_value) for key, value in score_map.items()}

    @staticmethod
    def _normalize_query_text(text: str) -> str:
        normalized = clean_value(text)
        normalized = normalized.replace("？", " ").replace("?", " ")
        normalized = normalized.replace("，", " ").replace(",", " ")
        normalized = normalized.replace("：", " ").replace(":", " ")
        return ThreeWayRetriever._compact_whitespace(normalized)

    @staticmethod
    def _compact_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", clean_value(text)).strip()

    def _extract_query_companies(self, query: str) -> List[str]:
        matches: List[Tuple[int, int, str]] = []
        used_spans: List[Tuple[int, int]] = []
        for company in self._company_names:
            start = query.find(company)
            if start < 0:
                continue
            end = start + len(company)
            if any(not (end <= span_start or start >= span_end) for span_start, span_end in used_spans):
                continue
            matches.append((start, end, company))
            used_spans.append((start, end))
        matches.sort(key=lambda item: item[0])
        return [company for _, _, company in matches]

    @staticmethod
    def _limit_hits_per_doc(hits: Sequence[RetrievalHit], *, max_hits_per_doc: int) -> List[RetrievalHit]:
        limited: List[RetrievalHit] = []
        per_doc_counter: Dict[str, int] = {}
        for hit in hits:
            count = per_doc_counter.get(hit.doc_id, 0)
            if count >= max_hits_per_doc:
                continue
            limited.append(hit)
            per_doc_counter[hit.doc_id] = count + 1
        return limited

    @staticmethod
    def _dedupe_hits_by_chunk(hits: Sequence[RetrievalHit]) -> List[RetrievalHit]:
        deduped: List[RetrievalHit] = []
        seen_chunk_ids = set()
        for hit in hits:
            if hit.chunk_id in seen_chunk_ids:
                continue
            deduped.append(hit)
            seen_chunk_ids.add(hit.chunk_id)
        return deduped

    def _take_best_unique_hits(
        self,
        hits: Sequence[RetrievalHit],
        *,
        limit: int,
        existing_hits: Sequence[RetrievalHit],
    ) -> List[RetrievalHit]:
        existing_chunk_ids = {hit.chunk_id for hit in existing_hits}
        unique_hits: List[RetrievalHit] = []
        for hit in sorted(hits, key=self._hit_sort_key, reverse=True):
            if hit.chunk_id in existing_chunk_ids:
                continue
            unique_hits.append(hit)
            existing_chunk_ids.add(hit.chunk_id)
            if len(unique_hits) >= limit:
                break
        return unique_hits

    @staticmethod
    def _hit_sort_key(hit: RetrievalHit) -> Tuple[float, float]:
        return (float(hit.reranker_score), float(hit.fused_score))

    @staticmethod
    def _top_indices(scores: np.ndarray, top_k: int) -> List[int]:
        if scores.size == 0 or top_k <= 0:
            return []
        top_k = min(top_k, scores.size)
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        return [int(idx) for idx in sorted_indices]

    @staticmethod
    def _read_json(path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _resolve_model_path_if_needed(model_path: Union[Path, str]) -> Path:
        path = Path(model_path).expanduser()
        if path.exists():
            return path.resolve()

        raw = str(model_path).strip()
        if not raw:
            raise RuntimeError("Model path is empty.")

        if "/" not in raw:
            raise RuntimeError(f"Model path does not exist: {model_path}")

        org, name = raw.split("/", 1)
        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_root = hub_root / f"models--{org}--{name}"
        if not model_root.exists():
            raise RuntimeError(f"Model path does not exist and cached hub directory was not found: {model_path}")

        snapshots_dir = model_root / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
            if snapshots:
                return snapshots[-1].resolve()

        refs_main = model_root / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_path = snapshots_dir / revision
            if snapshot_path.exists():
                return snapshot_path.resolve()

        raise RuntimeError(f"Unable to resolve a local snapshot for model: {model_path}")

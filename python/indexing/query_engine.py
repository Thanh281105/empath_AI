"""
Query Engine — Hybrid Retrieval + Reranking.

Tái sử dụng retrieval/hybrid_search.py + retrieval/reranker.py.
Cung cấp interface đơn giản cho LangGraph retrieve node.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console

from config import TOP_K_RETRIEVAL, TOP_K_RERANK
from retrieval.qdrant_client import QdrantWrapper
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from agents.model_registry import get_embed_model

from utils.console import console

# Singleton
_qdrant: QdrantWrapper | None = None




def _get_qdrant() -> QdrantWrapper:
    """Lazy load Qdrant connection."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantWrapper()
    return _qdrant


def retrieve(
    query_en: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Hybrid search (Dense + Sparse + RRF) trên Qdrant.

    Args:
        query_en: Query đã dịch sang tiếng Anh
        top_k: Số kết quả trả về

    Returns:
        List[dict] — mỗi dict chứa text, doc_title, authors, year, arxiv_id, rrf_score
    """
    model = get_embed_model()
    qdrant = _get_qdrant()

    # Embed query
    query_vector = model.encode(query_en, normalize_embeddings=True)

    # Hybrid search
    results = hybrid_search(
        query_vector=np.array(query_vector),
        query_text=query_en,
        qdrant=qdrant,
        top_k=top_k,
    )

    return results


def retrieve_and_rerank(
    query_en: str,
    top_k_search: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
) -> list[dict]:
    """
    Hybrid Search → Cross-encoder Reranking.

    Pipeline:
    1. Hybrid search (Dense + Sparse + RRF) → top_k_search candidates
    2. Cross-encoder reranking (BGE-Reranker-v2-M3) → top_k_rerank final

    Args:
        query_en: Query tiếng Anh
        top_k_search: Candidates từ hybrid search
        top_k_rerank: Kết quả cuối sau reranking

    Returns:
        Top-K reranked documents
    """
    # Step 1: Hybrid search
    candidates = retrieve(query_en, top_k=top_k_search)

    if not candidates:
        console.print("[yellow]⚠️ No results from hybrid search[/]")
        return []

    # Step 2: Rerank
    reranked = rerank(
        query=query_en,
        documents=candidates,
        top_k=top_k_rerank,
    )

    return reranked


def format_evidence(documents: list[dict]) -> str:
    """
    Format retrieved policy documents thanh context cho LLM.
    """
    if not documents:
        return ""

    parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("doc_title", "Unknown Policy")
        text = doc.get("text", "")
        score = doc.get("rerank_score", doc.get("rrf_score", 0))

        parts.append(
            f"[Chinh sach {i}] {title}\n"
            f"Do phu hop: {score:.4f}\n"
            f"{text}\n"
        )

    return "\n---\n".join(parts)


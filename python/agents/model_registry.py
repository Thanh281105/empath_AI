"""
Shared Model Registry — Singleton cho tất cả AI models.

Tránh load model nhiều lần (BGE-M3 ~2.3GB, Reranker ~1GB).
Trên P1000 4GB VRAM, load 2 lần = tràn bộ nhớ = chết.
"""
from rich.console import Console

from utils.console import console

_embed_model = None
_reranker_model = None


def get_embed_model():
    """Singleton embedding model — shared giữa router, query_engine, indexer."""
    global _embed_model
    if _embed_model is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL

        console.print(f"[cyan]🔄 Loading embedding model: {EMBEDDING_MODEL}...[/]")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        console.print(
            f"[green]✅ Embedding model ready "
            f"({_embed_model.get_sentence_embedding_dimension()}D)[/]"
        )
    return _embed_model


def get_reranker_model():
    """Singleton reranker model (CrossEncoder)."""
    global _reranker_model
    if _reranker_model is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from sentence_transformers import CrossEncoder
        from config import RERANKER_MODEL

        console.print(f"[cyan]🔄 Loading reranker model: {RERANKER_MODEL}...[/]")
        _reranker_model = CrossEncoder(RERANKER_MODEL, max_length=512)
        console.print("[green]✅ Reranker model ready[/]")
    return _reranker_model


def warmup():
    """Pre-load tất cả models lúc startup thay vì lúc query đầu tiên."""
    console.print("[bold cyan]🔥 Warming up models...[/]")
    get_embed_model()
    get_reranker_model()
    console.print("[bold green]✅ All models ready![/]")

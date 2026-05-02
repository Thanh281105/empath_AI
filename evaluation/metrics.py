"""
Metric computation cho EmpathAI evaluation.

- BLEU       : sacrebleu, char-level tokenizer (phù hợp tiếng Việt)
- ROUGE-L    : F1 Longest Common Subsequence
- BERTScore  : semantic similarity (multilingual-e5 / xlm-roberta)
- Recall@5   : tỷ lệ câu có relevant doc xuất hiện trong top-5 retrieved
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """BLEU corpus-level (0-100). Dùng char tokenizer cho tiếng Việt."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize="char")
    result = bleu.corpus_score(hypotheses, [references])
    return round(result.score, 2)


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """ROUGE-L F1 trung bình (0-100)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    return round(float(np.mean(scores)) * 100, 2)


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str = "vi",
    model_type: Optional[str] = None,
) -> float:
    """BERTScore F1 trung bình (0-100). Dùng bert-base-multilingual-cased cho tiếng Việt để tránh lỗi Tokenizer."""
    from bert_score import score as bs

    kwargs: dict = {"lang": lang, "verbose": False}
    if model_type:
        kwargs["model_type"] = model_type
    else:
        kwargs["model_type"] = "bert-base-multilingual-cased"
        
    # Xử lý chuỗi rỗng (thường do API timeout sinh ra [✗]) để tránh crash BERTScore
    safe_hypotheses = [h if h and h.strip() else "trống" for h in hypotheses]

    _, _, F1 = bs(safe_hypotheses, references, **kwargs)
    return round(float(F1.mean()) * 100, 2)


def compute_recall_at_k(
    retrieved_ids: List[List[str]],
    relevant_ids: List[str],
    k: int = 5,
) -> float:
    """
    Recall@k — chỉ dùng cho Req 3 và Req 4 (có RAG).

    Args:
        retrieved_ids : list độ dài N, mỗi phần tử là list policy_id của top-k docs
        relevant_ids  : list độ dài N, ground truth policy_id cho mỗi câu hỏi
        k             : cutoff

    Returns:
        Recall@k (0-100)
    """
    hits = sum(
        1
        for retrieved, relevant in zip(retrieved_ids, relevant_ids)
        if relevant in retrieved[:k]
    )
    return round(hits / max(len(relevant_ids), 1) * 100, 2)


def compute_all(
    hypotheses: List[str],
    references: List[str],
    retrieved_ids: Optional[List[List[str]]] = None,
    relevant_ids: Optional[List[str]] = None,
) -> dict:
    """Tính toàn bộ metric và trả về dict."""
    results: dict = {
        "BLEU": compute_bleu(hypotheses, references),
        "ROUGE-L": compute_rouge_l(hypotheses, references),
    }
    try:
        results["BERTScore"] = compute_bertscore(hypotheses, references)
    except Exception as e:
        results["BERTScore"] = "N/A"
    if retrieved_ids is not None and relevant_ids is not None:
        results["Recall@5"] = compute_recall_at_k(retrieved_ids, relevant_ids, k=5)
    else:
        results["Recall@5"] = None
    return results

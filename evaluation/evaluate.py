"""
EmpathAI Evaluation — So sánh 4 kiến trúc trên 50 câu hỏi CSKH.

Architectures:
  Req 1 : Groq llama-3.1-8b-instant  (no RAG)
  Req 2 : Vertex AI fine-tuned Llama 3.1-8B (no RAG)
  Req 3 : Groq llama-3.1-8b-instant  + RAG
  Req 4 : Vertex AI fine-tuned Llama 3.1-8B + RAG

Metrics:
  BLEU, ROUGE-L, BERTScore — generation quality
  Recall@5                  — retrieval quality (Req 3, 4 only)

Usage:
  pip install -r evaluation/requirements.txt
  python evaluation/evaluate.py [--arch all|1|2|3|4] [--limit N]
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR     = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
PYTHON_DIR   = PROJECT_ROOT / "python"
RESULTS_DIR  = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PYTHON_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from metrics import compute_all  # noqa: E402  (after sys.path setup)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS  = [k.strip() for k in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if k.strip()]
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL     = "llama-3.1-8b-instant"

VERTEX_PROJECT = os.getenv("VERTEX_PROJECT_ID", "")
VERTEX_REGION  = os.getenv("VERTEX_REGION", "asia-southeast1")

TOP_K_SEARCH   = int(os.getenv("TOP_K_RETRIEVAL", "8"))
TOP_K_RERANK   = int(os.getenv("TOP_K_RERANK",    "5"))

SYSTEM_PROMPT = (
    "Bạn là EmpathAI - trợ lý CSKH của MyKingdom (chuỗi cửa hàng đồ chơi trẻ em).\n"
    "Trả lời ngắn gọn, thân thiện và chính xác bằng tiếng Việt.\n"
    "Nếu không có thông tin, hướng khách liên hệ hotline 1900 1208 hoặc hotro@mykingdom.com.vn."
)

# ── Round-robin Groq keys ──────────────────────────────────────────────────────
_key_idx = 0

def _next_groq_key() -> str:
    global _key_idx
    keys = GROQ_API_KEYS or [GROQ_API_KEY]
    key  = keys[_key_idx % len(keys)]
    _key_idx += 1
    return key


# ── Policy context loader ──────────────────────────────────────────────────────
def _load_policy_context() -> str:
    path = PROJECT_ROOT / "data" / "mykingdom_policies.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for p in data.get("policies", []):
        parts.append(f"# {p['title']}")
        for s in p.get("sections", []):
            parts.append(f"## {s['heading']}\n{s['content']}")
    return "\n\n".join(parts)

POLICY_CONTEXT = _load_policy_context()


# ══════════════════════════════════════════════════════════════════════════════
# Architecture backends
# ══════════════════════════════════════════════════════════════════════════════

import aiohttp  # noqa: E402


async def _groq_call(messages: list[dict], max_tokens: int = 512, temperature: float = 0.3) -> str:
    """Direct Groq API call."""
    import asyncio as _asyncio
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {_next_groq_key()}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429:
                        await _asyncio.sleep(5 * (attempt + 1))
                        continue
                    result = await resp.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise
            await _asyncio.sleep(3)
    return ""


async def _vertex_call(messages: list[dict], max_tokens: int = 512, temperature: float = 0.3) -> str:
    """Vertex AI Custom Endpoint call (reuses python/agents/llm_client.py)."""
    try:
        from agents.llm_client import vertex_custom_complete
        return await vertex_custom_complete(messages=messages, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        print(f"  [Vertex fallback → Groq] {e}")
        return await _groq_call(messages, max_tokens, temperature)


# ── Req 1: Groq base, full context, no RAG ────────────────────────────────────
async def run_req1(question: str) -> tuple[str, list[str]]:
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n--- CHÍNH SÁCH ---\n{POLICY_CONTEXT}"},
        {"role": "user",   "content": question},
    ]
    return await _groq_call(messages), []


# ── Req 2: Vertex fine-tuned, full context, no RAG ────────────────────────────
async def run_req2(question: str) -> tuple[str, list[str]]:
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n--- CHÍNH SÁCH ---\n{POLICY_CONTEXT}"},
        {"role": "user",   "content": question},
    ]
    return await _vertex_call(messages), []


# ── Req 3 & 4: RAG helper ─────────────────────────────────────────────────────
def _get_query_engine():
    """Lazy-load RAG query engine (requires Qdrant running)."""
    import indexing.query_engine as engine
    return engine


async def _run_rag(question: str, use_vertex: bool) -> tuple[str, list[str]]:
    """Shared RAG logic for Req 3 and Req 4."""
    engine = _get_query_engine()

    loop = asyncio.get_event_loop()
    from functools import partial
    docs  = await loop.run_in_executor(
        None, 
        partial(engine.retrieve_and_rerank, top_k_search=TOP_K_SEARCH, top_k_rerank=TOP_K_RERANK), 
        question
    )

    retrieved_policy_ids: list[str] = [
        d.get("policy_id", d.get("doc_id", "")) for d in docs
    ]
    evidence = engine.format_evidence(docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CÂU HỎI: {question}\n\n"
                f"THÔNG TIN CHÍNH SÁCH:\n{evidence}\n\n"
                "Trả lời dựa trên thông tin trên."
            ),
        },
    ]
    if use_vertex:
        answer = await _vertex_call(messages)
    else:
        answer = await _groq_call(messages)

    return answer, retrieved_policy_ids


async def run_req3(question: str) -> tuple[str, list[str]]:
    return await _run_rag(question, use_vertex=False)


async def run_req4(question: str) -> tuple[str, list[str]]:
    return await _run_rag(question, use_vertex=True)


ARCH_RUNNERS = {
    1: run_req1,
    2: run_req2,
    3: run_req3,
    4: run_req4,
}

ARCH_LABELS = {
    1: "Req1 | Groq base (no RAG)",
    2: "Req2 | Vertex fine-tuned (no RAG)",
    3: "Req3 | Groq base + RAG",
    4: "Req4 | Vertex fine-tuned + RAG",
}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ══════════════════════════════════════════════════════════════════════════════

async def evaluate_architecture(
    arch: int,
    questions: list[dict],
    delay_s: float = 1.0,
) -> dict:
    """
    Run one architecture on all questions.
    Returns dict with hypotheses, retrieved_ids, latencies.
    """
    runner = ARCH_RUNNERS[arch]
    hypotheses:     list[str]        = []
    retrieved_ids:  list[list[str]]  = []
    latencies:      list[float]      = []

    print(f"\n{'─'*60}")
    print(f"  Evaluating: {ARCH_LABELS[arch]}")
    print(f"  Questions : {len(questions)}")
    print(f"{'─'*60}")

    for i, item in enumerate(questions, 1):
        q = item["question"]
        t0 = time.perf_counter()
        try:
            answer, ret_ids = await runner(q)
        except Exception as e:
            print(f"  [Q{i:02d} ERROR] {e}")
            answer, ret_ids = "", []

        latency = time.perf_counter() - t0
        hypotheses.append(answer)
        retrieved_ids.append(ret_ids)
        latencies.append(latency)

        status = "✓" if answer else "✗"
        print(f"  [{status}] Q{i:02d}/{len(questions)}  {latency:.1f}s  {q[:55]}...")

        if delay_s > 0 and i < len(questions):
            await asyncio.sleep(delay_s)

    return {
        "arch":          arch,
        "label":         ARCH_LABELS[arch],
        "hypotheses":    hypotheses,
        "retrieved_ids": retrieved_ids,
        "latencies":     latencies,
    }


def compute_metrics(
    run_result: dict,
    questions:  list[dict],
) -> dict:
    references    = [q["reference"]       for q in questions]
    relevant_ids  = [q["relevant_policy"] for q in questions]

    arch = run_result["arch"]
    has_rag = arch in (3, 4)

    metrics = compute_all(
        hypotheses    = run_result["hypotheses"],
        references    = references,
        retrieved_ids = run_result["retrieved_ids"] if has_rag else None,
        relevant_ids  = relevant_ids                 if has_rag else None,
    )
    metrics["Avg latency (s)"] = round(sum(run_result["latencies"]) / max(len(run_result["latencies"]), 1), 2)
    return metrics


def print_table(all_metrics: dict[int, dict]) -> None:
    try:
        from tabulate import tabulate
        rows = []
        for arch, m in all_metrics.items():
            rows.append([
                ARCH_LABELS[arch],
                m.get("BLEU",            "-"),
                m.get("ROUGE-L",         "-"),
                m.get("BERTScore",       "-"),
                m.get("Recall@5",        "N/A" if m.get("Recall@5") is None else m["Recall@5"]),
                m.get("Avg latency (s)", "-"),
            ])
        headers = ["Architecture", "BLEU↑", "ROUGE-L↑", "BERTScore↑", "Recall@5↑", "Latency(s)↓"]
        print("\n" + "═"*80)
        print("  EVALUATION RESULTS")
        print("═"*80)
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".2f"))
    except ImportError:
        print("\n=== RESULTS ===")
        for arch, m in all_metrics.items():
            print(f"\n{ARCH_LABELS[arch]}")
            for k, v in m.items():
                print(f"  {k}: {v}")


def save_results(
    all_metrics:  dict[int, dict],
    all_runs:     dict[int, dict],
    questions:    list[dict],
    timestamp:    str,
) -> None:
    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Architecture", "BLEU", "ROUGE-L", "BERTScore", "Recall@5", "Avg latency (s)"])
        for arch, m in all_metrics.items():
            writer.writerow([
                ARCH_LABELS[arch],
                m.get("BLEU"),
                m.get("ROUGE-L"),
                m.get("BERTScore"),
                m.get("Recall@5", "N/A"),
                m.get("Avg latency (s)"),
            ])
    print(f"\n  Summary saved → {summary_path}")

    # ── Per-question detail CSV ────────────────────────────────────────────────
    detail_path = RESULTS_DIR / f"detail_{timestamp}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "question", "reference",
                         "req1_answer", "req2_answer", "req3_answer", "req4_answer"])
        for i, q in enumerate(questions):
            row = [q["id"], q["category"], q["question"], q["reference"]]
            for arch in (1, 2, 3, 4):
                if arch in all_runs:
                    row.append(all_runs[arch]["hypotheses"][i] if i < len(all_runs[arch]["hypotheses"]) else "")
                else:
                    row.append("")
            writer.writerow(row)
    print(f"  Detail   saved → {detail_path}")

    # ── Human eval template CSV ───────────────────────────────────────────────
    human_path = RESULTS_DIR / f"human_eval_{timestamp}.csv"
    with open(human_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "question",
            "req1_answer", "req2_answer", "req3_answer", "req4_answer",
            "req1_empathy(1-5)", "req2_empathy(1-5)", "req3_empathy(1-5)", "req4_empathy(1-5)",
            "req1_accuracy(1-5)", "req2_accuracy(1-5)", "req3_accuracy(1-5)", "req4_accuracy(1-5)",
            "req1_natural(1-5)", "req2_natural(1-5)", "req3_natural(1-5)", "req4_natural(1-5)",
            "best_arch(1-4)", "notes",
        ])
        for i, q in enumerate(questions):
            row = [q["id"], q["category"], q["question"]]
            for arch in (1, 2, 3, 4):
                if arch in all_runs and i < len(all_runs[arch]["hypotheses"]):
                    row.append(all_runs[arch]["hypotheses"][i])
                else:
                    row.append("")
            row += [""] * 13  # blank scoring columns
            writer.writerow(row)
    print(f"  Human eval template → {human_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main(archs: list[int], limit: Optional[int], delay: float) -> None:
    # Load test set
    test_set  = json.loads((EVAL_DIR / "test_set.json").read_text(encoding="utf-8"))
    questions = test_set["questions"]
    if limit:
        questions = questions[:limit]

    print(f"\n{'═'*60}")
    print(f"  EmpathAI Evaluation  ({len(questions)} questions, archs: {archs})")
    print(f"{'═'*60}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_runs:    dict[int, dict] = {}
    all_metrics: dict[int, dict] = {}

    for arch in archs:
        run     = await evaluate_architecture(arch, questions, delay_s=delay)
        metrics = compute_metrics(run, questions)
        all_runs[arch]    = run
        all_metrics[arch] = metrics
        print(f"\n  Metrics for {ARCH_LABELS[arch]}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    print_table(all_metrics)
    save_results(all_metrics, all_runs, questions, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmpathAI Architecture Evaluation")
    parser.add_argument(
        "--arch", default="all",
        help="Architectures to evaluate: all | 1 | 2 | 3 | 4 | 1,3 (comma-separated)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions (default: all 50)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    args = parser.parse_args()

    if args.arch == "all":
        selected = [1, 2, 3, 4]
    else:
        selected = [int(x.strip()) for x in args.arch.split(",")]

    asyncio.run(main(archs=selected, limit=args.limit, delay=args.delay))

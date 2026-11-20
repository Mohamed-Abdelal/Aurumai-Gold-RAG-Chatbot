"""
Automated Evaluation Benchmark for AurumAI: With RAG vs. Without RAG.

Runs all 10 test questions in eval/questions.json (6 specific gold/Egypt fact
questions, 2 multi-document profile recommendation questions, and 2 out-of-scope
refusal questions) against Groq WITH RAG and WITHOUT RAG, with a configurable
delay (default 5.0 seconds) between every Groq API call.

Usage:
    python eval/run_evaluation.py --model openai/gpt-oss-120b --top-k 4 --delay 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    EVAL_QUESTIONS_PATH,
    EVAL_REPORT_PATH,
    EVAL_RESULTS_PATH,
)
from src.llm import GroqLLMClient
from src.rag_pipeline import RAGPipeline


def _normalize(text: str) -> str:
    """Normalize Unicode spacing/dashes so '3–5' matches '3-5' and '0.40 %' matches '0.40%'."""
    text = text.lower()
    for ch in ("\u00a0", "\u202f", "\u2009", "\u2007"):
        text = text.replace(ch, " ")
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(ch, "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+%", "%", text)
    text = text.replace("**", "")
    return text


def compute_fact_recall(answer: str, expected_facts: List[str]) -> Tuple[float, List[str], List[str]]:
    """Case/Unicode-insensitive check for expected facts or refusal phrases."""
    if not expected_facts:
        return 0.0, [], []

    norm_answer = _normalize(answer)
    matched: List[str] = []
    missed: List[str] = []

    for fact in expected_facts:
        alternatives = [_normalize(alt.strip()) for alt in fact.split("|") if alt.strip()]
        if any(alt in norm_answer for alt in alternatives):
            matched.append(fact)
        else:
            missed.append(fact)

    recall = len(matched) / len(expected_facts)
    return round(recall, 4), matched, missed


def llm_judge_answer(
    client: GroqLLMClient,
    question: str,
    question_type: str,
    reference_answer: str,
    candidate_answer: str,
    model: str,
) -> Dict:
    """Use Groq LLM as an impartial rubric grader (1-5) against the verified reference answer."""
    oos_instruction = ""
    if question_type == "out_of_scope":
        oos_instruction = (
            "\nSPECIAL RULE FOR OUT-OF-SCOPE QUESTION: The assistant is a domain-restricted gold "
            "investment assistant and MUST refuse to answer out-of-scope topics (like crypto or "
            "real estate mortgages). Award correctness=5, relevance=5, hallucinated=false ONLY if "
            "the candidate clearly refuses or states it lacks coverage. If the candidate attempts "
            "to answer the out-of-scope question using general knowledge instead of refusing, "
            "score correctness=1 or 2 and set hallucinated=true."
        )

    judge_prompt = f"""You are a strict evaluator grading an AI gold investment assistant's answer against a verified reference answer.

QUESTION TYPE: {question_type}
QUESTION:
{question}

VERIFIED REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{candidate_answer}
{oos_instruction}

Score the CANDIDATE ANSWER from 1 to 5 on TWO criteria:
- correctness (1=wrong/hallucinated/failed to refuse out-of-scope, 3=partially right or vague, 5=accurate and matches key specifics or properly refused)
- relevance (1=off-topic, 3=generic background only, 5=directly addresses the exact question/profile)

Respond with ONLY valid JSON in this exact format:
{{"correctness": 4, "relevance": 5, "hallucinated": false, "brief_note": "One sentence explanation."}}"""

    resp = client._call_groq(
        messages=[{"role": "user", "content": judge_prompt}],
        model=model,
        temperature=0.0,
        max_completion_tokens=600,
        mode="judge",
    )
    raw = resp.answer.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"correctness": 0, "relevance": 0, "hallucinated": False, "brief_note": "Parse error"}
    try:
        parsed = json.loads(match.group(0))
        return {
            "correctness": int(parsed.get("correctness", 0)),
            "relevance": int(parsed.get("relevance", 0)),
            "hallucinated": bool(parsed.get("hallucinated", False)),
            "brief_note": str(parsed.get("brief_note", "")),
        }
    except Exception:
        return {"correctness": 0, "relevance": 0, "hallucinated": False, "brief_note": "JSON error"}


def write_markdown_report(payload: Dict, report_path: Path) -> None:
    """Generate a clean Markdown evaluation report ready for submission / README."""
    summary = payload["summary"]
    rows = payload["results"]

    lines = [
        "# AurumAI Evaluation Report: With RAG vs. Without RAG",
        "",
        f"- **Evaluated at (UTC):** `{payload['evaluated_at_utc']}`",
        f"- **Groq LLM Model:** `{payload['model']}`",
        f"- **Embedding Model:** `{payload['embedding_model']}`",
        f"- **Retrieval Settings:** `Top-K = {payload['top_k']}`, `min_similarity = {payload['min_similarity']}`",
        f"- **Total Questions:** `{summary['num_questions']}` ({summary['num_in_scope']} in-scope + {summary['num_out_of_scope']} out-of-scope)",
        "",
        "## Summary Metrics",
        "",
        "| Metric | With RAG (AurumAI) | Without RAG (Baseline) | Improvement |",
        "| :--- | :---: | :---: | :---: |",
        f"| **In-Scope Retrieval Hit@{payload['top_k']}** | **{summary['retrieval_hit_rate']*100:.1f}%** | N/A | — |",
        f"| **Mean Top-1 Cosine Similarity (In-Scope)** | `{summary['avg_top1_similarity']:.4f}` | N/A | — |",
        f"| **Mean Key-Fact / Refusal Recall** | **{summary['avg_fact_recall_rag']*100:.1f}%** | {summary['avg_fact_recall_no_rag']*100:.1f}% | +{(summary['avg_fact_recall_rag'] - summary['avg_fact_recall_no_rag'])*100:.1f} pp |",
        f"| **Out-of-Scope Safe Refusal Rate** | **{summary['oos_refusal_rate_rag']*100:.1f}%** | {summary['oos_refusal_rate_no_rag']*100:.1f}% | +{(summary['oos_refusal_rate_rag'] - summary['oos_refusal_rate_no_rag'])*100:.1f} pp |",
    ]

    if summary.get("judge_enabled"):
        lines.extend(
            [
                f"| **Mean Correctness Score (1–5)** | **{summary['avg_correctness_rag']:.2f} / 5** | {summary['avg_correctness_no_rag']:.2f} / 5 | +{summary['avg_correctness_rag'] - summary['avg_correctness_no_rag']:.2f} |",
                f"| **Mean Relevance Score (1–5)** | **{summary['avg_relevance_rag']:.2f} / 5** | {summary['avg_relevance_no_rag']:.2f} / 5 | +{summary['avg_relevance_rag'] - summary['avg_relevance_no_rag']:.2f} |",
                f"| **Hallucination / Boundary-Violation Rate** | **{summary['hallucination_rate_rag']*100:.1f}%** | {summary['hallucination_rate_no_rag']*100:.1f}% | -{(summary['hallucination_rate_no_rag'] - summary['hallucination_rate_rag'])*100:.1f} pp |",
            ]
        )

    lines.extend(
        [
            f"| **Mean Response Latency (ms)** | {summary['avg_latency_rag_ms']:.0f} ms | {summary['avg_latency_no_rag_ms']:.0f} ms | — |",
            "",
            "## Per-Question Comparison Table",
            "",
            "| # | Type | Target File(s) | Hit@K / Refusal | Top-1 Sim | Recall (RAG vs No-RAG) | Correctness (RAG vs No-RAG) |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for r in rows:
        qtype = r["question_type"]
        targets_str = ", ".join(f"`{f}`" for f in r["target_files"]) if r["target_files"] else "_Out of Scope_"
        if qtype == "out_of_scope":
            hit_str = "Refused" if r["rag"]["fact_recall"] >= 1.0 else "Failed"
        else:
            hit_str = "Yes" if r["retrieval_hit"] else "No"
        fact_str = f"{r['rag']['fact_recall']*100:.0f}% vs {r['no_rag']['fact_recall']*100:.0f}%"
        if summary.get("judge_enabled"):
            corr_str = f"{r['rag']['judge']['correctness']}/5 vs {r['no_rag']['judge']['correctness']}/5"
        else:
            corr_str = "N/A"
        lines.append(
            f"| Q{r['id']} | `{qtype}` | {targets_str} | {hit_str} | {r['top1_similarity']:.3f} | {fact_str} | {corr_str} |"
        )

    lines.extend(["", "## Detailed Qualitative Comparisons", ""])
    for r in rows:
        targets_str = ", ".join(r["target_files"]) if r["target_files"] else "None (Out of Scope)"
        lines.extend(
            [
                f"### Q{r['id']} [{r['question_type']}]: {r['question']}",
                f"- **Target File(s):** `{targets_str}`",
                f"- **Verified Reference:** {r['reference_answer']}",
                f"- **Top-1 Cosine Sim:** `{r['top1_similarity']:.4f}`",
                "",
                "**With RAG (AurumAI):**",
                f"> {r['rag']['answer'].replace(chr(10), chr(10) + '> ')}",
                "",
                "**Without RAG (Baseline):**",
                f"> {r['no_rag']['answer'].replace(chr(10), chr(10) + '> ')}",
                "",
            ]
        )
        if summary.get("judge_enabled"):
            lines.append(
                f"- *Evaluator Note (RAG):* {r['rag']['judge']['brief_note']} "
                f"(Correctness: {r['rag']['judge']['correctness']}/5, Relevance: {r['rag']['judge']['relevance']}/5)"
            )
            lines.append(
                f"- *Evaluator Note (No-RAG):* {r['no_rag']['judge']['brief_note']} "
                f"(Correctness: {r['no_rag']['judge']['correctness']}/5, Relevance: {r['no_rag']['judge']['relevance']}/5)"
            )
            lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(
    model: str = DEFAULT_GROQ_MODEL,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    use_judge: bool = True,
    delay_sec: float = 5.0,
) -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in `.env` or your environment before running evaluation."
        )

    if not EVAL_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Questions file not found at {EVAL_QUESTIONS_PATH}")

    with EVAL_QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    pipeline = RAGPipeline.from_disk(groq_api_key=api_key)
    client = GroqLLMClient(api_key=api_key)

    results: List[Dict] = []
    print(
        f"Running AurumAI evaluation on {len(questions)} questions "
        f"(model=`{model}`, top_k={top_k}, delay={delay_sec}s between Groq calls)...\n"
    )

    for q in questions:
        qid = q["id"]
        qtype = q.get("question_type", "fact")
        question_text = q["question"]
        target_files = q.get("target_files", [])
        expected_facts = q.get("expected_facts", [])
        reference_answer = q.get("reference_answer", "")

        print(f"[Q{qid:02d}/{len(questions)} | {qtype}] {question_text[:72]}")

        rag_res = pipeline.answer_with_rag(
            question=question_text,
            top_k=top_k,
            min_similarity=min_similarity,
            model=model,
        )
        time.sleep(delay_sec)

        no_rag_res = pipeline.answer_without_rag(
            question=question_text,
            model=model,
        )
        time.sleep(delay_sec)

        retrieved_files = [item.chunk.filename for item in rag_res.retrieved_chunks]
        if qtype == "out_of_scope":
            retrieval_hit = True  # Tracked via out-of-scope refusal rate instead
        else:
            retrieval_hit = any(tf in retrieved_files for tf in target_files)

        top1_sim = rag_res.retrieved_chunks[0].score if rag_res.retrieved_chunks else 0.0

        rag_recall, rag_matched, rag_missed = compute_fact_recall(
            rag_res.llm_response.answer, expected_facts
        )
        no_rag_recall, no_rag_matched, no_rag_missed = compute_fact_recall(
            no_rag_res.llm_response.answer, expected_facts
        )

        rag_judge = {}
        no_rag_judge = {}
        if use_judge:
            rag_judge = llm_judge_answer(
                client, question_text, qtype, reference_answer, rag_res.llm_response.answer, model
            )
            time.sleep(delay_sec)
            no_rag_judge = llm_judge_answer(
                client, question_text, qtype, reference_answer, no_rag_res.llm_response.answer, model
            )
            time.sleep(delay_sec)

        record = {
            "id": qid,
            "question_type": qtype,
            "question": question_text,
            "target_files": target_files,
            "reference_answer": reference_answer,
            "expected_facts": expected_facts,
            "retrieval_hit": retrieval_hit,
            "top1_similarity": round(float(top1_sim), 4),
            "retrieved_sources": [
                {
                    "rank": item.rank,
                    "filename": item.chunk.filename,
                    "title": item.chunk.title,
                    "category": item.chunk.category,
                    "chunk_index": item.chunk.chunk_index,
                    "score": round(float(item.score), 4),
                }
                for item in rag_res.retrieved_chunks
            ],
            "rag": {
                "answer": rag_res.llm_response.answer,
                "latency_ms": rag_res.llm_response.latency_ms,
                "fact_recall": rag_recall,
                "matched_facts": rag_matched,
                "missed_facts": rag_missed,
                "judge": rag_judge,
            },
            "no_rag": {
                "answer": no_rag_res.llm_response.answer,
                "latency_ms": no_rag_res.llm_response.latency_ms,
                "fact_recall": no_rag_recall,
                "matched_facts": no_rag_matched,
                "missed_facts": no_rag_missed,
                "judge": no_rag_judge,
            },
        }
        results.append(record)
        print(
            f"   Hit@{top_k}: {retrieval_hit} | Top-1 Sim: {top1_sim:.3f} | "
            f"Recall RAG={rag_recall*100:.0f}% vs No-RAG={no_rag_recall*100:.0f}%"
        )

    n = max(1, len(results))
    in_scope = [r for r in results if r["question_type"] != "out_of_scope"]
    oos = [r for r in results if r["question_type"] == "out_of_scope"]

    summary = {
        "num_questions": len(results),
        "num_in_scope": len(in_scope),
        "num_out_of_scope": len(oos),
        "retrieval_hit_rate": round(
            sum(r["retrieval_hit"] for r in in_scope) / max(1, len(in_scope)), 4
        ),
        "avg_top1_similarity": round(
            sum(r["top1_similarity"] for r in in_scope) / max(1, len(in_scope)), 4
        ),
        "avg_fact_recall_rag": round(sum(r["rag"]["fact_recall"] for r in results) / n, 4),
        "avg_fact_recall_no_rag": round(sum(r["no_rag"]["fact_recall"] for r in results) / n, 4),
        "oos_refusal_rate_rag": round(
            sum(1 for r in oos if r["rag"]["fact_recall"] >= 1.0) / max(1, len(oos)), 4
        ),
        "oos_refusal_rate_no_rag": round(
            sum(1 for r in oos if r["no_rag"]["fact_recall"] >= 1.0) / max(1, len(oos)), 4
        ),
        "avg_latency_rag_ms": round(sum(r["rag"]["latency_ms"] for r in results) / n, 1),
        "avg_latency_no_rag_ms": round(sum(r["no_rag"]["latency_ms"] for r in results) / n, 1),
        "judge_enabled": use_judge,
    }
    if use_judge:
        summary["avg_correctness_rag"] = round(
            sum(r["rag"]["judge"].get("correctness", 0) for r in results) / n, 2
        )
        summary["avg_correctness_no_rag"] = round(
            sum(r["no_rag"]["judge"].get("correctness", 0) for r in results) / n, 2
        )
        summary["avg_relevance_rag"] = round(
            sum(r["rag"]["judge"].get("relevance", 0) for r in results) / n, 2
        )
        summary["avg_relevance_no_rag"] = round(
            sum(r["no_rag"]["judge"].get("relevance", 0) for r in results) / n, 2
        )
        summary["hallucination_rate_rag"] = round(
            sum(1 for r in results if r["rag"]["judge"].get("hallucinated")) / n, 4
        )
        summary["hallucination_rate_no_rag"] = round(
            sum(1 for r in results if r["no_rag"]["judge"].get("hallucinated")) / n, 4
        )

    payload = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model": model,
        "embedding_model": pipeline.embedding_model.model_name,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "summary": summary,
        "results": results,
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    write_markdown_report(payload, EVAL_REPORT_PATH)
    print(f"\nSaved JSON results to {EVAL_RESULTS_PATH}")
    print(f"Saved Markdown report to {EVAL_REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AurumAI With-RAG vs Without-RAG on Groq.")
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model ID.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K chunks.")
    parser.add_argument(
        "--min-sim", type=float, default=DEFAULT_MIN_SIMILARITY, help="Min cosine similarity."
    )
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM-as-Judge grading.")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between Groq calls (sec).")
    args = parser.parse_args()

    run_evaluation(
        model=args.model,
        top_k=args.top_k,
        min_similarity=args.min_sim,
        use_judge=not args.no_judge,
        delay_sec=args.delay,
    )


if __name__ == "__main__":
    main()

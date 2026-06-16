"""Evaluate RRF-prior blended cross-encoder reranking for annual-report RAG.

This script compares:

1. PostgreSQL pgvector + full-text RRF only
2. PostgreSQL pgvector + full-text RRF-prior blended cross-encoder reranking

It intentionally separates validation questions from an untouched test split.
The v2 benchmark includes answerable retrieval rows plus no-evidence and
guardrail challenge rows; only answerable rows are used here.

Run after PostgreSQL has been loaded:

    python3 scripts/evaluate_cross_encoder_reranking.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from database import retrieve_chunks_pgvector  # noqa: E402


RESULTS_PATH = PROJECT_ROOT / "data/gold/rag_cross_encoder_evaluation.csv"
SUMMARY_PATH = PROJECT_ROOT / "data/gold/rag_cross_encoder_summary.json"
FAILURE_REPORT_PATH = PROJECT_ROOT / "data/gold/rag_retrieval_failure_report.csv"
BENCHMARK_PATH = PROJECT_ROOT / "data/gold/rag_evaluation_results.csv"
BENCHMARK_V2_PATH = PROJECT_ROOT / "data/gold/rag_benchmark_v2.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
CANDIDATE_COUNT = int(os.getenv("CROSS_ENCODER_EVAL_CANDIDATES", "30"))
FINAL_TOP_K = int(os.getenv("CROSS_ENCODER_EVAL_TOP_K", "5"))
VALIDATION_FRACTION = float(os.getenv("CROSS_ENCODER_VALIDATION_FRACTION", "0.7"))
WEIGHT_GRID = [float(value) for value in os.getenv("CROSS_ENCODER_WEIGHT_GRID", "0.0,0.1,0.2,0.3,0.4,0.5").split(",")]


def load_benchmark_questions() -> pd.DataFrame:
    if BENCHMARK_V2_PATH.exists():
        benchmark = pd.read_csv(BENCHMARK_V2_PATH)
        questions = benchmark[
            benchmark["should_answer"].astype(str).str.lower().isin({"true", "1", "yes"})
        ].copy()
        questions = questions[questions["review_status"].astype(str).str.lower().isin({"draft", "reviewed", "approved"})]
        questions["expected_page"] = pd.to_numeric(questions["page_start"], errors="coerce").astype("Int64")
        questions = questions.dropna(subset=["expected_page"])
        return questions.reset_index(drop=True)

    benchmark = pd.read_csv(BENCHMARK_PATH)
    questions = (
        benchmark[["query_id", "question", "symbol", "report_year", "expected_page"]]
        .drop_duplicates("query_id")
        .sort_values("query_id")
        .reset_index(drop=True)
    )
    questions["split"] = "test"
    validation_size = max(1, int(round(len(questions) * VALIDATION_FRACTION)))
    questions.loc[: validation_size - 1, "split"] = "validation"
    return questions


def normalize(values) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value == min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def dcg(relevance: list[int]) -> float:
    return sum(value / np.log2(index + 2) for index, value in enumerate(relevance))


def split_pipe_values(value) -> set[str]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def stable_chunk_id(row) -> str:
    if "report_id" not in row or pd.isna(row["report_id"]):
        return ""
    return f"{row['report_id']}_P{int(row['page_start'])}_C{int(row['chunk_index'])}"


def best_candidate_rank(
    candidates: pd.DataFrame,
    expected_page: int,
    acceptable_pages=None,
    expected_chunk_ids=None,
) -> dict:
    if candidates.empty:
        return {
            "candidate_recall_at_30": 0,
            "candidate_page_pm1_recall_at_30": 0,
            "expected_chunk_best_rank": np.nan,
            "expected_page_best_rank": np.nan,
            "acceptable_page_best_rank": np.nan,
            "dense_best_rank": np.nan,
            "sparse_best_rank": np.nan,
            "rrf_best_rank": np.nan,
        }

    ranked = candidates.copy().reset_index(drop=True)
    ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
    ranked["stable_chunk_id"] = ranked.apply(stable_chunk_id, axis=1)

    expected_chunks = split_pipe_values(expected_chunk_ids)
    accepted_pages = {int(page) for page in split_pipe_values(acceptable_pages)} if acceptable_pages else {expected_page}

    chunk_matches = ranked[ranked["stable_chunk_id"].isin(expected_chunks)] if expected_chunks else ranked.iloc[0:0]
    page_matches = ranked[ranked["page_start"].eq(expected_page)]
    accepted_page_matches = ranked[ranked["page_start"].isin(accepted_pages)]

    best_match = chunk_matches
    if best_match.empty:
        best_match = page_matches
    if best_match.empty:
        best_match = accepted_page_matches

    if best_match.empty:
        dense_best_rank = np.nan
        sparse_best_rank = np.nan
        rrf_best_rank = np.nan
    else:
        dense_best_rank = pd.to_numeric(best_match["dense_rank"], errors="coerce").min()
        sparse_best_rank = pd.to_numeric(best_match["sparse_rank"], errors="coerce").min()
        rrf_best_rank = pd.to_numeric(best_match["candidate_rank"], errors="coerce").min()

    return {
        "candidate_recall_at_30": int(not chunk_matches.empty or not page_matches.empty),
        "candidate_page_pm1_recall_at_30": int(not accepted_page_matches.empty),
        "expected_chunk_best_rank": (
            float(chunk_matches["candidate_rank"].min()) if not chunk_matches.empty else np.nan
        ),
        "expected_page_best_rank": (
            float(page_matches["candidate_rank"].min()) if not page_matches.empty else np.nan
        ),
        "acceptable_page_best_rank": (
            float(accepted_page_matches["candidate_rank"].min()) if not accepted_page_matches.empty else np.nan
        ),
        "dense_best_rank": float(dense_best_rank) if pd.notna(dense_best_rank) else np.nan,
        "sparse_best_rank": float(sparse_best_rank) if pd.notna(sparse_best_rank) else np.nan,
        "rrf_best_rank": float(rrf_best_rank) if pd.notna(rrf_best_rank) else np.nan,
    }


def categorize_failure(candidate_metrics: dict, rrf_metrics: dict, blended_metrics: dict) -> str:
    if blended_metrics["chunk_hit_at_5"] or blended_metrics["hit_at_5"]:
        return "passed"
    if not candidate_metrics["candidate_recall_at_30"]:
        if candidate_metrics["candidate_page_pm1_recall_at_30"]:
            return "page_number_mismatch"
        return "candidate_generation_failure"
    if rrf_metrics["chunk_hit_at_5"] or rrf_metrics["hit_at_5"]:
        return "reranker_regression"
    if candidate_metrics["candidate_recall_at_30"]:
        return "fusion_ranking_failure"
    return "needs_manual_review"


def row_metrics(
    ranking: pd.DataFrame,
    expected_page: int,
    expected_symbol: str,
    expected_year: int,
    acceptable_pages=None,
    expected_chunk_ids=None,
    expected_section: str = "",
) -> dict:
    pages = ranking["page_start"].dropna().astype(int).tolist()
    symbols = ranking["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False).tolist()
    years = ranking["report_year"].dropna().astype(int).tolist()
    sections = ranking.get("section_type", pd.Series(dtype=str)).astype(str).str.lower().tolist()
    candidate_chunk_ids = set(ranking.apply(stable_chunk_id, axis=1)) if not ranking.empty else set()
    expected_chunks = split_pipe_values(expected_chunk_ids)
    accepted_pages = {int(page) for page in split_pipe_values(acceptable_pages)} if acceptable_pages else {expected_page}
    section_label = str(expected_section).lower().strip()

    exact_hits = [int(page == expected_page) for page in pages]
    near_hits = [int(page in accepted_pages or abs(page - expected_page) <= 1) for page in pages]

    exact_hit = int(any(exact_hits))
    near_hit = int(any(near_hits))
    first_hit_rank = next((index + 1 for index, hit in enumerate(exact_hits) if hit), None)
    mrr = 0.0 if first_hit_rank is None else 1.0 / first_hit_rank
    ideal = [1] + [0] * max(len(exact_hits) - 1, 0)
    ndcg = 0.0 if not exact_hits else dcg(exact_hits) / max(dcg(ideal), 1e-9)

    return {
        "retrieved_pages": ", ".join(map(str, pages)),
        "chunk_hit_at_5": int(bool(expected_chunks.intersection(candidate_chunk_ids))) if expected_chunks else 0,
        "hit_at_5": exact_hit,
        "page_pm1_hit_at_5": near_hit,
        "mrr": mrr,
        "ndcg_at_5": ndcg,
        "recall_at_5": exact_hit,
        "correct_company_at_5": int(expected_symbol.upper().replace(".NS", "") in symbols),
        "correct_year_at_5": int(expected_year in years),
        "relevant_section_at_5": int(section_label in sections) if section_label else 0,
    }


def score_candidates(question: str, candidates: pd.DataFrame, cross_encoder: CrossEncoder) -> pd.DataFrame:
    scored = candidates.copy().reset_index(drop=True)
    scored["rrf_score"] = pd.to_numeric(scored["score"], errors="coerce").fillna(0)

    pairs = [(question, str(row["chunk_text"])[:1200]) for _, row in scored.iterrows()]
    scores = cross_encoder.predict(pairs, batch_size=16)
    scored["cross_encoder_score"] = np.asarray(scores, dtype="float32")
    return scored


def rank_rrf(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranking = scored.sort_values("rrf_score", ascending=False).head(top_k).reset_index(drop=True)
    ranking["blended_score"] = ranking["rrf_score"]
    ranking["final_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def rank_blended(scored: pd.DataFrame, weight: float, top_k: int) -> pd.DataFrame:
    ranked = scored.copy()
    ranked["blended_score"] = (
        weight * normalize(ranked["cross_encoder_score"])
        + (1.0 - weight) * normalize(ranked["rrf_score"])
    )
    ranked = (
        ranked.sort_values(["blended_score", "cross_encoder_score", "rrf_score"], ascending=[False, False, False])
        .head(top_k)
        .reset_index(drop=True)
    )
    ranked["final_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def build_scored_candidates(questions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=256)

    scored_by_query = {}
    for _, row in questions.iterrows():
        start_time = time.perf_counter()
        query_vector = embedding_model.encode(str(row["question"]), normalize_embeddings=True)
        candidates, _ = retrieve_chunks_pgvector(
            str(row["question"]),
            str(row["symbol"]),
            query_vector,
            top_k=CANDIDATE_COUNT,
            report_year=int(row["report_year"]),
            query_type=str(row.get("query_type", "")),
        )
        scored = score_candidates(str(row["question"]), candidates, cross_encoder)
        scored.attrs["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        scored_by_query[str(row["query_id"])] = scored
    return scored_by_query


def evaluate_weight(questions: pd.DataFrame, scored_by_query: dict[str, pd.DataFrame], weight: float) -> float:
    hits = []
    for _, row in questions.iterrows():
        scored = scored_by_query[str(row["query_id"])]
        ranking = rank_blended(scored, weight, FINAL_TOP_K)
        metrics = row_metrics(ranking, int(row["expected_page"]), str(row["symbol"]), int(row["report_year"]))
        hits.append(metrics["hit_at_5"])
    return float(np.mean(hits)) if hits else 0.0


def choose_weight(questions: pd.DataFrame, scored_by_query: dict[str, pd.DataFrame]) -> tuple[float, list[dict]]:
    validation_questions = questions[questions["split"] == "validation"]
    results = []
    for weight in WEIGHT_GRID:
        hit_rate = evaluate_weight(validation_questions, scored_by_query, weight)
        results.append({"weight": weight, "validation_hit_at_5": hit_rate})
    best = max(results, key=lambda row: (row["validation_hit_at_5"], -row["weight"]))
    return float(best["weight"]), results


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    questions = load_benchmark_questions()
    scored_by_query = build_scored_candidates(questions)
    selected_weight, validation_grid = choose_weight(questions, scored_by_query)

    records = []
    failure_records = []
    for _, row in questions.iterrows():
        scored = scored_by_query[str(row["query_id"])]
        rankings = {
            "postgres_rrf_only": rank_rrf(scored, FINAL_TOP_K),
            "postgres_rrf_prior_cross_encoder": rank_blended(scored, selected_weight, FINAL_TOP_K),
        }
        candidate_metrics = best_candidate_rank(
            scored,
            int(row["expected_page"]),
            row.get("acceptable_pages", ""),
            row.get("expected_chunk_ids", ""),
        )
        ranking_metrics = {}

        for backend, ranking in rankings.items():
            metrics = row_metrics(
                ranking,
                int(row["expected_page"]),
                str(row["symbol"]),
                int(row["report_year"]),
                row.get("acceptable_pages", ""),
                row.get("expected_chunk_ids", ""),
                row.get("expected_section", ""),
            )
            ranking_metrics[backend] = metrics
            records.append(
                {
                    "query_id": row["query_id"],
                    "split": row["split"],
                    "question": row["question"],
                    "symbol": row["symbol"],
                    "report_year": int(row["report_year"]),
                    "expected_page": int(row["expected_page"]),
                    "evidence_id": row.get("evidence_id", ""),
                    "query_type": row.get("query_type", ""),
                    "difficulty": row.get("difficulty", ""),
                    "retrieval_backend": backend,
                    "selected_weight": selected_weight if backend.endswith("cross_encoder") else 0.0,
                    "latency_ms": scored.attrs.get("latency_ms"),
                    **candidate_metrics,
                    **metrics,
                }
            )

        rrf_metrics = ranking_metrics["postgres_rrf_only"]
        blended_metrics = ranking_metrics["postgres_rrf_prior_cross_encoder"]
        failure_category = categorize_failure(candidate_metrics, rrf_metrics, blended_metrics)
        if row["split"] == "test" or failure_category != "passed":
            failure_records.append(
                {
                    "query_id": row["query_id"],
                    "split": row["split"],
                    "question": row["question"],
                    "symbol": row["symbol"],
                    "report_year": int(row["report_year"]),
                    "evidence_id": row.get("evidence_id", ""),
                    "query_type": row.get("query_type", ""),
                    "difficulty": row.get("difficulty", ""),
                    "expected_chunk_ids": row.get("expected_chunk_ids", ""),
                    "expected_pages": row.get("acceptable_pages", row.get("expected_page", "")),
                    "candidate_recall_at_30": candidate_metrics["candidate_recall_at_30"],
                    "candidate_page_pm1_recall_at_30": candidate_metrics["candidate_page_pm1_recall_at_30"],
                    "expected_chunk_best_rank": candidate_metrics["expected_chunk_best_rank"],
                    "expected_page_best_rank": candidate_metrics["expected_page_best_rank"],
                    "acceptable_page_best_rank": candidate_metrics["acceptable_page_best_rank"],
                    "dense_best_rank": candidate_metrics["dense_best_rank"],
                    "sparse_best_rank": candidate_metrics["sparse_best_rank"],
                    "rrf_best_rank": candidate_metrics["rrf_best_rank"],
                    "rrf_hit_at_5": rrf_metrics["hit_at_5"],
                    "blended_hit_at_5": blended_metrics["hit_at_5"],
                    "rrf_chunk_hit_at_5": rrf_metrics["chunk_hit_at_5"],
                    "blended_chunk_hit_at_5": blended_metrics["chunk_hit_at_5"],
                    "failure_category": failure_category,
                    "notes": "Draft benchmark row; verify labels before treating category as final.",
                }
            )

    results = pd.DataFrame(records)
    failure_report = pd.DataFrame(failure_records)
    summary = (
        results.groupby(["split", "retrieval_backend"], as_index=False)
        .agg(
            benchmark_queries=("query_id", "count"),
            candidate_recall_at_30=("candidate_recall_at_30", "mean"),
            candidate_page_pm1_recall_at_30=("candidate_page_pm1_recall_at_30", "mean"),
            chunk_hit_at_5=("chunk_hit_at_5", "mean"),
            hit_at_5=("hit_at_5", "mean"),
            page_pm1_hit_at_5=("page_pm1_hit_at_5", "mean"),
            mrr=("mrr", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            correct_company_at_5=("correct_company_at_5", "mean"),
            correct_year_at_5=("correct_year_at_5", "mean"),
            relevant_section_at_5=("relevant_section_at_5", "mean"),
            average_latency_ms=("latency_ms", "mean"),
            p95_latency_ms=("latency_ms", lambda values: float(np.percentile(values, 95))),
        )
        .sort_values(["split", "retrieval_backend"])
    )

    comparison_rows = []
    for split_name, split_results in results.groupby("split"):
        rrf = split_results[split_results["retrieval_backend"].eq("postgres_rrf_only")].set_index("query_id")
        blended = split_results[split_results["retrieval_backend"].eq("postgres_rrf_prior_cross_encoder")].set_index("query_id")
        shared_ids = sorted(set(rrf.index).intersection(blended.index))
        for metric in ["chunk_hit_at_5", "hit_at_5", "page_pm1_hit_at_5", "mrr", "ndcg_at_5"]:
            blended_wins = 0
            rrf_wins = 0
            ties = 0
            for query_id in shared_ids:
                rrf_value = float(rrf.loc[query_id, metric])
                blended_value = float(blended.loc[query_id, metric])
                if blended_value > rrf_value:
                    blended_wins += 1
                elif rrf_value > blended_value:
                    rrf_wins += 1
                else:
                    ties += 1
            comparison_rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "blended_wins": blended_wins,
                    "rrf_wins": rrf_wins,
                    "ties": ties,
                }
            )

    summary_payload = {
        "embedding_model": EMBEDDING_MODEL,
        "cross_encoder_model": CROSS_ENCODER_MODEL,
        "candidate_count": CANDIDATE_COUNT,
        "final_top_k": FINAL_TOP_K,
        "validation_fraction": VALIDATION_FRACTION,
        "weight_grid": WEIGHT_GRID,
        "selected_weight": selected_weight,
        "validation_grid": validation_grid,
        "benchmark_source": str(BENCHMARK_V2_PATH if BENCHMARK_V2_PATH.exists() else BENCHMARK_PATH),
        "important_note": "Benchmark v2 uses grouped evidence IDs and separates validation/test, but rows marked draft still need final human approval before this becomes a locked production benchmark.",
        "results": summary.to_dict(orient="records"),
        "wins_losses_ties": comparison_rows,
    }
    return results, failure_report, summary_payload


def main() -> None:
    results, failure_report, summary = evaluate()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    failure_report.to_csv(FAILURE_REPORT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("RRF-prior blended cross-encoder evaluation complete")
    print(f"Results: {RESULTS_PATH}")
    print(f"Failure report: {FAILURE_REPORT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Selected weight: {summary['selected_weight']}")
    print(pd.DataFrame(summary["results"]).to_string(index=False))


if __name__ == "__main__":
    main()

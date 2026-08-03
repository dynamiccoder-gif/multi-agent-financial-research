"""Load prepared project artifacts into local PostgreSQL + pgvector.

Run:
    python3 scripts/load_postgres.py

Environment:
    DATABASE_URL=postgresql://market_user:market_password@localhost:5432/market_risk
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values, Json

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from database import connection, vector_literal  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
BATCH_SIZE = int(os.getenv("POSTGRES_LOAD_BATCH_SIZE", "1000"))


def read_csv(relative_path: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / relative_path
    if not path.exists():
        print(f"Missing {relative_path}; skipping")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def clean_symbol(symbol: str) -> str:
    return str(symbol).upper().strip().replace(".NS", "")


def yahoo_symbol(symbol: str) -> str:
    short = clean_symbol(symbol)
    return f"{short}.NS"


def none_if_nan(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def title_hash(title: str) -> str:
    normalized = " ".join(str(title).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    normalized = " ".join(str(text).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_news_embedding_text(row: dict) -> str:
    headline = str(row.get("headline", "")).strip()
    chunk_text = str(row.get("chunk_text", "")).strip()
    publisher = str(row.get("publisher", "")).strip()
    content_level = str(row.get("content_level", "")).strip()
    return (
        f"Headline: {headline}\n"
        f"Publisher: {publisher}\n"
        f"Content level: {content_level}\n"
        f"Content: {chunk_text}"
    )


def word_count(text: str) -> int:
    return len(str(text or "").split())


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    words = str(text or "").split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def load_news_content_artifact() -> dict:
    content = read_csv("clean/news_article_content.csv", parse_dates=["published_at"])
    if content.empty:
        return {}

    content_map = {}
    for _, row in content.iterrows():
        headline = row.get("headline", row.get("title"))
        key = (clean_symbol(row.get("symbol")), title_hash(headline))
        content_map[key] = {
            "resolved_url": none_if_nan(row.get("resolved_url", row.get("source_url"))),
            "article_text": none_if_nan(row.get("article_text")),
            "content_level": none_if_nan(row.get("content_level")) or "headline_only",
            "extraction_status": none_if_nan(row.get("extraction_status")) or "not_attempted",
            "word_count": none_if_nan(row.get("word_count")),
            "content_hash": none_if_nan(row.get("content_hash")),
            "source_type": none_if_nan(row.get("source_type")) or "publisher",
        }
    return content_map


def run_schema():
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        conn.commit()
    print("Schema ready")


def load_companies():
    profiles = read_csv("clean/company_profiles.csv")
    market = read_csv("clean/stock_prices.csv", usecols=["symbol"])
    if profiles.empty and market.empty:
        return {}

    if profiles.empty:
        companies = pd.DataFrame({"symbol": sorted(market["symbol"].dropna().unique())})
        companies["company_name"] = companies["symbol"].map(clean_symbol)
        companies["industry"] = None
        companies["source"] = "market data"
    else:
        companies = profiles.copy()

    companies["symbol_short"] = companies["symbol"].map(clean_symbol)
    companies["yahoo_symbol"] = companies["symbol"].map(yahoo_symbol)
    companies["company_name"] = companies["company_name"].fillna(companies["symbol_short"])
    if "industry" not in companies.columns:
        companies["industry"] = None

    rows = [
        (
            row["symbol_short"],
            row["yahoo_symbol"],
            row["company_name"],
            row.get("industry"),
            row.get("industry"),
        )
        for _, row in companies.drop_duplicates("symbol_short").iterrows()
    ]

    sql = """
        INSERT INTO core.companies (symbol, yahoo_symbol, company_name, sector, industry)
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
            yahoo_symbol = EXCLUDED.yahoo_symbol,
            company_name = EXCLUDED.company_name,
            sector = EXCLUDED.sector,
            industry = EXCLUDED.industry,
            updated_at = CURRENT_TIMESTAMP
        RETURNING company_id, symbol, yahoo_symbol
    """
    with connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
            returned = cur.fetchall()
        conn.commit()

    mapping = {symbol: company_id for company_id, symbol, _ in returned}
    mapping.update({yahoo: company_id for company_id, _, yahoo in returned})
    print(f"Loaded companies: {len(returned)}")
    return mapping


def get_company_map():
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT company_id, symbol, yahoo_symbol FROM core.companies")
            rows = cur.fetchall()
    mapping = {}
    for company_id, symbol, yahoo in rows:
        mapping[str(symbol)] = company_id
        mapping[str(yahoo)] = company_id
    return mapping


def load_stock_prices(company_map):
    df = read_csv("clean/stock_prices.csv", parse_dates=["date"])
    if df.empty:
        return
    rows = []
    for _, row in df.iterrows():
        company_id = company_map.get(yahoo_symbol(row["symbol"]))
        if not company_id:
            continue
        rows.append(
            (
                company_id,
                row["date"].date(),
                none_if_nan(row.get("open")),
                none_if_nan(row.get("high")),
                none_if_nan(row.get("low")),
                none_if_nan(row.get("close")),
                none_if_nan(row.get("adjusted_close")),
                int(row["volume"]) if pd.notna(row.get("volume")) else None,
            )
        )
    sql = """
        INSERT INTO core.stock_prices (
            company_id, trade_date, open_price, high_price, low_price,
            close_price, adjusted_close, volume
        )
        VALUES %s
        ON CONFLICT (company_id, trade_date) DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            adjusted_close = EXCLUDED.adjusted_close,
            volume = EXCLUDED.volume
    """
    insert_rows(sql, rows, "stock prices")


def load_news_articles(company_map):
    df = read_csv("clean/news_articles.csv", parse_dates=["published_at"])
    if df.empty:
        return
    content_map = load_news_content_artifact()
    rows = []
    for _, row in df.iterrows():
        company_id = company_map.get(yahoo_symbol(row["symbol"]))
        if not company_id or pd.isna(row.get("title")):
            continue
        key = (clean_symbol(row["symbol"]), title_hash(row["title"]))
        content = content_map.get(key, {})
        article_text = content.get("article_text")
        rows.append(
            (
                company_id,
                row["title"],
                none_if_nan(row.get("source")),
                none_if_nan(row.get("published_at")),
                none_if_nan(row.get("url")),
                content.get("resolved_url") or none_if_nan(row.get("url")),
                clean_symbol(row["symbol"]),
                title_hash(row["title"]),
                article_text,
                content.get("content_level") or "headline_only",
                content.get("extraction_status") or "not_attempted",
                int(content.get("word_count") or word_count(article_text)),
                content.get("content_hash") or content_hash(article_text or row["title"]),
                content.get("source_type") or "publisher",
            )
        )
    sql = """
        INSERT INTO core.news_articles (
            company_id, headline, publisher, published_at, rss_url,
            resolved_url, search_query, title_hash, article_text, content_level,
            extraction_status, word_count, content_hash, source_type
        )
        VALUES %s
        ON CONFLICT (company_id, title_hash) DO UPDATE SET
            publisher = EXCLUDED.publisher,
            published_at = EXCLUDED.published_at,
            rss_url = EXCLUDED.rss_url,
            resolved_url = EXCLUDED.resolved_url,
            article_text = EXCLUDED.article_text,
            content_level = EXCLUDED.content_level,
            extraction_status = EXCLUDED.extraction_status,
            word_count = EXCLUDED.word_count,
            content_hash = EXCLUDED.content_hash,
            source_type = EXCLUDED.source_type
    """
    insert_rows(sql, rows, "news articles")


def load_fundamentals(company_map):
    df = read_csv("clean/financial_fundamentals.csv", parse_dates=["scraped_at"])
    if df.empty:
        return
    rows = []
    for _, row in df.iterrows():
        company_id = company_map.get(yahoo_symbol(row["symbol"]))
        if not company_id:
            continue
        rows.append(
            (
                company_id,
                int(row["fiscal_year"]),
                none_if_nan(row.get("revenue_crore")),
                none_if_nan(row.get("net_income_crore")),
                none_if_nan(row.get("total_assets_crore")),
                none_if_nan(row.get("total_equity_crore")),
                none_if_nan(row.get("profit_margin")),
                none_if_nan(row.get("roa")),
                none_if_nan(row.get("roe")),
                none_if_nan(row.get("source_url")),
                none_if_nan(row.get("scraped_at")),
            )
        )
    sql = """
        INSERT INTO core.fundamentals (
            company_id, fiscal_year, revenue_crore, net_income_crore,
            total_assets_crore, total_equity_crore, profit_margin,
            roa, roe, source_url, scraped_at
        )
        VALUES %s
        ON CONFLICT (company_id, fiscal_year) DO UPDATE SET
            revenue_crore = EXCLUDED.revenue_crore,
            net_income_crore = EXCLUDED.net_income_crore,
            total_assets_crore = EXCLUDED.total_assets_crore,
            total_equity_crore = EXCLUDED.total_equity_crore,
            profit_margin = EXCLUDED.profit_margin,
            roa = EXCLUDED.roa,
            roe = EXCLUDED.roe,
            source_url = EXCLUDED.source_url,
            scraped_at = EXCLUDED.scraped_at
    """
    insert_rows(sql, rows, "fundamentals")


def load_annual_reports(company_map):
    df = read_csv("clean/annual_reports.csv", parse_dates=["scraped_at"])
    if df.empty:
        return
    rows_by_report = {}
    for _, row in df.iterrows():
        short = clean_symbol(row["symbol"])
        company_id = company_map.get(short) or company_map.get(yahoo_symbol(short))
        if not company_id:
            continue
        report_id = f"{short}_{int(row['from_year'])}_{int(row['to_year'])}"
        rows_by_report[report_id] = (
            report_id,
            company_id,
            int(row["to_year"]),
            int(row["from_year"]),
            int(row["to_year"]),
            none_if_nan(row.get("submission_type")),
            none_if_nan(row.get("pdf_url")),
            none_if_nan(row.get("declared_size")),
            none_if_nan(row.get("broadcast_at")),
            none_if_nan(row.get("scraped_at")),
            f"data/raw/annual_reports/{report_id}.pdf",
        )
    rows = list(rows_by_report.values())
    sql = """
        INSERT INTO core.annual_reports (
            report_id, company_id, financial_year, from_year, to_year,
            submission_type, pdf_url, declared_size, broadcast_at, scraped_at,
            file_path
        )
        VALUES %s
        ON CONFLICT (report_id) DO UPDATE SET
            pdf_url = EXCLUDED.pdf_url,
            declared_size = EXCLUDED.declared_size,
            broadcast_at = EXCLUDED.broadcast_at,
            scraped_at = EXCLUDED.scraped_at,
            file_path = EXCLUDED.file_path
    """
    insert_rows(sql, rows, "annual reports")


def load_market_features(company_map):
    df = read_csv("gold/market_features.csv", parse_dates=["date"])
    if df.empty:
        return
    rows = []
    for _, row in df.iterrows():
        company_id = company_map.get(yahoo_symbol(row["symbol"]))
        if not company_id:
            continue
        payload = {
            key: none_if_nan(value)
            for key, value in row.drop(labels=["symbol", "date"]).to_dict().items()
        }
        rows.append((company_id, row["date"].date(), Json(payload)))
    sql = """
        INSERT INTO analytics.market_features (company_id, trade_date, features)
        VALUES %s
        ON CONFLICT (company_id, trade_date) DO UPDATE SET
            features = EXCLUDED.features,
            generated_at = CURRENT_TIMESTAMP
    """
    insert_rows(sql, rows, "market features")


def load_sentiment_results(company_map):
    df = read_csv("gold/news_sentiment_finbert_daily.csv", parse_dates=["date"])
    if df.empty:
        df = read_csv("gold/news_sentiment_daily.csv", parse_dates=["date"])
    if df.empty:
        return
    rows = []
    for _, row in df.iterrows():
        company_id = company_map.get(yahoo_symbol(row["symbol"]))
        if not company_id:
            continue
        rows.append(
            (
                company_id,
                row["date"].date(),
                none_if_nan(row.get("article_count")),
                none_if_nan(row.get("positive_count")),
                none_if_nan(row.get("negative_count")),
                none_if_nan(row.get("neutral_count")),
                none_if_nan(row.get("average_sentiment_score")),
                none_if_nan(row.get("average_finbert_confidence")),
            )
        )
    sql = """
        INSERT INTO analytics.sentiment_results (
            company_id, sentiment_date, article_count, positive_count,
            negative_count, neutral_count, average_sentiment_score,
            average_finbert_confidence
        )
        VALUES %s
        ON CONFLICT (company_id, sentiment_date) DO UPDATE SET
            article_count = EXCLUDED.article_count,
            positive_count = EXCLUDED.positive_count,
            negative_count = EXCLUDED.negative_count,
            neutral_count = EXCLUDED.neutral_count,
            average_sentiment_score = EXCLUDED.average_sentiment_score,
            average_finbert_confidence = EXCLUDED.average_finbert_confidence
    """
    insert_rows(sql, rows, "sentiment results")


def load_risk_scores(company_map):
    df = read_csv("gold/risk_scores.csv")
    if df.empty:
        return
    rows = []
    for _, row in df.iterrows():
        company_id = company_map.get(yahoo_symbol(row["symbol"]))
        if not company_id:
            continue
        rows.append((company_id, none_if_nan(row.get("risk_score")), row.get("risk_level"), row.get("risk_reasons")))
    sql = """
        INSERT INTO analytics.risk_scores (company_id, risk_score, risk_level, risk_reasons)
        VALUES %s
        ON CONFLICT (company_id) DO UPDATE SET
            risk_score = EXCLUDED.risk_score,
            risk_level = EXCLUDED.risk_level,
            risk_reasons = EXCLUDED.risk_reasons,
            generated_at = CURRENT_TIMESTAMP
    """
    insert_rows(sql, rows, "risk scores")


def load_model_predictions(company_map):
    df = read_csv("gold/trend_signals.csv", parse_dates=["date"])
    if df.empty:
        return
    rows = []
    for _, row in df.iterrows():
        company_id = company_map.get(yahoo_symbol(row["symbol"]))
        if not company_id:
            continue
        rows.append(
            (
                company_id,
                row["date"].date(),
                row.get("trend_signal"),
                row.get("predicted_direction"),
                none_if_nan(row.get("confidence")),
                none_if_nan(row.get("probability_down")),
                none_if_nan(row.get("probability_neutral")),
                none_if_nan(row.get("probability_up")),
            )
        )
    sql = """
        INSERT INTO analytics.model_predictions (
            company_id, prediction_date, trend_signal, predicted_direction,
            confidence, probability_down, probability_neutral, probability_up
        )
        VALUES %s
        ON CONFLICT (company_id, prediction_date, model_name, model_version) DO UPDATE SET
            trend_signal = EXCLUDED.trend_signal,
            predicted_direction = EXCLUDED.predicted_direction,
            confidence = EXCLUDED.confidence,
            probability_down = EXCLUDED.probability_down,
            probability_neutral = EXCLUDED.probability_neutral,
            probability_up = EXCLUDED.probability_up,
            generated_at = CURRENT_TIMESTAMP
    """
    insert_rows(sql, rows, "model predictions")


def load_rag_evaluation_results():
    df = read_csv("gold/rag_evaluation_results.csv")
    if df.empty:
        return
    rows = [
        (
            row.get("query_id"),
            row.get("question"),
            row.get("symbol"),
            none_if_nan(row.get("report_year")),
            none_if_nan(row.get("expected_page")),
            row.get("retrieval_backend"),
            row.get("retrieved_pages"),
            bool(row.get("page_hit")),
        )
        for _, row in df.iterrows()
    ]
    sql = """
        INSERT INTO rag.evaluation_results (
            query_id, question, company_symbol, report_year, expected_page,
            retrieval_backend, retrieved_pages, page_hit
        )
        VALUES %s
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE rag.evaluation_results")
        conn.commit()
    insert_rows(sql, rows, "RAG evaluation results")


def load_report_chunks(company_map):
    chunks = read_csv("clean/report_chunks.csv")
    if chunks.empty:
        return

    import faiss

    index = faiss.read_index(str(DATA_DIR / "models" / "rag_faiss.index"))
    if index.ntotal != len(chunks):
        raise ValueError(f"FAISS vector count {index.ntotal} does not match chunks {len(chunks)}")

    rows = []
    for position, (_, row) in enumerate(chunks.iterrows()):
        short = clean_symbol(row["symbol"])
        company_id = company_map.get(short) or company_map.get(yahoo_symbol(short))
        if not company_id:
            continue
        vector = index.reconstruct(position)
        rows.append(
            (
                row["report_id"],
                company_id,
                int(row["report_year"]),
                int(row["page_start"]) if pd.notna(row.get("page_start")) else None,
                int(row["page_end"]) if pd.notna(row.get("page_end")) else None,
                none_if_nan(row.get("section_type")),
                int(row["chunk_index"]),
                row["chunk_text"],
                vector_literal(vector),
            )
        )

    sql = """
        INSERT INTO rag.annual_report_chunks (
            report_id, company_id, financial_year, page_number, page_end,
            section_name, chunk_number, chunk_text, embedding
        )
        VALUES %s
        ON CONFLICT (report_id, chunk_number) DO UPDATE SET
            page_number = EXCLUDED.page_number,
            page_end = EXCLUDED.page_end,
            section_name = EXCLUDED.section_name,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding
    """
    template = "(%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)"
    insert_rows(sql, rows, "annual report chunks + vectors", template=template)


def load_news_chunks(company_map):
    news = read_csv("clean/news_articles.csv", parse_dates=["published_at"])
    if news.empty:
        return

    sentiment = read_csv("gold/news_sentiments_finbert.csv", parse_dates=["published_at"])
    if sentiment.empty:
        sentiment = read_csv("gold/news_sentiments.csv", parse_dates=["published_at"])

    sentiment_map = {}
    if not sentiment.empty:
        for _, row in sentiment.iterrows():
            key = (clean_symbol(row.get("symbol")), title_hash(row.get("title")))
            label = row.get("finbert_label", row.get("sentiment_label"))
            sentiment_map[key] = {
                "sentiment_label": none_if_nan(label),
                "sentiment_score": none_if_nan(row.get("sentiment_score")),
            }

    article_map = {}
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    n.article_id,
                    c.symbol,
                    n.title_hash,
                    n.article_text,
                    n.content_level,
                    n.extraction_status,
                    n.word_count,
                    n.content_hash,
                    n.source_type,
                    n.resolved_url
                FROM core.news_articles n
                JOIN core.companies c USING (company_id)
                """
            )
            for (
                article_id,
                symbol,
                hash_value,
                article_text,
                content_level,
                extraction_status,
                row_word_count,
                row_content_hash,
                source_type,
                resolved_url,
            ) in cur.fetchall():
                article_map[(str(symbol), str(hash_value))] = {
                    "article_id": article_id,
                    "article_text": article_text,
                    "content_level": content_level or "headline_only",
                    "extraction_status": extraction_status or "not_attempted",
                    "word_count": row_word_count,
                    "content_hash": row_content_hash,
                    "source_type": source_type or "publisher",
                    "resolved_url": resolved_url,
                }

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required to build rag.news_chunks") from exc

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    prepared = []
    for _, row in news.iterrows():
        symbol = clean_symbol(row.get("symbol"))
        company_id = company_map.get(symbol) or company_map.get(yahoo_symbol(symbol))
        headline = str(row.get("title", "")).strip()
        if not company_id or not headline:
            continue

        hash_value = title_hash(headline)
        article_info = article_map.get((symbol, hash_value))
        if not article_info:
            continue

        article_id = article_info["article_id"]
        article_text = str(article_info.get("article_text") or "").strip()
        content_level = str(article_info.get("content_level") or "headline_only").strip()
        extraction_status = str(article_info.get("extraction_status") or "not_attempted").strip()
        source_type = str(article_info.get("source_type") or "publisher").strip()
        source_url = article_info.get("resolved_url") or none_if_nan(row.get("url"))

        if content_level in {"official_release", "full_article"} and word_count(article_text) >= 120:
            content_chunks = chunk_text(article_text)
        elif content_level in {"publisher_summary", "rss_description"} and article_text:
            content_chunks = [article_text]
        else:
            content_level = "headline_only"
            extraction_status = extraction_status if extraction_status != "success" else "headline_fallback"
            content_chunks = [headline]

        sentiment_info = sentiment_map.get((symbol, hash_value), {})
        for chunk_index, article_chunk in enumerate(content_chunks):
            prepared.append(
                {
                    "news_chunk_id": f"NEWS_{article_id}_C{chunk_index:03d}",
                    "article_id": article_id,
                    "company_id": company_id,
                    "symbol": symbol,
                    "headline": headline,
                    "chunk_text": article_chunk,
                    "chunk_index": chunk_index,
                    "publisher": none_if_nan(row.get("source")),
                    "published_at": none_if_nan(row.get("published_at")),
                    "source_url": source_url,
                    "sentiment_label": sentiment_info.get("sentiment_label"),
                    "sentiment_score": sentiment_info.get("sentiment_score"),
                    "content_level": content_level,
                    "extraction_status": extraction_status,
                    "word_count": word_count(article_chunk),
                    "source_type": source_type,
                    "content_hash": content_hash(f"{symbol}|{headline}|{source_url}|{chunk_index}|{article_chunk}"),
                }
            )

    if not prepared:
        print("No rows for news chunks + vectors")
        return

    embedding_texts = [build_news_embedding_text(row) for row in prepared]
    embeddings = model.encode(embedding_texts, normalize_embeddings=True, batch_size=64)

    rows = []
    for row, vector in zip(prepared, embeddings):
        rows.append(
            (
                row["news_chunk_id"],
                row["article_id"],
                row["company_id"],
                row["symbol"],
                row["headline"],
                row["chunk_text"],
                row["chunk_index"],
                row["publisher"],
                row["published_at"],
                row["source_url"],
                row["sentiment_label"],
                row["sentiment_score"],
                row["content_level"],
                row["extraction_status"],
                row["word_count"],
                row["source_type"],
                row["content_hash"],
                vector_literal(vector),
            )
        )

    sql = """
        INSERT INTO rag.news_chunks (
            news_chunk_id, article_id, company_id, symbol, headline, chunk_text,
            chunk_index, publisher, published_at, source_url, sentiment_label,
            sentiment_score, content_level, extraction_status, word_count,
            source_type, content_hash, embedding
        )
        VALUES %s
        ON CONFLICT (news_chunk_id) DO UPDATE SET
            headline = EXCLUDED.headline,
            chunk_text = EXCLUDED.chunk_text,
            publisher = EXCLUDED.publisher,
            published_at = EXCLUDED.published_at,
            source_url = EXCLUDED.source_url,
            sentiment_label = EXCLUDED.sentiment_label,
            sentiment_score = EXCLUDED.sentiment_score,
            content_level = EXCLUDED.content_level,
            extraction_status = EXCLUDED.extraction_status,
            word_count = EXCLUDED.word_count,
            source_type = EXCLUDED.source_type,
            content_hash = EXCLUDED.content_hash,
            embedding = EXCLUDED.embedding
    """
    template = "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)"
    insert_rows(sql, rows, "news chunks + vectors", template=template)


def insert_rows(sql: str, rows: list[tuple], label: str, template=None):
    if not rows:
        print(f"No rows for {label}")
        return
    total = 0
    with connection() as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                execute_values(cur, sql, batch, page_size=len(batch), template=template)
                total += len(batch)
                if total % (BATCH_SIZE * 10) == 0:
                    print(f"Loaded {total:,} {label}...")
        conn.commit()
    print(f"Loaded {total:,} {label}")


def print_counts():
    checks = [
        ("core.companies", "SELECT count(*) FROM core.companies"),
        ("core.stock_prices", "SELECT count(*) FROM core.stock_prices"),
        ("core.news_articles", "SELECT count(*) FROM core.news_articles"),
        ("core.fundamentals", "SELECT count(*) FROM core.fundamentals"),
        ("core.annual_reports", "SELECT count(*) FROM core.annual_reports"),
        ("analytics.market_features", "SELECT count(*) FROM analytics.market_features"),
        ("analytics.sentiment_results", "SELECT count(*) FROM analytics.sentiment_results"),
        ("analytics.risk_scores", "SELECT count(*) FROM analytics.risk_scores"),
        ("analytics.model_predictions", "SELECT count(*) FROM analytics.model_predictions"),
        ("rag.annual_report_chunks", "SELECT count(*) FROM rag.annual_report_chunks"),
        ("rag.news_chunks", "SELECT count(*) FROM rag.news_chunks"),
        ("rag.evaluation_results", "SELECT count(*) FROM rag.evaluation_results"),
    ]
    with connection() as conn:
        with conn.cursor() as cur:
            print("\nPostgreSQL load summary")
            print("=======================")
            for label, sql in checks:
                cur.execute(sql)
                print(f"{label}: {cur.fetchone()[0]:,}")


def main():
    run_schema()
    company_map = load_companies()
    if not company_map:
        company_map = get_company_map()

    load_stock_prices(company_map)
    load_news_articles(company_map)
    load_fundamentals(company_map)
    load_annual_reports(company_map)
    load_market_features(company_map)
    load_sentiment_results(company_map)
    load_risk_scores(company_map)
    load_model_predictions(company_map)
    load_rag_evaluation_results()
    load_report_chunks(company_map)
    load_news_chunks(company_map)
    print_counts()


if __name__ == "__main__":
    main()

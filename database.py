"""Primary PostgreSQL/pgvector access layer.

The dashboard uses PostgreSQL + pgvector by default. Set USE_POSTGRES=false only
for local notebook debugging, and set ALLOW_LOCAL_FALLBACK=true only when you
want the app to fall back to CSV/FAISS artifacts after a database failure.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd


DEFAULT_DATABASE_URL = "postgresql://market_user:market_password@localhost:5433/market_risk"


def postgres_enabled() -> bool:
    return os.getenv("USE_POSTGRES", "true").strip().lower() in {"1", "true", "yes", "on"}


def pgvector_enabled() -> bool:
    return os.getenv("USE_POSTGRES_RAG", "true").strip().lower() in {"1", "true", "yes", "on"}


def local_fallback_enabled() -> bool:
    return os.getenv("ALLOW_LOCAL_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def _connect():
    import psycopg2

    return psycopg2.connect(database_url())


@contextmanager
def connection():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _read_sql(query: str, params=None) -> pd.DataFrame:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [description[0] for description in cur.description]
    return pd.DataFrame(rows, columns=columns)


def _table_loaders():
    return {
        "clean/company_profiles.csv": load_company_profiles,
        "clean/stock_prices.csv": load_stock_prices,
        "clean/news_articles.csv": load_news_articles,
        "clean/financial_fundamentals.csv": load_fundamentals,
        "clean/annual_reports.csv": load_annual_reports,
        "clean/report_chunks.csv": load_report_chunks,
        "gold/market_features.csv": load_market_features,
        "gold/news_sentiment_finbert_daily.csv": load_sentiment_results,
        "gold/risk_scores.csv": load_risk_scores,
        "gold/trend_signals.csv": load_model_predictions,
        "gold/rag_evaluation_results.csv": load_rag_evaluation_results,
    }


def has_postgres_loader(relative_path: str) -> bool:
    return relative_path in _table_loaders()


def load_table(relative_path: str) -> pd.DataFrame:
    """Return a dataframe shaped like the existing CSV artifact."""
    loader = _table_loaders().get(relative_path)
    if loader is None:
        return pd.DataFrame()
    return loader()


def load_company_profiles() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            yahoo_symbol AS symbol,
            company_name,
            industry,
            created_at AS saved_at,
            'PostgreSQL core.companies' AS source
        FROM core.companies
        ORDER BY yahoo_symbol
        """
    )


def load_stock_prices() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            c.yahoo_symbol AS symbol,
            s.trade_date AS date,
            s.open_price AS open,
            s.high_price AS high,
            s.low_price AS low,
            s.close_price AS close,
            s.volume,
            s.adjusted_close
        FROM core.stock_prices s
        JOIN core.companies c USING (company_id)
        ORDER BY c.yahoo_symbol, s.trade_date
        """
    )


def load_news_articles() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            c.yahoo_symbol AS symbol,
            n.headline AS title,
            n.publisher AS source,
            COALESCE(n.resolved_url, n.rss_url) AS url,
            n.published_at
        FROM core.news_articles n
        JOIN core.companies c USING (company_id)
        ORDER BY c.yahoo_symbol, n.published_at
        """
    )


def load_fundamentals() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            c.yahoo_symbol AS symbol,
            f.fiscal_year,
            f.revenue_crore,
            f.net_income_crore,
            f.total_assets_crore,
            f.total_equity_crore,
            f.profit_margin,
            f.roa,
            f.roe,
            f.source_url,
            f.scraped_at
        FROM core.fundamentals f
        JOIN core.companies c USING (company_id)
        ORDER BY c.yahoo_symbol, f.fiscal_year
        """
    )


def load_annual_reports() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            c.symbol,
            c.company_name,
            r.from_year,
            r.to_year,
            r.submission_type,
            r.pdf_url,
            r.declared_size,
            r.broadcast_at,
            r.scraped_at
        FROM core.annual_reports r
        JOIN core.companies c USING (company_id)
        ORDER BY c.symbol, r.financial_year
        """
    )


def load_report_chunks() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            r.report_id,
            c.symbol,
            ch.financial_year AS report_year,
            ch.page_number AS page_start,
            ch.page_end,
            ch.chunk_number AS chunk_index,
            ch.chunk_text,
            ch.section_name AS section_type
        FROM rag.annual_report_chunks ch
        JOIN core.companies c ON c.company_id = ch.company_id
        JOIN core.annual_reports r ON r.report_id = ch.report_id
        ORDER BY c.symbol, ch.financial_year, ch.chunk_number
        """
    )


def load_market_features() -> pd.DataFrame:
    df = _read_sql(
        """
        SELECT
            c.yahoo_symbol AS symbol,
            m.trade_date AS date,
            m.features
        FROM analytics.market_features m
        JOIN core.companies c USING (company_id)
        ORDER BY c.yahoo_symbol, m.trade_date
        """
    )
    if df.empty:
        return df
    features = pd.json_normalize(df["features"])
    return pd.concat([df[["symbol", "date"]], features], axis=1)


def load_sentiment_results() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            c.yahoo_symbol AS symbol,
            s.sentiment_date AS date,
            s.article_count,
            s.positive_count,
            s.negative_count,
            s.neutral_count,
            s.average_sentiment_score,
            s.average_finbert_confidence
        FROM analytics.sentiment_results s
        JOIN core.companies c USING (company_id)
        ORDER BY c.yahoo_symbol, s.sentiment_date
        """
    )


def load_risk_scores() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            c.yahoo_symbol AS symbol,
            r.risk_score,
            r.risk_level,
            r.risk_reasons
        FROM analytics.risk_scores r
        JOIN core.companies c USING (company_id)
        ORDER BY c.yahoo_symbol
        """
    )


def load_model_predictions() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            c.yahoo_symbol AS symbol,
            p.prediction_date AS date,
            p.trend_signal,
            p.predicted_direction,
            p.confidence,
            p.probability_down,
            p.probability_neutral,
            p.probability_up
        FROM analytics.model_predictions p
        JOIN core.companies c USING (company_id)
        ORDER BY c.yahoo_symbol, p.prediction_date
        """
    )


def load_rag_evaluation_results() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT
            query_id,
            question,
            company_symbol AS symbol,
            report_year,
            expected_page,
            retrieval_backend,
            retrieved_pages,
            page_hit
        FROM rag.evaluation_results
        ORDER BY query_id, retrieval_backend
        """
    )


def vector_literal(vector) -> str:
    values = np.asarray(vector, dtype="float32").ravel()
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def preferred_sections_for_query(query_type: str | None, question: str = "") -> list[str]:
    text = f"{query_type or ''} {question}".lower()
    if any(term in text for term in ["revenue", "income", "profit", "asset", "equity", "financial"]):
        return ["financial_highlights", "financial_statements", "notes_to_accounts"]
    if any(term in text for term in ["risk", "threat", "uncertainty", "mitigation"]):
        return ["risk", "management_discussion"]
    if any(term in text for term in ["management", "discussion", "commentary", "outlook", "performance"]):
        return ["management_discussion"]
    if any(term in text for term in ["business", "overview", "segment", "operation", "about"]):
        return ["business_overview", "management_discussion"]
    return []


def expanded_sparse_query(question: str, query_type: str | None) -> str:
    text = str(question)
    expansions = {
        "risk": "risk risks mitigation uncertainty compliance operational regulatory",
        "management_discussion": "management discussion analysis outlook performance industry",
        "business_overview": "business overview segments operations products customers",
        "revenue": "revenue income sales financial statements",
        "profit": "profit net income financial statements",
        "assets": "assets balance sheet financial statements",
        "equity": "equity shareholders funds balance sheet financial statements",
        "fundamentals": "revenue profit assets equity financial statements",
    }
    addition = expansions.get(str(query_type or "").lower(), "")
    return f"{text} {addition}".strip()


def retrieve_chunks_pgvector(
    question: str,
    symbol: str,
    query_vector,
    top_k: int = 30,
    report_year: int | None = None,
    query_type: str | None = None,
) -> tuple[pd.DataFrame, str]:
    short_symbol = str(symbol).upper().replace(".NS", "")
    vector_text = vector_literal(query_vector)
    section_names = preferred_sections_for_query(query_type, question)
    section_names_param = section_names or ["__none__"]
    sparse_query = expanded_sparse_query(question, query_type)
    candidate_limit = max(top_k * 8, 120)
    year_filter = "AND ch.financial_year = %s" if report_year is not None else ""
    query = """
        WITH dense AS (
            SELECT
                ch.chunk_id,
                row_number() OVER (ORDER BY ch.embedding <=> %s::vector) AS dense_rank
            FROM rag.annual_report_chunks ch
            JOIN core.companies c ON c.company_id = ch.company_id
            WHERE c.symbol = %s
              AND ch.embedding IS NOT NULL
              {year_filter}
            ORDER BY ch.embedding <=> %s::vector
            LIMIT %s
        ),
        sparse AS (
            SELECT
                ch.chunk_id,
                row_number() OVER (
                    ORDER BY ts_rank_cd(ch.search_vector, plainto_tsquery('english', %s)) DESC
                ) AS sparse_rank
            FROM rag.annual_report_chunks ch
            JOIN core.companies c ON c.company_id = ch.company_id
            WHERE c.symbol = %s
              AND ch.search_vector @@ plainto_tsquery('english', %s)
              {year_filter}
            ORDER BY ts_rank_cd(ch.search_vector, plainto_tsquery('english', %s)) DESC
            LIMIT %s
        ),
        section_match AS (
            SELECT
                ch.chunk_id,
                row_number() OVER (
                    ORDER BY
                        CASE
                            WHEN ch.search_vector @@ plainto_tsquery('english', %s)
                            THEN ts_rank_cd(ch.search_vector, plainto_tsquery('english', %s))
                            ELSE 0
                        END DESC,
                        length(ch.chunk_text) DESC,
                        ch.page_number ASC
                ) AS section_rank
            FROM rag.annual_report_chunks ch
            JOIN core.companies c ON c.company_id = ch.company_id
            WHERE c.symbol = %s
              AND ch.section_name = ANY(%s::text[])
              {year_filter}
            LIMIT %s
        ),
        fused AS (
            SELECT
                COALESCE(d.chunk_id, s.chunk_id, sm.chunk_id) AS chunk_id,
                d.dense_rank,
                s.sparse_rank,
                sm.section_rank,
                COALESCE(1.0 / (60 + d.dense_rank), 0) +
                COALESCE(1.0 / (60 + s.sparse_rank), 0) +
                COALESCE(0.75 / (60 + sm.section_rank), 0) AS fusion_score
            FROM dense d
            FULL OUTER JOIN sparse s USING (chunk_id)
            FULL OUTER JOIN section_match sm ON sm.chunk_id = COALESCE(d.chunk_id, s.chunk_id)
        )
        SELECT
            ch.chunk_id,
            ch.report_id,
            c.symbol,
            ch.financial_year AS report_year,
            ch.page_number AS page_start,
            ch.page_end,
            ch.chunk_number AS chunk_index,
            ch.chunk_text,
            ch.section_name AS section_type,
            f.fusion_score AS score,
            f.dense_rank,
            f.sparse_rank,
            f.section_rank
        FROM fused f
        JOIN rag.annual_report_chunks ch ON ch.chunk_id = f.chunk_id
        JOIN core.companies c ON c.company_id = ch.company_id
        ORDER BY f.fusion_score DESC
        LIMIT %s
    """.format(year_filter=year_filter)
    params = [
        vector_text,
        short_symbol,
    ]
    if report_year is not None:
        params.append(int(report_year))
    params.extend([vector_text, candidate_limit, sparse_query, short_symbol, sparse_query])
    if report_year is not None:
        params.append(int(report_year))
    params.extend([sparse_query, candidate_limit, sparse_query, sparse_query, short_symbol, section_names_param])
    if report_year is not None:
        params.append(int(report_year))
    params.extend([candidate_limit, top_k])
    df = _read_sql(query, params=params)
    method = "PostgreSQL pgvector + full-text RRF candidates with metadata filters and section boosts"
    return df, method


def retrieve_news_chunks_pgvector(
    question: str,
    symbol: str,
    query_vector,
    days: int = 30,
    top_k: int = 30,
) -> tuple[pd.DataFrame, str]:
    short_symbol = str(symbol).upper().replace(".NS", "")
    vector_text = vector_literal(query_vector)
    sparse_query = str(question)
    candidate_limit = max(top_k * 8, 120)
    query = """
        WITH bounds AS (
            SELECT
                COALESCE(
                    max(published_at) - (%s * INTERVAL '1 day'),
                    CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
                ) AS lower_bound
            FROM rag.news_chunks
            WHERE symbol = %s
        ),
        filtered AS (
            SELECT nc.*
            FROM rag.news_chunks nc, bounds b
            WHERE nc.symbol = %s
              AND nc.published_at >= b.lower_bound
        ),
        dense AS (
            SELECT
                news_chunk_id,
                row_number() OVER (ORDER BY embedding <=> %s::vector) AS dense_rank
            FROM filtered
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        sparse AS (
            SELECT
                news_chunk_id,
                row_number() OVER (
                    ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC
                ) AS sparse_rank
            FROM filtered
            WHERE search_vector @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC
            LIMIT %s
        ),
        fresh AS (
            SELECT
                news_chunk_id,
                row_number() OVER (ORDER BY published_at DESC NULLS LAST) AS freshness_rank
            FROM filtered
            ORDER BY published_at DESC NULLS LAST
            LIMIT %s
        ),
        fused AS (
            SELECT
                COALESCE(d.news_chunk_id, s.news_chunk_id, fr.news_chunk_id) AS news_chunk_id,
                d.dense_rank,
                s.sparse_rank,
                fr.freshness_rank,
                CASE
                    WHEN nc.content_level = 'official_release' THEN 1.00
                    WHEN nc.content_level = 'full_article' THEN 0.85
                    WHEN nc.content_level = 'publisher_summary' THEN 0.65
                    WHEN nc.content_level = 'rss_description' THEN 0.50
                    ELSE 0.25
                END AS source_quality,
                COALESCE(1.0 / (60 + d.dense_rank), 0) +
                COALESCE(1.0 / (60 + s.sparse_rank), 0) +
                COALESCE(0.55 / (60 + fr.freshness_rank), 0) +
                (
                    0.35 *
                    CASE
                        WHEN nc.content_level = 'official_release' THEN 1.00
                        WHEN nc.content_level = 'full_article' THEN 0.85
                        WHEN nc.content_level = 'publisher_summary' THEN 0.65
                        WHEN nc.content_level = 'rss_description' THEN 0.50
                        ELSE 0.25
                    END / 60
                ) AS fusion_score
            FROM dense d
            FULL OUTER JOIN sparse s USING (news_chunk_id)
            FULL OUTER JOIN fresh fr ON fr.news_chunk_id = COALESCE(d.news_chunk_id, s.news_chunk_id)
            JOIN rag.news_chunks nc ON nc.news_chunk_id = COALESCE(d.news_chunk_id, s.news_chunk_id, fr.news_chunk_id)
        )
        SELECT
            nc.news_chunk_id,
            nc.article_id,
            nc.symbol,
            nc.headline,
            nc.chunk_text,
            nc.chunk_index,
            nc.publisher,
            nc.published_at,
            nc.source_url,
            nc.sentiment_label,
            nc.sentiment_score,
            nc.content_level,
            nc.extraction_status,
            nc.word_count,
            nc.source_type AS news_source_type,
            'news' AS source_type,
            f.fusion_score AS score,
            f.source_quality,
            f.dense_rank,
            f.sparse_rank,
            f.freshness_rank
        FROM fused f
        JOIN rag.news_chunks nc ON nc.news_chunk_id = f.news_chunk_id
        ORDER BY f.fusion_score DESC, nc.published_at DESC NULLS LAST
        LIMIT %s
    """
    params = (
        int(days),
        int(days),
        short_symbol,
        short_symbol,
        vector_text,
        vector_text,
        candidate_limit,
        sparse_query,
        sparse_query,
        sparse_query,
        candidate_limit,
        candidate_limit,
        top_k,
    )
    df = _read_sql(query, params=params)
    method = "PostgreSQL pgvector + full-text RRF news retrieval with freshness prior"
    return df, method

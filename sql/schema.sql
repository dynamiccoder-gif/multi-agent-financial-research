CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS core.companies (
    company_id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(30) NOT NULL UNIQUE,
    yahoo_symbol VARCHAR(30) UNIQUE,
    company_name TEXT NOT NULL,
    sector TEXT,
    industry TEXT,
    exchange VARCHAR(20) DEFAULT 'NSE',
    is_active BOOLEAN DEFAULT TRUE,
    universe_as_of_date DATE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS core.stock_prices (
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    trade_date DATE NOT NULL,
    open_price NUMERIC(18,4),
    high_price NUMERIC(18,4),
    low_price NUMERIC(18,4),
    close_price NUMERIC(18,4) NOT NULL,
    adjusted_close NUMERIC(18,4),
    volume BIGINT,
    source VARCHAR(50) DEFAULT 'Yahoo Finance',
    collected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_id, trade_date),
    CHECK (volume IS NULL OR volume >= 0),
    CHECK (high_price IS NULL OR low_price IS NULL OR high_price >= low_price)
);

CREATE TABLE IF NOT EXISTS core.news_articles (
    article_id BIGSERIAL PRIMARY KEY,
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    headline TEXT NOT NULL,
    publisher TEXT,
    published_at TIMESTAMPTZ,
    rss_url TEXT,
    resolved_url TEXT,
    search_query TEXT,
    title_hash VARCHAR(64),
    article_text TEXT,
    content_level VARCHAR(30) DEFAULT 'headline_only',
    extraction_status VARCHAR(50) DEFAULT 'not_attempted',
    word_count INTEGER,
    content_hash VARCHAR(64),
    source_type VARCHAR(50) DEFAULT 'publisher',
    collected_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (company_id, title_hash)
);

ALTER TABLE core.news_articles ADD COLUMN IF NOT EXISTS article_text TEXT;
ALTER TABLE core.news_articles ADD COLUMN IF NOT EXISTS content_level VARCHAR(30) DEFAULT 'headline_only';
ALTER TABLE core.news_articles ADD COLUMN IF NOT EXISTS extraction_status VARCHAR(50) DEFAULT 'not_attempted';
ALTER TABLE core.news_articles ADD COLUMN IF NOT EXISTS word_count INTEGER;
ALTER TABLE core.news_articles ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
ALTER TABLE core.news_articles ADD COLUMN IF NOT EXISTS source_type VARCHAR(50) DEFAULT 'publisher';

CREATE TABLE IF NOT EXISTS core.fundamentals (
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    fiscal_year INTEGER NOT NULL,
    revenue_crore NUMERIC(20,4),
    net_income_crore NUMERIC(20,4),
    total_assets_crore NUMERIC(20,4),
    total_equity_crore NUMERIC(20,4),
    profit_margin NUMERIC(18,8),
    roa NUMERIC(18,8),
    roe NUMERIC(18,8),
    source_url TEXT,
    scraped_at TIMESTAMPTZ,
    PRIMARY KEY (company_id, fiscal_year)
);

CREATE TABLE IF NOT EXISTS core.annual_reports (
    report_id TEXT PRIMARY KEY,
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    financial_year INTEGER NOT NULL,
    from_year INTEGER,
    to_year INTEGER,
    submission_type TEXT,
    pdf_url TEXT,
    declared_size TEXT,
    broadcast_at TEXT,
    scraped_at TIMESTAMPTZ,
    file_path TEXT,
    file_hash VARCHAR(64),
    page_count INTEGER,
    pages_extracted INTEGER,
    empty_pages INTEGER,
    ocr_required BOOLEAN DEFAULT FALSE,
    extraction_version TEXT,
    UNIQUE (company_id, financial_year)
);

CREATE TABLE IF NOT EXISTS analytics.market_features (
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    trade_date DATE NOT NULL,
    features JSONB NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_id, trade_date)
);

CREATE TABLE IF NOT EXISTS analytics.sentiment_results (
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    sentiment_date DATE NOT NULL,
    article_count INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    average_sentiment_score NUMERIC(18,8),
    average_finbert_confidence NUMERIC(18,8),
    generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_id, sentiment_date)
);

CREATE TABLE IF NOT EXISTS analytics.risk_scores (
    company_id BIGINT PRIMARY KEY REFERENCES core.companies(company_id),
    risk_score INTEGER,
    risk_level TEXT,
    risk_reasons TEXT,
    generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics.model_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    prediction_date DATE NOT NULL,
    trend_signal TEXT,
    predicted_direction TEXT,
    confidence NUMERIC(18,8),
    probability_down NUMERIC(18,8),
    probability_neutral NUMERIC(18,8),
    probability_up NUMERIC(18,8),
    model_name TEXT DEFAULT 'dashboard_trend_model',
    model_version TEXT DEFAULT 'local',
    generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (company_id, prediction_date, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS rag.annual_report_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES core.annual_reports(report_id),
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    financial_year INTEGER NOT NULL,
    page_number INTEGER,
    page_end INTEGER,
    section_name TEXT,
    chunk_number INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384),
    source_url TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (report_id, chunk_number)
);

ALTER TABLE rag.annual_report_chunks
ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(section_name, '') || ' ' || chunk_text)
) STORED;

CREATE INDEX IF NOT EXISTS idx_stock_prices_trade_date ON core.stock_prices(trade_date);
CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON core.news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_market_features_trade_date ON analytics.market_features(trade_date);
CREATE INDEX IF NOT EXISTS idx_chunks_company_year ON rag.annual_report_chunks(company_id, financial_year);
CREATE INDEX IF NOT EXISTS idx_chunks_fulltext ON rag.annual_report_chunks USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON rag.annual_report_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS rag.news_chunks (
    news_chunk_id TEXT PRIMARY KEY,
    article_id BIGINT REFERENCES core.news_articles(article_id),
    company_id BIGINT NOT NULL REFERENCES core.companies(company_id),
    symbol VARCHAR(30) NOT NULL,
    headline TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    publisher TEXT,
    published_at TIMESTAMPTZ,
    source_url TEXT,
    sentiment_label VARCHAR(20),
    sentiment_score DOUBLE PRECISION,
    content_level VARCHAR(30) NOT NULL,
    extraction_status VARCHAR(50) DEFAULT 'not_attempted',
    word_count INTEGER,
    source_type VARCHAR(50) DEFAULT 'publisher',
    content_hash VARCHAR(64),
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_id, chunk_index)
);

ALTER TABLE rag.news_chunks ADD COLUMN IF NOT EXISTS extraction_status VARCHAR(50) DEFAULT 'not_attempted';
ALTER TABLE rag.news_chunks ADD COLUMN IF NOT EXISTS word_count INTEGER;
ALTER TABLE rag.news_chunks ADD COLUMN IF NOT EXISTS source_type VARCHAR(50) DEFAULT 'publisher';

ALTER TABLE rag.news_chunks
ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(headline, '') || ' ' || coalesce(chunk_text, ''))
) STORED;

CREATE INDEX IF NOT EXISTS idx_news_chunks_embedding_hnsw
ON rag.news_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_news_chunks_search_gin
ON rag.news_chunks USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_news_chunks_symbol_date
ON rag.news_chunks(symbol, published_at DESC);

CREATE TABLE IF NOT EXISTS rag.retrieval_logs (
    retrieval_id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    company_id BIGINT REFERENCES core.companies(company_id),
    retrieval_method TEXT,
    dense_top_k INTEGER,
    sparse_top_k INTEGER,
    final_top_k INTEGER,
    retrieved_chunk_ids BIGINT[],
    fallback_used BOOLEAN DEFAULT FALSE,
    no_evidence BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag.evaluation_results (
    evaluation_id BIGSERIAL PRIMARY KEY,
    query_id TEXT,
    question TEXT,
    company_symbol TEXT,
    report_year INTEGER,
    expected_page INTEGER,
    retrieval_backend TEXT,
    retrieved_pages TEXT,
    page_hit BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ops.workflow_runs (
    run_id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    company_id BIGINT REFERENCES core.companies(company_id),
    router_intent TEXT,
    requested_nodes TEXT[],
    final_provider TEXT,
    fallback_used BOOLEAN DEFAULT FALSE,
    guardrail_triggered BOOLEAN DEFAULT FALSE,
    total_latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ops.node_execution_logs (
    log_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES ops.workflow_runs(run_id),
    node_name TEXT NOT NULL,
    node_status TEXT NOT NULL,
    latency_ms INTEGER,
    output_summary TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ops.data_quality_results (
    result_id BIGSERIAL PRIMARY KEY,
    pipeline_run_id TEXT,
    check_name TEXT NOT NULL,
    check_status TEXT NOT NULL,
    observed_value TEXT,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

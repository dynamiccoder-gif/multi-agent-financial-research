CREATE TABLE companies (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100),
    sector VARCHAR(50)
);

CREATE TABLE stock_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    date DATE,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume BIGINT
);
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE report_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER,
    chunk_text TEXT,
    embedding vector(384)
);

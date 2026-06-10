"""Pre-deployment checks for the Streamlit dashboard.

Run:
    python3 check_deploy_ready.py
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent

REQUIRED_FILES = [
    "app.py",
    "multi_agent.py",
    "database.py",
    "evidence_validator.py",
    "Dockerfile",
    "docker-compose.full.yml",
    "requirements-deploy.txt",
    ".streamlit/config.toml",
    "scripts/load_postgres.py",
    "sql/schema.sql",
    "data/models/rag_faiss.index",
    "data/models/trend_model.joblib",
    "data/clean/report_chunks.csv",
    "data/gold/market_features.csv",
    "data/gold/news_sentiment_finbert_daily.csv",
    "data/gold/trend_signals.csv",
    "data/gold/risk_scores.csv",
    "data/clean/company_profiles.csv",
    "data/clean/financial_fundamentals.csv",
    "data/clean/news_article_content.csv",
    "data/clean/news_articles.csv",
    "data/clean/sector_fundamentals.csv",
    "data/clean/stock_prices.csv",
]


def file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def main():
    missing = []
    large_files = []

    for relative_path in REQUIRED_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            missing.append(relative_path)
            continue
        size_mb = file_size_mb(path)
        if size_mb > 95:
            large_files.append((relative_path, size_mb))

    print("Deployment readiness check")
    print("==========================")

    if missing:
        print("\nMissing required files:")
        for path in missing:
            print(f"- {path}")
    else:
        print("\nAll required runtime files are present.")

    if large_files:
        print("\nLarge files above GitHub's normal 100 MB limit:")
        for path, size_mb in large_files:
            print(f"- {path}: {size_mb:.1f} MB")
        print("\nUse Git LFS, a Docker image registry, or external artifact storage for these files.")
    else:
        print("\nNo required file is above 95 MB.")

    if missing:
        raise SystemExit(1)

    print("\nReady for Docker deployment.")


if __name__ == "__main__":
    main()

"""Build optional full-content news evidence for News RAG.

This script starts from data/clean/news_articles.csv and writes
data/clean/news_article_content.csv. It does not bypass paywalls,
authentication, robots.txt, or publisher access controls. Failed article-body
extraction falls back to RSS description when present, otherwise headline-only
evidence.

Example:
    python3 scripts/enrich_news_content.py --limit 100
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "clean" / "news_articles.csv"
OUTPUT_PATH = DATA_DIR / "clean" / "news_article_content.csv"

USER_AGENT = "AI-Market-Risk-ResearchBot/1.0 (+educational project; respects robots.txt)"
MIN_FULL_ARTICLE_WORDS = 180
MIN_SUMMARY_WORDS = 30
REQUEST_TIMEOUT = 12
SLEEP_SECONDS = 0.4

OFFICIAL_DOMAINS = {
    "infosys.com",
    "tcs.com",
    "wipro.com",
    "hcltech.com",
    "nseindia.com",
    "bseindia.com",
}

BOILERPLATE_TERMS = {
    "cookie",
    "privacy policy",
    "subscribe",
    "sign in",
    "advertisement",
    "enable javascript",
    "terms of use",
}


def clean_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace(".NS", "").strip()


def stable_hash(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(str(text or "").split())


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text


def domain_from_url(url: str) -> str:
    netloc = urlparse(str(url or "")).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def source_type_for_url(url: str) -> str:
    domain = domain_from_url(url)
    if "nseindia.com" in domain or "bseindia.com" in domain:
        return "exchange_disclosure"
    if any(domain == official or domain.endswith("." + official) for official in OFFICIAL_DOMAINS):
        return "official"
    return "publisher"


def can_fetch_url(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    if not parsed.scheme or not parsed.netloc:
        return False

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
        return parser.can_fetch(USER_AGENT, url)
    except Exception:
        return False


def resolve_url(url: str) -> tuple[str, str]:
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        if response.url:
            return response.url, "resolved"
    except Exception:
        pass

    try:
        response = requests.get(
            url,
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        if response.url:
            response.close()
            return response.url, "resolved"
    except Exception as exc:
        return url, f"resolve_failed:{type(exc).__name__}"

    return url, "resolve_failed"


def extract_with_trafilatura(html: str, url: str) -> str:
    try:
        import trafilatura
    except ImportError:
        return ""

    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    return clean_text(extracted)


def extract_with_bs4(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    page_title = clean_text(soup.title.get_text(" ")) if soup.title else ""

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    candidates = []
    for selector in ["article", "main", "[role='main']"]:
        for node in soup.select(selector):
            text = clean_text(node.get_text(" "))
            if word_count(text) >= MIN_SUMMARY_WORDS:
                candidates.append(text)

    if not candidates:
        paragraphs = [
            clean_text(paragraph.get_text(" "))
            for paragraph in soup.find_all("p")
        ]
        candidates.append(" ".join(text for text in paragraphs if word_count(text) >= 6))

    article_text = max(candidates, key=word_count, default="")
    return clean_text(article_text), page_title


def looks_like_boilerplate(text: str) -> bool:
    lower_text = str(text or "").lower()
    if word_count(text) < MIN_SUMMARY_WORDS:
        return True
    boilerplate_hits = sum(term in lower_text for term in BOILERPLATE_TERMS)
    return boilerplate_hits >= 3


def headline_matches_content(headline: str, page_title: str, article_text: str) -> bool:
    headline_terms = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", str(headline).lower())
        if token not in {"stock", "share", "shares", "after", "with", "from", "that", "this"}
    }
    if not headline_terms:
        return True

    searchable_text = f"{page_title} {article_text[:1200]}".lower()
    matched_terms = sum(term in searchable_text for term in headline_terms)
    return matched_terms / max(len(headline_terms), 1) >= 0.25


def fetch_article(row: pd.Series) -> dict:
    headline = str(row.get("title", "")).strip()
    original_url = str(row.get("url", "")).strip()
    rss_description = clean_text(row.get("description", ""))

    resolved_url, resolve_status = resolve_url(original_url)
    source_type = source_type_for_url(resolved_url)

    if not can_fetch_url(resolved_url):
        fallback_text = rss_description or headline
        fallback_level = "rss_description" if word_count(rss_description) >= MIN_SUMMARY_WORDS else "headline_only"
        return {
            "resolved_url": resolved_url,
            "article_text": fallback_text,
            "content_level": fallback_level,
            "extraction_status": f"robots_or_access_blocked:{resolve_status}",
            "word_count": word_count(fallback_text),
            "source_type": source_type,
            "content_hash": stable_hash(fallback_text),
        }

    try:
        response = requests.get(
            resolved_url,
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except Exception as exc:
        fallback_text = rss_description or headline
        fallback_level = "rss_description" if word_count(rss_description) >= MIN_SUMMARY_WORDS else "headline_only"
        return {
            "resolved_url": resolved_url,
            "article_text": fallback_text,
            "content_level": fallback_level,
            "extraction_status": f"fetch_failed:{type(exc).__name__}",
            "word_count": word_count(fallback_text),
            "source_type": source_type,
            "content_hash": stable_hash(fallback_text),
        }

    html = response.text
    article_text = extract_with_trafilatura(html, resolved_url)
    page_title = ""
    if not article_text:
        article_text, page_title = extract_with_bs4(html)
    else:
        _, page_title = extract_with_bs4(html)

    if (
        word_count(article_text) >= MIN_FULL_ARTICLE_WORDS
        and not looks_like_boilerplate(article_text)
        and headline_matches_content(headline, page_title, article_text)
    ):
        content_level = "official_release" if source_type in {"official", "exchange_disclosure"} else "full_article"
        extraction_status = "success"
        final_text = article_text
    elif word_count(rss_description) >= MIN_SUMMARY_WORDS:
        content_level = "rss_description"
        extraction_status = "article_extraction_failed_used_rss_description"
        final_text = rss_description
    else:
        content_level = "headline_only"
        extraction_status = "article_extraction_failed_used_headline"
        final_text = headline

    return {
        "resolved_url": response.url or resolved_url,
        "article_text": clean_text(final_text),
        "content_level": content_level,
        "extraction_status": extraction_status,
        "word_count": word_count(final_text),
        "source_type": source_type,
        "content_hash": stable_hash(final_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to process; 0 means all rows.")
    parser.add_argument("--symbol", type=str, default="", help="Optional ticker filter, for example INFY.NS.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output file.")
    args = parser.parse_args()

    news = pd.read_csv(INPUT_PATH, parse_dates=["published_at"])
    if args.symbol:
        requested_symbol = clean_symbol(args.symbol)
        news = news[news["symbol"].map(clean_symbol).eq(requested_symbol)]

    existing = pd.DataFrame()
    processed_keys = set()
    if args.resume and OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        processed_keys = set(zip(existing["symbol"], existing["title_hash"]))

    rows = []
    for _, row in news.iterrows():
        title_hash = stable_hash(row.get("title"))
        key = (row.get("symbol"), title_hash)
        if key in processed_keys:
            continue

        result = fetch_article(row)
        rows.append(
            {
                "symbol": row.get("symbol"),
                "headline": row.get("title"),
                "publisher": row.get("source"),
                "published_at": row.get("published_at"),
                "source_url": row.get("url"),
                "resolved_url": result["resolved_url"],
                "article_text": result["article_text"],
                "content_level": result["content_level"],
                "extraction_status": result["extraction_status"],
                "word_count": result["word_count"],
                "source_type": result["source_type"],
                "title_hash": title_hash,
                "content_hash": result["content_hash"],
            }
        )

        print(
            f"{row.get('symbol')} | {result['content_level']} | "
            f"{result['word_count']} words | {result['extraction_status']}"
        )
        time.sleep(SLEEP_SECONDS)

        if args.limit and len(rows) >= args.limit:
            break

    output = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    output = output.drop_duplicates(["symbol", "title_hash"], keep="last")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("Rows:", len(output))
    if not output.empty:
        print(output["content_level"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

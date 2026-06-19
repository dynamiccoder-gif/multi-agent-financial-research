import pandas as pd

from evidence_validator import validate_answer_citations, validate_evidence


def sample_evidence():
    return pd.DataFrame(
        [
            {
                "chunk_id": 1,
                "symbol": "INFY",
                "report_year": 2026,
                "page_start": 134,
                "section_type": "risk",
                "chunk_text": "Infosys reports AI transformation risk and operational risk on this page.",
            },
            {
                "chunk_id": 2,
                "symbol": "INFY",
                "report_year": 2026,
                "page_start": 135,
                "section_type": "risk",
                "chunk_text": "The annual report discusses cybersecurity risk and mitigation controls.",
            },
        ]
    )


def test_evidence_validator_accepts_correct_company_year_and_pages():
    result = validate_evidence(sample_evidence(), "INFY.NS", expected_report_year=2026)

    assert result["status"] == "strong_evidence"
    assert result["valid_chunks"] == 2
    assert result["strong"] is True


def test_evidence_validator_rejects_wrong_company():
    evidence = sample_evidence()
    evidence.loc[0, "symbol"] = "TCS"

    result = validate_evidence(evidence, "INFY.NS", expected_report_year=2026)

    assert result["status"] == "weak_evidence"
    assert any("wrong_company" in issue for issue in result["issues"])


def test_citation_validator_rejects_invalid_citation_number():
    answer = "Infosys discusses AI transformation risk in its annual report [3]."

    result = validate_answer_citations(answer, sample_evidence())

    assert result["status"] == "invalid"
    assert any("Invalid citation numbers" in issue for issue in result["invalid_citations"])


def test_citation_validator_rejects_unsupported_numeric_claim_in_citation():
    answer = "Infosys reported revenue of 999999 crore in the cited evidence [1]."

    result = validate_answer_citations(answer, sample_evidence())

    assert result["status"] == "invalid"
    assert result["numeric_claim_issues"]


def test_news_evidence_validator_uses_news_metadata_not_report_pages():
    evidence = pd.DataFrame(
        [
            {
                "news_chunk_id": "NEWS_1_C000",
                "article_id": 1,
                "symbol": "INFY",
                "headline": "Infosys shares fall after sector warning",
                "chunk_text": "Infosys shares fall after sector warning",
                "publisher": "Example News",
                "published_at": "2026-06-19T05:35:00+00:00",
                "source_url": "https://example.com/infosys",
                "content_level": "headline_only",
                "source_type": "news",
            },
            {
                "news_chunk_id": "NEWS_2_C000",
                "article_id": 2,
                "symbol": "INFY",
                "headline": "Infosys IT sector sentiment remains weak",
                "chunk_text": "Infosys IT sector sentiment remains weak",
                "publisher": "Example News",
                "published_at": "2026-06-18T05:35:00+00:00",
                "source_url": "https://example.com/infosys-2",
                "content_level": "headline_only",
                "source_type": "news",
            },
        ]
    )

    result = validate_evidence(evidence, "INFY.NS", source_type="news")

    assert result["status"] == "limited_evidence"
    assert result["source_type"] == "news"


def test_news_evidence_validator_treats_full_articles_as_strong_evidence():
    evidence = pd.DataFrame(
        [
            {
                "news_chunk_id": "NEWS_1_C000",
                "article_id": 1,
                "symbol": "INFY",
                "headline": "Infosys shares fall after sector warning",
                "chunk_text": "Infosys shares fell as full article coverage discussed IT spending concerns and sector-wide selling.",
                "publisher": "Example News",
                "published_at": "2026-06-19T05:35:00+00:00",
                "source_url": "https://example.com/infosys",
                "content_level": "full_article",
                "source_type": "news",
            },
            {
                "news_chunk_id": "NEWS_2_C000",
                "article_id": 2,
                "symbol": "INFY",
                "headline": "Infosys IT sector sentiment remains weak",
                "chunk_text": "A second full article described Infosys within a broader Indian IT sector decline.",
                "publisher": "Example News",
                "published_at": "2026-06-18T05:35:00+00:00",
                "source_url": "https://example.com/infosys-2",
                "content_level": "full_article",
                "source_type": "news",
            },
        ]
    )

    result = validate_evidence(evidence, "INFY.NS", source_type="news")

    assert result["status"] == "strong_evidence"
    assert result["strong"] is True

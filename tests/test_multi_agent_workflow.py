from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent import app as agent_app, call_llm, rule_based_route  # noqa: E402


def sample_state(query):
    market = pd.DataFrame(
        [
            {
                "symbol": "INFY.NS",
                "date": "2026-06-19",
                "close": 1051.40,
                "ma_20": 1163.55,
                "volatility_20": 0.0264,
                "momentum_5": -0.034,
            }
        ]
    )
    sentiment = pd.DataFrame(
        [
            {
                "symbol": "INFY.NS",
                "date": "2026-06-19",
                "article_count": 43,
                "positive_count": 1,
                "negative_count": 33,
                "neutral_count": 9,
                "average_sentiment_score": -0.65,
            }
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "symbol": "INFY.NS",
                "fiscal_year": 2026,
                "revenue_crore": 178650,
                "net_income_crore": 29211,
                "profit_margin": 0.1635,
                "roe": 0.3131,
                "roa": 0.1873,
            }
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "symbol": "INFY.NS",
                "risk_score": 40,
                "risk_level": "Medium",
                "risk_reasons": "negative sentiment; price below MA20",
            }
        ]
    )

    def rag_func(question, ticker, route="annual_report", top_k=5):
        evidence = pd.DataFrame(
            [
                {
                    "symbol": "INFY",
                    "report_year": 2026,
                    "page_start": 134,
                    "section_type": "risk",
                    "score": 0.91,
                    "chunk_text": "AI transformation and operational risk are discussed in the annual report.",
                }
            ]
        )
        if "quantum banana" in question.lower():
            evidence["chunk_text"] = "General enterprise risk management discussion."
        return evidence, "test hybrid retrieval"

    return {
        "query": query,
        "ticker": "INFY.NS",
        "use_llm": False,
        "groq_api_key": "",
        "market_df": market,
        "sentiment_df": sentiment,
        "fundamentals_df": fundamentals,
        "risk_df": risk,
        "rag_func": rag_func,
        "tool_trace": [],
    }


def test_research_query_runs_all_core_agents():
    result = agent_app.invoke(sample_state("Analyze INFY condition and risks"))

    trace = " ".join(result["tool_trace"])

    assert result["next_agent"] == "research_agent"
    assert "Market Agent" in trace
    assert "Sentiment Agent" in trace
    assert "Fundamentals Agent" in trace
    assert "Risk Agent" in trace
    assert "RAG Agent" in trace
    assert result["confidence_score"] == 0.85


def test_condition_and_risks_routes_to_full_research_workflow():
    assert rule_based_route("Analyze HDFCLIFE condition and risks") == "research_agent"


def test_sentiment_why_routes_to_sentiment_and_news_workflow():
    assert rule_based_route("Is recent HDFCLIFE news positive or negative, and why?") == "sentiment_and_news"


def test_disclosed_risk_query_routes_to_annual_report_only():
    assert rule_based_route("What risks did INFY disclose?") == "rag_agent"


def test_personalized_investment_question_uses_research_guardrail():
    result = agent_app.invoke(sample_state("Should I invest ₹1 lakh in INFY today?"))
    answer = result["final_answer"].lower()
    trace = " ".join(result["tool_trace"])

    assert result["next_agent"] == "research_with_guardrail"
    assert result["required_agents"] == [
        "market_agent",
        "fundamentals_agent",
        "risk_agent",
        "rag_agent",
        "news_agent",
    ]
    assert "i cannot recommend whether you should invest" in answer
    assert "Guardrail Agent" in trace
    assert result["confidence_score"] <= 0.50


def test_market_only_route_does_not_leak_sentiment_feature_columns():
    state = sample_state("What is INFY price trend?")
    state["market_df"] = pd.DataFrame(
        [
            {
                "symbol": "INFY.NS",
                "date": "2026-06-19",
                "close": 1051.40,
                "ma_20": 1163.55,
                "volatility_20": 0.0264,
                "momentum_5": -0.034,
                "average_sentiment_score": -0.65,
                "sentiment_lag_1": -0.40,
                "sentiment_lag_3": -0.30,
                "sentiment_shock": -0.22,
            }
        ]
    )

    result = agent_app.invoke(state)
    answer = result["final_answer"].lower()

    assert result["next_agent"] == "market_agent"
    assert result["required_agents"] == ["market_agent"]
    assert "sentiment lag" not in answer
    assert "sentiment shock" not in answer
    assert "finbert" not in answer
    assert "average sentiment" not in answer
    assert result["structured_validation"]["status"] == "valid"


def test_langgraph_trace_order_matches_planned_validation_flow():
    result = agent_app.invoke(sample_state("Is recent INFY news positive or negative, and why?"))
    trace = result["tool_trace"]

    planner_index = next(i for i, row in enumerate(trace) if "Router / Planner" in row)
    agent_index = next(i for i, row in enumerate(trace) if "Sentiment Agent" in row)
    join_index = next(i for i, row in enumerate(trace) if "Join Agent Outputs" in row)
    pre_validation_index = next(i for i, row in enumerate(trace) if "Pre-Synthesis Validation" in row)
    synthesis_index = next(i for i, row in enumerate(trace) if "Synthesis Agent" in row)
    post_validation_index = next(i for i, row in enumerate(trace) if "Post-Synthesis Validation" in row)
    guardrail_index = next(i for i, row in enumerate(trace) if "Guardrail Agent" in row)

    assert result["required_agents"] == ["sentiment_agent", "news_agent"]
    assert planner_index < agent_index < join_index
    assert join_index < pre_validation_index < synthesis_index
    assert synthesis_index < post_validation_index < guardrail_index


def test_fundamentals_only_answer_keeps_structured_data_and_skips_citations():
    state = sample_state("What is INFY revenue and ROE?")

    result = agent_app.invoke(state)
    answer = result["final_answer"]

    assert result["next_agent"] == "fundamentals_agent"
    assert "**Fundamentals**" in answer
    assert "Revenue" in answer
    assert "No source data was available" not in answer
    assert result["citation_validation"]["status"] == "not_required"
    assert result["structured_validation"]["status"] == "valid"


def test_sentiment_and_news_without_news_evidence_does_not_invent_reason():
    def empty_news_rag(question, ticker, top_k=5):
        return pd.DataFrame(), "test empty news retrieval"

    state = sample_state("Is recent INFY news positive or negative, and why?")
    state["news_rag_func"] = empty_news_rag

    result = agent_app.invoke(state)
    answer = result["final_answer"]
    trace = " ".join(result["tool_trace"])

    assert result["next_agent"] == "sentiment_and_news"
    assert "Sentiment Agent" in trace
    assert "News RAG Agent" in trace
    assert "not reveal the events causing that tone" in answer
    assert "not inferring a reason" in answer
    assert "negative articles likely outweigh" not in answer
    assert result["confidence_score"] == 0.50
    assert result["citation_validation"]["status"] == "not_required"
    assert result["structured_validation"]["status"] == "valid"


def test_evidence_contains_page_citation_fields():
    result = agent_app.invoke(sample_state("What are INFY AI transformation risks?"))

    evidence = result["retrieved_chunks"]

    assert evidence
    assert evidence[0]["symbol"] == "INFY"
    assert evidence[0]["report_year"] == 2026
    assert evidence[0]["page_start"] == 134
    assert evidence[0]["section_type"] == "risk"


def test_weak_evidence_drops_confidence_and_names_missing_terms():
    result = agent_app.invoke(sample_state("What does INFY say about quantum banana risk in annual report?"))

    assert result["confidence_score"] == 0.25
    assert "banana" in result["missing_terms"]
    assert "quantum" in result["missing_terms"]
    assert "Not enough annual-report evidence" in result["final_answer"]


def test_structured_fallback_is_readable_not_raw_dict_dump():
    result = agent_app.invoke(sample_state("Analyze INFY condition and risks"))

    answer = result["final_answer"]

    assert "**Market Condition**" in answer
    assert "**FinBERT Sentiment**" in answer
    assert "**Fundamentals**" in answer
    assert "**Annual Report Evidence**" in answer
    assert "Source Data Used" not in answer


def test_cost_safe_mode_skips_llm_calls(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM network call should not happen when use_llm is false")

    monkeypatch.setattr("multi_agent.requests.post", fail_if_called)

    answer, provider = call_llm("test prompt", {"use_llm": False, "groq_api_key": ""})

    assert answer is None
    assert provider == "Structured fallback"


def test_groq_env_key_is_used_when_ui_key_is_blank(monkeypatch):
    captured_headers = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "env key worked"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured_headers.update(headers or {})
        return FakeResponse()

    monkeypatch.setenv("GROQ_API_KEY", "env-test-key")
    monkeypatch.setattr("multi_agent.requests.post", fake_post)

    answer, provider = call_llm("test prompt", {"use_llm": True, "groq_api_key": ""})

    assert answer == "env key worked"
    assert provider == "Groq"
    assert captured_headers["Authorization"] == "Bearer env-test-key"


def test_news_agent_headline_only_summary_answers_why():
    def news_rag_func(question, ticker, top_k=5):
        evidence = pd.DataFrame(
            [
                {
                    "news_chunk_id": "NEWS_1_C000",
                    "article_id": 1,
                    "symbol": "INFY",
                    "headline": "Why Infosys took the biggest hit after Accenture's warning",
                    "chunk_text": "Why Infosys took the biggest hit after Accenture's warning",
                    "publisher": "Example News",
                    "published_at": "2026-06-19T05:35:00+00:00",
                    "source_url": "https://example.com/1",
                    "content_level": "headline_only",
                    "source_type": "news",
                },
                {
                    "news_chunk_id": "NEWS_2_C000",
                    "article_id": 2,
                    "symbol": "INFY",
                    "headline": "TCS, Infosys and HCLTech decline as Accenture drags IT stocks",
                    "chunk_text": "TCS, Infosys and HCLTech decline as Accenture drags IT stocks",
                    "publisher": "Example News",
                    "published_at": "2026-06-19T04:10:00+00:00",
                    "source_url": "https://example.com/2",
                    "content_level": "headline_only",
                    "source_type": "news",
                },
            ]
        )
        return evidence, "test news retrieval"

    state = sample_state("Why has INFY news sentiment been negative recently?")
    state["news_rag_func"] = news_rag_func

    result = agent_app.invoke(state)

    assert result["next_agent"] == "sentiment_and_news"
    assert result["news_validation"]["status"] == "limited_evidence"
    assert result["confidence_score"] == 0.55
    assert "**Sentiment Direction**" in result["final_answer"]
    assert "Accenture warning" in result["final_answer"]
    assert "headline-only RSS evidence" in result["final_answer"]

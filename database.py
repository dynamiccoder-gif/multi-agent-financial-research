import psycopg2
from sqlalchemy import create_engine

def get_engine():
    return create_engine("postgresql://postgres:password@localhost/financial")

def load_report_chunks():
    # Load chunks into pgvector
    pass

def full_text_search(query):
    # Full-text search over report chunks
    pass

def hybrid_retrieval(query):
    # RRF hybrid retrieval
    pass

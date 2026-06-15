import pandas as pd

def build_benchmark():
    data = pd.DataFrame({
        'query': ['Q1', 'Q2'],
        'relevant_chunk': ['C1', 'C2']
    })
    data.to_csv('data/gold/rag_benchmark_v2.csv', index=False)

if __name__ == "__main__":
    build_benchmark()

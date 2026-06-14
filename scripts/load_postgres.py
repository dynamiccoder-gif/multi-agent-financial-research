import psycopg2
import pandas as pd

def load_data():
    conn = psycopg2.connect(
        host="localhost",
        database="financial",
        user="postgres",
        password="password"
    )
    # Load data logic here
    conn.close()

if __name__ == "__main__":
    load_data()

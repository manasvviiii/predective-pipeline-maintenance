import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from prefect import task, flow

# Load database credentials from .env
load_dotenv()
DB_URL = os.getenv('DB_URL')

@task(retries=3, retry_delay_seconds=10)
def extract_data():
    """Task 1: Read raw NASA telemetry data."""
    print("🚀 Extracting data from file...")
    cols = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    # Note: using r'\s+' to avoid syntax warnings
    df = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=cols)
    return df

@task
def transform_data(df):
    """Task 2: Calculate Remaining Useful Life (RUL)."""
    print("⚙️ Transforming data and calculating RUL...")
    max_cycle = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycle.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycle, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    return df.drop('max_cycle', axis=1)

@task
def load_to_sql(df):
    """Task 3: Push processed data to PostgreSQL."""
    print("📤 Loading data to PostgreSQL...")
    engine = create_engine(DB_URL)
    df.to_sql('train_labeled', engine, if_exists='replace', index=False)
    return "✅ Data successfully pushed to 'maintenance_db'!"

@flow(name="Predictive Maintenance Ingestion Pipeline")
def maintenance_pipeline():
    """The main entry point for the Prefect Flow."""
    raw_data = extract_data()
    processed_data = transform_data(raw_data)
    status = load_to_sql(processed_data)
    print(status)

if __name__ == "__main__":
    # This runs the entire pipeline
    maintenance_pipeline()

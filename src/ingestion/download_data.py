import os
import urllib.request
import pandas as pd
from src.database.db_manager import DatabaseManager
from src.database.models import RawMatch

def download_results_csv(output_path: str = 'data/raw/results.csv') -> str:
    """Download Mart Jürisoo's international results CSV if it doesn't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if os.path.exists(output_path):
        print(f"File already exists at {output_path}. Skipping download.")
        return output_path

    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    print(f"Downloading historical match results from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Successfully downloaded results to {output_path}")
    return output_path

def load_results_to_db(db_manager: DatabaseManager, csv_path: str = 'data/raw/results.csv', min_date: str = '2000-01-01'):
    """Load results from CSV into the raw_matches table, filtering by date."""
    print(f"Loading matches from {csv_path} starting from {min_date}...")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[df['date'] >= pd.to_datetime(min_date)]

    # Fill any empty scores/venues
    df['home_score'] = df['home_score'].fillna(0).astype(int)
    df['away_score'] = df['away_score'].fillna(0).astype(int)
    df['neutral'] = df['neutral'].fillna(False).astype(bool)

    db_manager.bulk_insert_matches(df)
    print(f"Successfully loaded {len(df)} matches into raw_matches table.")

def run_ingestion(min_date: str = '2000-01-01') -> DatabaseManager:
    """Orchestrate the download and database insertion pipeline."""
    db_manager = DatabaseManager()
    db_manager.init_db()

    csv_path = download_results_csv()

    if db_manager.is_empty(RawMatch):
        load_results_to_db(db_manager, csv_path, min_date=min_date)
    else:
        print("Database already contains matches. Skipping ingestion.")

    return db_manager

if __name__ == '__main__':
    # Add project root to sys.path for running directly
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    run_ingestion()

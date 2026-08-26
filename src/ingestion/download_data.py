import os
import requests
import pandas as pd
from datetime import datetime
from src.database.db_manager import DatabaseManager

def download_results_csv(output_path: str = 'data/raw/results.csv') -> str:
    """Download Mart Jürisoo's international results CSV if it doesn't exist."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}. Skipping download.")
        return output_path
        
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    print(f"Downloading historical match results from {url}...")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
        
    print(f"Successfully downloaded results to {output_path}")
    return output_path

def load_results_to_db(db_manager: DatabaseManager, csv_path: str = 'data/raw/results.csv', min_date: str = '2000-01-01'):
    """Load results from CSV into the raw_matches table, filtering by date."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Match CSV file not found at {csv_path}")
        
    print(f"Loading matches from {csv_path} starting from {min_date}...")
    df = pd.read_csv(csv_path)
    
    # Filter by date
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= pd.to_datetime(min_date)]
    
    # Convert date back to string format for DB ingestion parsing
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Fill any empty scores/venues
    df['home_score'] = df['home_score'].fillna(0).astype(int)
    df['away_score'] = df['away_score'].fillna(0).astype(int)
    df['neutral'] = df['neutral'].fillna(False).astype(bool)
    
    total_matches = len(df)
    print(f"Found {total_matches} matches. Injecting into database...")
    
    # Insert in chunks: a single bulk insert of ~25k rows would exceed the
    # backend's bound-parameter limit.
    chunk_size = 500
    for i in range(0, total_matches, chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        db_manager.bulk_insert_matches(chunk)
        
    print(f"Successfully loaded {total_matches} matches into raw_matches table.")

def run_ingestion(min_date: str = '2000-01-01') -> DatabaseManager:
    """Orchestrate the download and database insertion pipeline."""
    db_manager = DatabaseManager()
    db_manager.init_db()
    
    csv_path = download_results_csv()
    
    # Check if matches table already has data to avoid duplicating
    with db_manager:
        session = db_manager.get_session()
        from src.database.models import RawMatch
        count = session.query(RawMatch).count()
        if count > 0:
            print(f"Database already contains {count} matches. Skipping ingestion.")
        else:
            load_results_to_db(db_manager, csv_path, min_date=min_date)
            
    return db_manager

if __name__ == '__main__':
    # Add project root to sys.path for running directly
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    run_ingestion()


import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.rag.core.config import SQLITE_DB_PATH
from scripts.rag.run_full_indexing import run_full_pipeline

def run_clean_indexing():
    print(f"Project Root: {ROOT_DIR}")
    print(f"Database Path: {SQLITE_DB_PATH}")
    
    # 1. Delete existing DB to avoid mixing chunk strategies
    if SQLITE_DB_PATH.exists():
        try:
            os.remove(SQLITE_DB_PATH)
            print(f"SUCCESS: Deleted old database at {SQLITE_DB_PATH}")
        except Exception as e:
            print(f"ERROR: Could not delete database: {e}")
            print("Please ensure no other process is using the file (e.g. SQLite browser or running app).")
            # Try to continue? No, mixed data is bad.
            return
    else:
        print("No existing database found. Starting fresh.")
        
    # 2. Run the pipeline
    run_full_pipeline()

if __name__ == "__main__":
    run_clean_indexing()

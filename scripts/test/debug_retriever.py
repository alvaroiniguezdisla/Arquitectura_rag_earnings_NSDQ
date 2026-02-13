
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.retrieval.retriever import Retriever

def main():
    print("DEBUG RETRIEVER")
    r = Retriever()
    
    print("\n--- Companies in DB ---")
    companies = r.get_available_companies()
    print(companies)
    
    query = "outlook financial"
    ticker = "AAPL"
    
    print(f"\n--- Searching '{query}' for {ticker} (No Year Filter) ---")
    results = r.search(query, top_k=5, filter_company=ticker)
    for c in results:
        print(f"[{c.score:.4f}] {c.metadata}")
        
    print(f"\n--- Searching '{query}' for {ticker} (Year=2020) ---")
    results_2020 = r.search(query, top_k=5, filter_company=ticker, filter_year=2020)
    for c in results_2020:
        print(f"[{c.score:.4f}] {c.metadata}")

    if not results_2020:
        print("NO RESULTADOS PARA 2020.")

if __name__ == "__main__":
    main()

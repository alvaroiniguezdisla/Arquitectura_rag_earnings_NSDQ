import argparse
import json
import sys
from pathlib import Path

# Add src to python path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.ingest_transcripts import extract_metadata, clean_text, generate_doc_id

def prepare_corpus(root_dir, output_file, years_filter, max_docs=None):
    root_path = Path(root_dir)
    output_path = Path(output_file)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_years = [int(y) for y in years_filter.split(",")]
    
    # Check for 'Transcripts' subdirectory (common in this dataset)
    if (root_path / "Transcripts").exists() and (root_path / "Transcripts").is_dir():
        print(f"Found 'Transcripts' subdirectory, using it as root.")
        root_path = root_path / "Transcripts"

    print(f"Preparing corpus...")
    print(f"Source: {root_path}")
    print(f"Output: {output_file}")
    print(f"Years: {allowed_years}")
    
    stats = {
        "saved": 0,
        "excluded_year": 0,
        "excluded_size": 0,
        "excluded_error": 0
    }

    companies = [d for d in root_path.iterdir() if d.is_dir()]
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for company in companies:
            if max_docs and stats["saved"] >= max_docs:
                break
                
            txt_files = list(company.glob("*.txt"))
            
            for f in txt_files:
                if max_docs and stats["saved"] >= max_docs:
                    break

                try:
                    metadata = extract_metadata(f, company.name)
                    
                    if not metadata or metadata['year'] not in allowed_years:
                        stats["excluded_year"] += 1
                        continue

                    with open(f, 'r', encoding='utf-8', errors='ignore') as file_content:
                        raw_text = file_content.read()
                    
                    if len(raw_text) < 200:
                        stats["excluded_size"] += 1
                        continue
                        
                    clean_content = clean_text(raw_text)
                    doc_id = generate_doc_id(company.name, f.name)
                    
                    doc = {
                        "doc_id": doc_id,
                        "text": clean_content,
                        "metadata": metadata
                    }
                    
                    out_f.write(json.dumps(doc) + "\n")
                    stats["saved"] += 1
                    
                except Exception as e:
                    print(f"Error processing {f}: {e}")
                    stats["excluded_error"] += 1

    print("\nProcessing Complete.")
    print(f"Docs Saved: {stats['saved']}")
    print(f"Excluded (Year): {stats['excluded_year']}")
    print(f"Excluded (Size < 200 chars): {stats['excluded_size']}")
    print(f"Excluded (Errors): {stats['excluded_error']}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and filter transcripts.")
    parser.add_argument("--root", default="data/raw", help="Path to raw data.")
    parser.add_argument("--out", default="data/processed/corpus.jsonl", help="Path to output JSONL.")
    parser.add_argument("--years", default="2019,2020", help="Comma-separated years to include.")
    parser.add_argument("--max_docs", type=int, default=None, help="Max docs to process (for testing).")
    
    args = parser.parse_args()
    
    prepare_corpus(args.root, args.out, args.years, args.max_docs)

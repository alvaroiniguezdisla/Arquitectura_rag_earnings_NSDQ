import re
import hashlib
from pathlib import Path

def clean_text(text):
    """
    Normalizes whitespace and basic text cleanup.
    """
    if not text:
        return ""
    # Replace multiple newlines with a single newline, but keep paragraph breaks?
    # For RAG, distinct paragraphs are good.
    # Let's reduce multiple spaces to one, and normalize newlines.
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_metadata(file_path, company_name):
    """
    Extracts metadata from filename and content.
    """
    filename = file_path.name
    content_snippet = ""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read first 30 lines for year inference
            lines = [next(f) for _ in range(30)]
            content_snippet = " ".join(lines)
    except StopIteration:
        pass
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 1. Year Extraction
    # Try filename first (e.g. "TRANSCRIPT-2019-...")
    year_match = re.search(r'(2016|2017|2018|2019|2020)', filename)
    year = int(year_match.group(1)) if year_match else None

    if year is None:
        # Fallback to content snippet
        year_match = re.search(r'(2016|2017|2018|2019|2020)', content_snippet)
        year = int(year_match.group(1)) if year_match else None

    # 2. Quarter Extraction
    quarter_match = re.search(r'(Q1|Q2|Q3|Q4)', filename, re.IGNORECASE)
    quarter = quarter_match.group(1).upper() if quarter_match else None
    
    if quarter is None:
         quarter_match = re.search(r'(Q1|Q2|Q3|Q4)', content_snippet, re.IGNORECASE)
         quarter = quarter_match.group(1).upper() if quarter_match else None

    return {
        "company": company_name,
        "year": year,
        "quarter": quarter,
        "source_path": str(file_path),
        "doc_type": "earnings_call_transcript",
        "status": "active"
    }

def generate_doc_id(company, filename):
    """
    Generates a stable ID based on company and filename.
    """
    unique_str = f"{company}_{filename}"
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()


import os
import re
import glob
import pandas as pd
from textblob import TextBlob
from datetime import datetime

# ==========================================
# Configuración
# ==========================================
RAW_DATA_PATH = os.path.join("data", "raw", "Transcripts")
OUTPUT_PATH = os.path.join("data", "processed", "financial_features.csv")

# Diccionario simplificado de Loughran-McDonald
POSITIVE_WORDS = {
    "growth", "increase", "record", "strong", "exceeded", "confident",
    "optimistic", "momentum", "solid", "robust", "success", "higher",
    "gain", "profit", "benefit", "improved", "positive", "accelerating"
}

NEGATIVE_WORDS = {
    "decline", "loss", "weak", "lower", "decrease", "difficult",
    "challenging", "headwind", "uncertainty", "risk", "slowdown",
    "fail", "miss", "adverse", "negative", "pressure", "volatility"
}

# ==========================================
# Funciones Helpers
# ==========================================
def parse_filename(filepath):
    """Extrae metadatos del nombre del fichero: YYYY-Mon-DD-TICKER.txt"""
    basename = os.path.basename(filepath)
    try:
        # Ejemplo: 2019-Apr-30-AAPL.txt
        parts = basename.replace(".txt", "").split("-")
        date_str = "-".join(parts[:3]) # 2019-Apr-30
        ticker = parts[3]
        
        date_obj = datetime.strptime(date_str, "%Y-%b-%d")
        
        # Calcular Quarter
        quarter = (date_obj.month - 1) // 3 + 1
        return ticker, date_obj, date_obj.year, quarter
    except Exception as e:
        print(f"Error parsing filename {basename}: {e}")
        return None, None, None, None

def get_sentiment(text):
    """Calcula polaridad con TextBlob"""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_word_counts(text):
    """Cuenta palabras positivas y negativas"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    pos_count = sum(1 for word in tokens if word in POSITIVE_WORDS)
    neg_count = sum(1 for word in tokens if word in NEGATIVE_WORDS)
    total_count = len(tokens)
    return pos_count, neg_count, total_count

def extract_revenue(text):
    """
    Intenta extraer el revenue mencionado usando Regex.
    Busca patrones como: "revenue was $58.3 billion"
    Normaliza todo a Billions.
    """
    # Regex para capturar: "revenue [was/of] $XX.X [billion/million]"
    # Group 2: numero, Group 3: unidad
    pattern = r"revenue\s+(?:was|of|totaled)?\s*\$([\d\.]+)\s*(billion|million)"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        
        if unit == "million":
            amount = amount / 1000.0 # Convertir a Billions
            
        return amount
    return None

def extract_guidance(text):
    """Heurística simple para guidance"""
    text_lower = text.lower()
    if "guidance" in text_lower or "outlook" in text_lower:
        # Buscar palabras cercanas a guidance
        idx = text_lower.find("guidance") if "guidance" in text_lower else text_lower.find("outlook")
        context = text_lower[idx:idx+200] # 200 chars siguientes
        
        if any(w in context for w in ["increase", "raise", "strong", "growth"]):
            return 1
        elif any(w in context for w in ["decrease", "cut", "lower", "weak"]):
            return -1
    return 0

# ==========================================
# Main Extraction Loop
# ==========================================
def main():
    print(f"Buscando transcripts en: {RAW_DATA_PATH}")
    files = glob.glob(os.path.join(RAW_DATA_PATH, "**", "*.txt"), recursive=True)
    print(f"Encontrados {len(files)} ficheros.")
    
    data = []
    
    for filepath in files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
        # 1. Metadatos
        ticker, date_obj, year, quarter = parse_filename(filepath)
        if not ticker: continue
        
        # 2. Sentiment TextBlob
        sentiment = get_sentiment(text)
        
        # 3. Word Counts & Confidence
        pos, neg, total = get_word_counts(text)
        confidence = pos / (pos + neg + 1) # Smoothing +1
        
        # 4. Financial Specifics
        revenue = extract_revenue(text)
        guidance = extract_guidance(text)
        
        data.append({
            "ticker": ticker,
            "date": date_obj,
            "year": year,
            "quarter": quarter,
            "sentiment_score": sentiment,
            "word_count_pos": pos,
            "word_count_neg": neg,
            "word_count_total": total,
            "confidence_ratio": confidence,
            "revenue": revenue,
            "guidance_score": guidance,
            "source_file": os.path.basename(filepath)
        })
        
    # Convertir a DataFrame
    df = pd.DataFrame(data)
    
    # Manejo de nulos en revenue (imputar por media del ticker)
    df["revenue"] = df.groupby("ticker")["revenue"].transform(lambda x: x.fillna(x.mean()))
    
    # Si aún quedan nulos (porque no se encontro NINGUNO para ese ticker), rellenar con 0
    df["revenue"] = df["revenue"].fillna(0)

    # Guardar
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Features procesadas. CSV guardado en: {OUTPUT_PATH}")
    print(f"Dimensiones: {df.shape}")
    print(df.head(3))

if __name__ == "__main__":
    main()

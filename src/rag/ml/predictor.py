
import os
import re
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob
from pathlib import Path
from src.rag.core.logger import get_logger

logger = get_logger(__name__)

# ==========================================
# Constantes (Deben coincidir con entrenamiento)
# ==========================================
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

FEATURES_ORDER = [
    'sentiment_score', 
    'word_count_pos', 
    'word_count_neg', 
    'confidence_ratio', 
    'revenue', 
    'guidance_score',
    'quarter'
]

class FinancialPredictor:
    """
    Clase encargada de cargar el modelo ML y generar predicciones
    sobre el outlook financiero de una empresa basado en texto.
    """
    
    def __init__(self, models_dir: str = None):
        """
        Inicializa el predictor cargando el modelo y el scaler.
        """
        if models_dir is None:
            # Asumimos estructura: raiz/data/models
            # __file__ es src/rag/ml/predictor.py -> raiz es ../../../
            root_dir = Path(__file__).parent.parent.parent.parent
            models_dir = root_dir / "data" / "models"
            
        self.model_path = os.path.join(models_dir, "sentiment_model.pkl")
        self.scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
        
        self.model = None
        self.scaler = None
        
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                logger.info(f"Loading ML model from {self.model_path}...")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            else:
                logger.warning(f"Model artifacts not found in {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")

    def _extract_features(self, text: str, current_revenue: float, quarter: int) -> pd.DataFrame:
        """
        Extrae las features del texto usando la misma lógica que el entrenamiento.
        """
        # 1. Sentiment
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        # 2. Word Counts
        tokens = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for word in tokens if word in POSITIVE_WORDS)
        neg_count = sum(1 for word in tokens if word in NEGATIVE_WORDS)
        
        # 3. Confidence Ratio
        confidence_ratio = pos_count / (pos_count + neg_count + 1)
        
        # 4. Guidance Score
        guidance_score = 0
        text_lower = text.lower()
        if "guidance" in text_lower or "outlook" in text_lower:
            idx = text_lower.find("guidance") if "guidance" in text_lower else text_lower.find("outlook")
            context = text_lower[idx:idx+200]
            if any(w in context for w in ["increase", "raise", "strong", "growth"]):
                guidance_score = 1
            elif any(w in context for w in ["decrease", "cut", "lower", "weak"]):
                guidance_score = -1
                
        # Crear DataFrame con una fila
        features = {
            'sentiment_score': sentiment_score,
            'word_count_pos': pos_count,
            'word_count_neg': neg_count,
            'confidence_ratio': confidence_ratio,
            'revenue': current_revenue,
            'guidance_score': guidance_score,
            'quarter': quarter
        }
        
        return pd.DataFrame([features], columns=FEATURES_ORDER)

    def predict(self, text: str, current_revenue: float = 0.0, quarter: int = 1) -> dict:
        """
        Genera una predicción completa.
        
        Args:
            text: El texto del transcript (o un resumen representativo).
            current_revenue: Revenue actual (en billions). Si no se tiene, usar 0 o media.
            quarter: Trimestre fiscal (1-4).
            
        Returns:
            Dict con prediction, probability, y factors.
        """
        if not self.model or not self.scaler:
            return {"error": "Model not loaded"}
            
        # Extraer features
        X_df = self._extract_features(text, current_revenue, quarter)
        
        # Escalar
        X_scaled = self.scaler.transform(X_df)
        
        # Predecir
        prediction_class = self.model.predict(X_scaled)[0]
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        
        class_label = "POSITIVE" if prediction_class == 1 else "NEGATIVE"
        confidence = prediction_proba[1] if prediction_class == 1 else prediction_proba[0]
        
        result = {
            "prediction": class_label,
            "confidence": round(float(confidence), 2),
            "class_probabilities": {
                "negative": round(float(prediction_proba[0]), 2),
                "positive": round(float(prediction_proba[1]), 2)
            },
            "key_factors": X_df.to_dict(orient='records')[0]
        }
        
        return result

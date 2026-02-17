
import pytest
from src.rag.ml.predictor import FinancialPredictor

class TestRevenueExtraction:
    def test_extract_revenue_standard(self):
        predictor = FinancialPredictor(models_dir="dummy")
        
        text = "Apple reports revenue of $91.8 billion for the quarter."
        revenue = predictor.extract_revenue_from_text(text)
        assert revenue == 91.8
        
    def test_extract_revenue_sales(self):
        predictor = FinancialPredictor(models_dir="dummy")
        
        text = "Net sales were $50.5 billion, up 5%."
        revenue = predictor.extract_revenue_from_text(text)
        assert revenue == 50.5

    def test_extract_revenue_million(self):
        predictor = FinancialPredictor(models_dir="dummy")
        
        text = "Revenue reached $500 million."
        revenue = predictor.extract_revenue_from_text(text)
        assert revenue == 0.5 # Convert to billion

    def test_extract_revenue_no_match(self):
        predictor = FinancialPredictor(models_dir="dummy")
        
        text = "We expect strong growth next year."
        revenue = predictor.extract_revenue_from_text(text)
        assert revenue is None

    def test_extract_revenue_multiple_matches(self):
        # Should populate the first or most prominent?
        # Let's say we take the first finding for now.
        predictor = FinancialPredictor(models_dir="dummy")
        
        text = "Revenue was $10.0 billion. Operating income was $5.0 billion."
        revenue = predictor.extract_revenue_from_text(text)
        assert revenue == 10.0

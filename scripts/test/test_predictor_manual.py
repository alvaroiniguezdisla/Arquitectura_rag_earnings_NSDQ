
import sys
import os
from pathlib import Path

# Agregar root al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.ml.predictor import FinancialPredictor

def main():
    print("Test Manual del FinancialPredictor")
    
    try:
        predictor = FinancialPredictor()
        
        if not predictor.model:
            print("Error: Modelo no cargado.")
            return

        print("Modelo cargado correctamente.")
        
        # Caso de prueba: Transcript REAL (2020-Jul-30-AAPL.txt)
        real_file_path = os.path.join("data", "raw", "Transcripts", "AAPL", "2020-Jul-30-AAPL.txt")
        if os.path.exists(real_file_path):
            with open(real_file_path, "r", encoding="utf-8", errors="ignore") as f:
                real_text = f.read()
                
            print(f"\nCaso REAL (AAPL 2020-Jul-30):")
            print(f"Longitud texto: {len(real_text)} caracteres")
            
            # Revenue approx para AAPL en ese Q
            result_real = predictor.predict(real_text, current_revenue=59.7, quarter=3)
            
            print(f"Prediccion: {result_real['prediction']} ({result_real['confidence']:.2f})")
            print(f"Probabilidades: {result_real['class_probabilities']}")
            
            # Formatear features para que se vean bien
            print("Features:")
            for k, v in result_real['key_factors'].items():
                print(f"  {k}: {v}")
                
        else:
            print(f"No se encontro el fichero real: {real_file_path}")
        
    except Exception as e:
        print(f"Excepcion durante el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

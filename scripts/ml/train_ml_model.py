
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# Configuración
# ==========================================
INPUT_PATH = os.path.join("data", "processed", "financial_features.csv")
MODELS_DIR = os.path.join("data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# Features numéricas a usar
FEATURES = [
    'sentiment_score', 
    'word_count_pos', 
    'word_count_neg', 
    'confidence_ratio', 
    'revenue', 
    'guidance_score',
    'quarter'  # Importante para la estacionalidad
]

def main():
    print("Iniciando entrenamiento del modelo ML...")
    
    # 1. Cargar Datos
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: No se encuentra {INPUT_PATH}")
        return
        
    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Ingeniería del Target (Next Quarter Up?)
    print("Generando target (Next Quarter Revenue Up?)...")
    df = df.sort_values(by=['ticker', 'date'])
    
    # Shift para ver el siguiente revenue
    df['revenue_next'] = df.groupby('ticker')['revenue'].shift(-1)
    
    # Target: 1 si sube, 0 si baja o igual
    df['target'] = (df['revenue_next'] > df['revenue']).astype(int)
    
    # Eliminar las últimas filas de cada empresa (donde no sabemos el futuro)
    df_clean = df.dropna(subset=['revenue_next']).copy()
    
    print(f"   Muestras totales con target: {len(df_clean)}")
    
    # 3. Split Temporal (Train < 2020, Test = 2020)
    print("Aplicando Split Temporal (Train < 2020 / Test = 2020)...")
    
    train_mask = df_clean['year'] < 2020
    test_mask = df_clean['year'] == 2020
    
    X_train = df_clean.loc[train_mask, FEATURES]
    y_train = df_clean.loc[train_mask, 'target']
    
    X_test = df_clean.loc[test_mask, FEATURES]
    y_test = df_clean.loc[test_mask, 'target']
    
    print(f"   Train set: {X_train.shape[0]} muestras")
    print(f"   Test set:  {X_test.shape[0]} muestras")
    
    # 4. Escalar Datos
    print("Escalando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Entrenar Modelo
    print("Entrenando GradientBoostingClassifier...")
    # Usamos un learning rate bajo y pocos estimadores para evitar overfitting con pocos datos
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 6. Evaluar
    print("\nEvaluación del Modelo (Test Set 2020):")
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.2%}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 7. Guardar Artefactos
    print(f"\nGuardando modelo y scaler en {MODELS_DIR}...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Entrenamiento completado exitosamente.")

if __name__ == "__main__":
    main()

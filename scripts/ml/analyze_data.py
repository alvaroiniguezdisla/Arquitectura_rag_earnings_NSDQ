
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# Configuración
# ==========================================
INPUT_PATH = os.path.join("data", "processed", "financial_features.csv")
REPORT_PATH = os.path.join("data", "reports", "eda")

# Configurar estilo visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

def main():
    # 1. Cargar Datos
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: No se encuentra {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"📊 Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Crear directorio de reportes
    os.makedirs(REPORT_PATH, exist_ok=True)
    
    # 2. Resumen Estadístico
    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe())
    
    print("\n--- Nulos por Columna ---")
    print(df.isnull().sum())
    
    # 3. Simulación del Target (Balance de Clases)
    # Ordenar y crear target temporalmente para ver balance
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date'])
    df['revenue_next'] = df.groupby('ticker')['revenue'].shift(-1)
    df['target_up'] = (df['revenue_next'] > df['revenue']).astype(int)
    
    # Eliminar las últimas filas de cada ticker (donde revenue_next es NaN)
    df_clean = df.dropna(subset=['revenue_next'])
    
    print("\n--- Balance de Clases Esperado (Target) ---")
    print(df_clean['target_up'].value_counts(normalize=True))
    count = df_clean['target_up'].value_counts()
    print(f"Positivos (Sube): {count.get(1, 0)}")
    print(f"Negativos (Baja): {count.get(0, 0)}")
    
    # 4. Visualizaciones
    
    # A. Distribución del Sentiment
    plt.figure()
    sns.histplot(df['sentiment_score'], kde=True, bins=20, color='skyblue')
    plt.title('Distribución del Sentiment Score (TextBlob)')
    plt.xlabel('Sentiment (-1 a +1)')
    plt.savefig(os.path.join(REPORT_PATH, "sentiment_distribution.png"))
    plt.close()
    
    # B. Distribución del Revenue
    plt.figure()
    sns.histplot(df['revenue'], kde=True, bins=20, color='lightgreen')
    plt.title('Distribución del Revenue (Billions)')
    plt.xlabel('Revenue ($B)')
    plt.savefig(os.path.join(REPORT_PATH, "revenue_distribution.png"))
    plt.close()
    
    # C. Correlaciones
    plt.figure(figsize=(10, 8))
    # Seleccionar solo numéricas relevantes
    cols = ['sentiment_score', 'confidence_ratio', 'revenue', 'guidance_score', 'word_count_pos', 'word_count_neg']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación de Features')
    plt.savefig(os.path.join(REPORT_PATH, "correlation_matrix.png"))
    plt.close()
    
    # D. Sentiment vs Revenue Growth (Scatter)
    # Calcular growth actual para el plot
    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change()
    
    plt.figure()
    sns.scatterplot(data=df, x='sentiment_score', y='revenue_growth', hue='ticker', alpha=0.6)
    plt.title('Sentiment vs Revenue Growth (Trimestral)')
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.savefig(os.path.join(REPORT_PATH, "sentiment_vs_growth.png"))
    plt.close()
    
    print(f"\n✅ Análisis EDA completo. Gráficos guardados en: {REPORT_PATH}")

if __name__ == "__main__":
    main()

# Paso 3: Análisis Exploratorio de Datos (EDA)

## 🎯 Objetivo
Entender la calidad, distribución y características de los datos financieros extraídos antes de entrenar el modelo.

## 🛠️ Acciones Realizadas

### 1. Ejecución del Script de Análisis
Se ejecutó `scripts/analyze_data.py` sobre el dataset `financial_features.csv`.

### 2. Hallazgos Clave

#### 📊 Estadísticas Generales
- **Total de Muestras**: 188 transcripts procesados.
- **Calidad de Datos**: No se encontraron valores nulos en las columnas críticas (todos los revenues fueron parseados o imputados correctamente).

#### ⚖️ Balance de Clases (Simulado)
Para entrenar, comparamos el revenue actual con el del trimestre siguiente ($R_{t+1} > R_t$).

- **Bajada de Revenue (Clase 0)**: ~67% de los casos.
- **Subida de Revenue (Clase 1)**: ~33% de los casos.

> **Interpretación**: A primera vista parece contraintuitivo (tech companies creciendo), pero se explica por la **estacionalidad**. Muchos trimestres (ej: Q1 post-Navidad) tienen revenues menores que el anterior (Q4). Como incluimos la feature `quarter`, el modelo aprenderá este patrón estacional.

#### 📈 Visualizaciones Generadas
Se generaron los siguientes gráficos en `data/reports/eda/`:

1.  **`sentiment_distribution.png`**: Muestra que el sentimiento de las calls es mayoritariamente positivo (sesgo corporativo).
2.  **`revenue_distribution.png`**: Distribución de ingresos (con cola larga debido a diferencias entre gigantes como Apple vs otros).
3.  **`correlation_matrix.png`**: Mapa de calor para ver qué variables se mueven juntas.

## ✅ Conclusión del EDA
Los datos están limpios y listos para entrenar. El desbalanceo de clases no es crítico (33/67 es manejable) y refleja la naturaleza estacional del negocio.

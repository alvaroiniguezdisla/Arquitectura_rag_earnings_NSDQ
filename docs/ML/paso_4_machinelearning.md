# Paso 4: Entrenamiento y Validación del Modelo ML

## 1. Objetivo
Entrenar un modelo capaz de predecir si el revenue de una empresa subirá o bajará en el siguiente trimestre, basándose en el análisis de sentimiento del transcript actual.

## 2. Estrategia de Validación (Anti-Data Leakage)
Para garantizar que el modelo no "haga trampas" viendo el futuro, hemos implementado un **Split Temporal Estricto**:

*   **Training Set (Entrenamiento)**: Años **2016, 2017, 2018, 2019**.
    *   El modelo aprende patrones históricos de estos 4 años.
    *   Total muestras: **159**.
*   **Test Set (Validación)**: Año **2020**.
    *   Reservamos el último año completo para evaluar. El modelo NUNCA ha visto estos datos durante el entrenamiento.
    *   Total muestras: **19**.

> **Nota**: Aunque tenemos datos hasta 2020, al hacer el shift de `revenue_next`, perdemos el último trimestre de 2020 de cada empresa porque no tenemos el dato de 2021 para comparar. Por eso el test set son 19 muestras (aprox 2 por empresa).

## 3. Configuración del Modelo
*   **Algoritmo**: `GradientBoostingClassifier` (scikit-learn).
*   **Justificación**: Mantiene buen rendimiento con datasets pequeños y captura relaciones no lineales entre sentimiento y revenue.
*   **Hiperparámetros**:
    *   `n_estimators=100`: Número de árboles.
    *   `learning_rate=0.1`: Velocidad de aprendizaje.
    *   `max_depth=3`: Profundidad máxima (evita overfitting).
    *   `random_state=42`: Para reproducibilidad.
*   **Scaling**: `StandardScaler` aplicado a todas las features numéricas para que variables como `revenue` ($billions) no dominen sobre `sentiment` (0-1).

## 4. Resultados Obtenidos

### Métricas Globales
| Métrica | Valor | Interpretación |
|---|---|---|
| **Accuracy** | **73.68%** | El modelo acierta 3 de cada 4 veces. Muy superior al azar (50%). |
| **Precision (Clase 1)** | 60% | Cuando predice subida, acierta el 60% de las veces. |
| **Recall (Clase 1)** | 50% | Detecta la mitad de las subidas reales. |

### Matriz de Confusión
Sobre las 19 predicciones del año 2020:

```
[[11  2]   <-- Realmente BAJÓ (0): 11 aciertos, 2 fallos
 [ 3  3]]  <-- Realmente SUBIÓ (1): 3 fallos, 3 aciertos
```

*   **Aciertos (True Negatives + True Positives)**: 11 + 3 = **14**.
*   **Fallos**: 5.

### Interpretación del Negocio
El modelo es **conservador**: es muy bueno detectando cuando el revenue va a bajar o mantenerse (High Recal en clase 0), pero le cuesta más arriesgarse a predecir subidas (Clase 1).
Esto es útil en un contexto financiero donde "protegerse de la bajada" suele ser prioritario.

## 5. Artefactos Generados
Los siguientes ficheros se han guardado en `data/models/` para su uso en producción:

1.  `sentiment_model.pkl`: El cerebro entrenado.
2.  `feature_scaler.pkl`: La escala para normalizar nuevos datos.

## 6. Siguiente Paso
**Paso 5**: Crear el `FinancialPredictor` python class que cargue estos `.pkl` y sirva predicciones en tiempo real para el Chatbot.

# Paso 2: Extracción de Features (Ingeniería de Características)

## 🎯 Objetivo
Transformar los transcripts de texto no estructurado en un dataset tabular (`CSV`) con variables numéricas listas para el entrenamiento del modelo de Machine Learning.

## 🛠️ Acciones Realizadas

### 1. Creación del Script `extract_financial_features.py`
Se implementó un script en Python que itera sobre los 159 ficheros de transcripts y extrae información clave.

### 2. Diccionario de Features (Detalle de Lógica)

Aquí se explica paso a paso cómo se calcula cada columna y por qué es importante para el modelo:

| Feature | Tipo | Lógica de Extracción | ¿Por qué es importante? |
|---|---|---|---|
| `ticker` | Metadata | Nombre de la carpeta donde está el archivo. | Identifica la empresa (útil para agrupar y ver tendencias individuales). |
| `date` / `year` / `quarter` | Metadata | Parseado del nombre del fichero (`2019-Apr-30...`). | Permite el **Split Temporal** (entrenar con pasado, validar con futuro) y captura la estacionalidad (Q4 suele ser mejor que Q1). |
| `sentiment_score` | NLP | **Librería**: `TextBlob`.<br>**Lógica**: Asigna un valor entre -1 (muy negativo) y +1 (muy positivo) a cada frase y saca la media.<br>**Ejemplo**: "We are happy" → +0.8. | Captura el "tono" general de la llamada. Un CEO preocupado usará palabras menos positivas. |
| `word_count_pos` | NLP | **Lógica**: Cuenta palabras de una lista financiera positiva (Loughran-McDonald simplificado).<br>**Lista**: *growth, strong, record, exceeded, ...* | Mide la frecuencia de "buenas noticias". A veces el sentimiento es neutro pero usan muchas palabras de "crecimiento". |
| `word_count_neg` | NLP | **Lógica**: Cuenta palabras negativas financieras.<br>**Lista**: *decline, loss, weak, headwind, ...* | Mide la frecuencia de "malas noticias/riesgos". Es un predictor fuerte de bajadas. |
| `confidence_ratio` | Derivada | **Fórmula**: `Positivas / (Positivas + Negativas + 1)`.<br>**Rango**: 0 a 1. | Normaliza el conteo. Si un texto es muy largo tendrá muchas positivas y negativas. El ratio nos dice **qué proporción** es positiva. |
| `revenue` | Regex | **Lógica**: Busca patrones como `revenue was $58.3 billion`. Normaliza todo a Billions.<br>**Fallback**: Si falla, usa la media histórica de la empresa. | Es la métrica financiera más importante. El modelo necesita saber el volumen de negocio actual para predecir si subirá. |
| `guidance_score` | Heurística | **Lógica**: Busca la palabra "guidance" o "outlook" y mira si cerca hay palabras de subida (+1) o bajada (-1). | Captura explícitamente lo que la empresa *dice* que va a pasar. Es la feature más directa si funciona bien. |

### 3. Sobre los Datos de 2020 en el CSV
El archivo `financial_features.csv` contiene **TODO** el historial disponible (2016, 2017, 2018, 2019 y 2020).

**¿Por qué está el 2020 aquí?**
Es correcto tenerlo en el dataset "crudo". La separación (Split) se hace en el **Paso 4 (Entrenamiento)**:
- **Entrenamiento**: El script `train_ml_model.py` filtrará `year < 2020`.
- **Validación**: El script usará `year == 2020` para probar el modelo.

> **Nota Importante**: No debemos borrar el 2020 del CSV. Lo necesitamos ahí para poder medir qué tan bueno es nuestro modelo prediciendo ese año.

## ✅ Resultado
- **Archivo Generado**: `data/processed/financial_features.csv`
- **Volumen**: 188 filas procesadas (incluyendo variantes de ficheros).
- **Calidad**: El dataset contiene todas las columnas necesarias sin nulos críticos, listo para el Análisis Exploratorio (EDA).

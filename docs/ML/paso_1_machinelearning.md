# Paso 1: Instalación de Dependencias y Preparación del Entorno

## 🎯 Objetivo
Preparar el entorno de desarrollo Python con las librerías necesarias para el componente de Machine Learning y el Análisis Exploratorio de Datos (EDA).

## 🛠️ Acciones Realizadas

### 1. Actualización de `requirements.txt`
Se añadieron las siguientes librerías al archivo de dependencias del proyecto:

```text
# --- ML & EDA ---
scikit-learn>=1.3          # Algoritmos de ML (GradientBoosting)
textblob>=0.18             # Procesamiento de Lenguaje Natural (Sentiment Analysis)
pandas>=2.0                # Manipulación de datos estructurados (DataFrames)
matplotlib>=3.7            # Gráficos básicos
seaborn>=0.12              # Gráficos estadísticos avanzados
```

### 2. Instalación en el Entorno Virtual (`.venv`)
Se ejecutó la instalación dentro del entorno virtual activo para garantizar el aislamiento:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Descarga de Recursos NLP
La librería `TextBlob` requiere corpus léxicos (diccionarios) para funcionar. Se descargaron mediante el comando:

```bash
python -m textblob.download_corpora
```

## ✅ Resultado
El entorno está listo. Todas las librerías se instalaron correctamente y son accesibles desde los scripts del proyecto.

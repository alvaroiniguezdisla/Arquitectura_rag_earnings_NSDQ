# Arquitectura RAG Earnings NSDQ

Este proyecto consiste en un sistema de RAG (Retrieval-Augmented Generation) local que utiliza transcripts de earnings calls de los años 2019-2020.

## Estructura del Proyecto

- `data/raw`: Datos originales sin procesar.
- `data/processed`: Datos limpios y procesados.
- `data/indexes`: Índices vectoriales para la recuperación de información.
- `scripts`: Scripts de utilidad y ejecución.
- `src`: Código fuente del proyecto.
- `docs`: Documentación adicional.

## Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPO>
cd <NOMBRE_DEL_REPO>
```

### 2. Crear y activar entorno virtual

**Windows (PowerShell/CMD):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

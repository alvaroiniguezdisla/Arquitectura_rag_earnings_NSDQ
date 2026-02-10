# Kaggle Setup & Dataset Download

## 1. Instalación de Kaggle CLI

Para interactuar con los datasets de Kaggle, necesitas instalar la herramienta de línea de comandos:

```bash
pip install kaggle
```

## 2. Configuración de API Token

### Método Recomendado: Archivo `.env`

Este proyecto está configurado para leer las credenciales desde un archivo `.env` en la raíz del proyecto.

1. Ve a tu cuenta de Kaggle: https://www.kaggle.com/settings
2. En la sección "API", haz clic en **Create New Token**.
3. Abre el archivo `kaggle.json` descargado con un editor de texto.
4. Copia el `username` y la `key`.
5. Abre el archivo `.env` del proyecto y pega tus credenciales:

```env
KAGGLE_USERNAME=tu_usuario_real
KAGGLE_KEY=tu_clave_real
```

### Método Alternativo: Archivo `kaggle.json`

Si lo prefieres, puedes mover el archivo `kaggle.json` a la carpeta de configuración global:

**Windows:** `C:\Users\<Usuario>\.kaggle\kaggle.json`
**Linux/Mac:** `~/.kaggle/kaggle.json` (recuerda `chmod 600`)

## 3. Descarga del Dataset

Una vez configurado, puedes descargar y descomprimir el dataset de Earnings Call Transcripts con el siguiente comando:

```bash
kaggle datasets download -d ashwinm500/earnings-call-transcripts -p data/raw --unzip
```

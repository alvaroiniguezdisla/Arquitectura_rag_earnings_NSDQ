"""
LOGGER CENTRALIZADO
-------------------
Configura el logging del proyecto en un solo lugar.
Todos los módulos importan 'get_logger' de aquí.

Uso:
    from src.rag.core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Mensaje informativo")
    logger.debug("Solo visible si LOG_LEVEL=DEBUG")
    logger.error("Algo ha fallado")

Nivel por defecto: INFO
Configurable via variable de entorno LOG_LEVEL (DEBUG, INFO, WARNING, ERROR)
"""
import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Crea y devuelve un logger configurado para el módulo que lo pida.

    Args:
        name: Normalmente se pasa __name__ para identificar el módulo.

    Returns:
        Logger configurado con formato y nivel apropiados.
    """
    logger = logging.getLogger(name)

    # Solo configurar si no tiene handlers (evitar duplicados)
    if not logger.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

        # Handler de consola
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level, logging.INFO))

        # Formato: [NIVEL] nombre_modulo — mensaje
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # No propagar al root logger (evita mensajes duplicados)
        logger.propagate = False

    return logger

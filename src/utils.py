"""Helpers compartidos por todos los módulos del flujo SAND.

Concentra las convenciones del proyecto: nombres de columnas del formato SAND,
prefijos de región y funciones de normalización/comparación numérica usadas
tanto por el comparador como por el regionalizador.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura logging a consola para los módulos del proyecto (idempotente)."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                                                datefmt="%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(nivel)

# --- Convenciones del formato SAND -----------------------------------------
# Ojo: "Time indipendent variables" está escrito así (sic) en los archivos.
TIME_INDEP_COL = "Time indipendent variables"
HOJA_SAND = "Parameters"
DIMENSION_COLS = [
    "Parameter", "REGION", "TECHNOLOGY", "EMISSION", "MODE_OF_OPERATION",
    "FUEL", "TIMESLICE", "STORAGE", "REGION2",
]

# Prefijos de las 7 regiones del modelo (ver config/params_config.yaml para
# el mapeo nombre de región -> prefijo).
REGIONES = ["AN", "CA", "IN", "NE", "OR", "SE", "SO"]


# --- Normalización de valores -----------------------------------------------
def normalize_text(value: Any) -> str | None:
    """Texto limpio (strip) o None si el valor es nulo/vacío."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def safe_float(value: Any) -> float | None:
    """float o None si el valor es nulo o no convertible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round3(value: Any) -> float | None:
    """Redondeo a 3 decimales (convención de comparación de la regionalización)."""
    val = safe_float(value)
    return None if val is None else round(val, 3)


def values_close(a: Any, b: Any, tol: float = 1e-4) -> bool:
    """True si ambos valores son nulos o difieren menos que la tolerancia."""
    fa, fb = safe_float(a), safe_float(b)
    if fa is None and fb is None:
        return True
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= tol


def es_centinela(value: Any, valor_centinela: float = 99999.0, patron_9s: bool = True) -> bool:
    """True si el valor es el centinela "sin límite".

    valor_centinela: valor exacto configurado (config/params_config.yaml).
    patron_9s: además acepta cualquier entero "solo 9s" (9999, 99999, ...),
    porque los SAND existentes mezclan ambas variantes.
    """
    f = safe_float(value)
    if f is None:
        return False
    if f == valor_centinela:
        return True
    if patron_9s and f == int(f) and f != 0:
        digits = str(int(abs(f)))
        return set(digits) == {"9"}
    return False


def norm_indice(value: Any) -> str | None:
    """Normaliza un valor de índice para cruces: '1.0' -> '1', textos -> strip.

    Evita que MODE_OF_OPERATION leído como float (1.0) no cruce contra 1.
    """
    f = safe_float(value)
    if f is not None and f == int(f):
        return str(int(f))
    return normalize_text(value)


def is_nines_or_zero(value: Any) -> bool:
    """True si el valor es nulo, 0 o un patrón "solo 9s" (9999, 99999, ...).

    Regla de los parámetros condicionales: una fila nacional cuyos valores son
    todos 0/9s se trata como no aditiva (se copia igual a cada región).
    """
    f = safe_float(value)
    if f is None or f == 0.0:
        return True
    if f == int(f):
        digits = str(int(abs(f)))
        return bool(digits) and set(digits) == {"9"}
    return False


# --- Prefijos regionales -----------------------------------------------------
def split_regional_name(name: Any, regiones: list[str] | None = None) -> tuple[str | None, str | None]:
    """Separa 'CA_TRABUS' -> ('CA', 'TRABUS'); sin prefijo -> (None, valor)."""
    regiones = regiones or REGIONES
    normalized = normalize_text(name)
    if normalized is None:
        return None, None
    parts = normalized.split("_", 1)
    if len(parts) == 2 and parts[0] in regiones:
        return parts[0], parts[1]
    return None, normalized


def quitar_prefijo_region(value: Any, regiones: list[str] | None = None) -> Any:
    """Quita el prefijo de región (CA_TRABUS -> TRABUS). Deja igual si no aplica."""
    if pd.isna(value):
        return value
    _, base = split_regional_name(value, regiones)
    return base


def extraer_region(value: Any, regiones: list[str] | None = None) -> str | None:
    """Devuelve la región del prefijo (CA_TRABUS -> CA) o None si no tiene."""
    region, _ = split_regional_name(value, regiones)
    return region


def is_trade_technology(name: Any) -> bool:
    """Tecnologías TRN* = rutas de comercio entre regiones (solo existen en el Regional)."""
    normalized = normalize_text(name)
    return bool(normalized and normalized.startswith("TRN"))

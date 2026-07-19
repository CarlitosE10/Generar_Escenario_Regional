"""Lectura y escritura de archivos SAND (Excel de una hoja con parámetros apilados).

Un SAND tiene columnas de dimensión fijas (ver utils.DIMENSION_COLS), la columna
`Time indipendent variables` para parámetros sin índice YEAR y los años como
columnas anchas (2022, 2023, ...). Este módulo normaliza los encabezados a texto
para que el resto del flujo trabaje siempre con nombres de columna string.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import DIMENSION_COLS, HOJA_SAND, TIME_INDEP_COL


def cargar_sand(path: str | Path, hoja: str | int = HOJA_SAND, validar: bool = True) -> pd.DataFrame:
    """Lee un SAND y devuelve el DataFrame con encabezados normalizados a str.

    validar=True exige las columnas de dimensión estándar (los SAND reducidos
    también las traen; usar validar=False solo para archivos no estándar).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo SAND: {path}")
    df = pd.read_excel(path, sheet_name=hoja)
    df.columns = [str(c).strip() for c in df.columns]
    if validar:
        faltantes = [c for c in DIMENSION_COLS if c not in df.columns]
        if faltantes:
            raise ValueError(f"Columnas de dimensión faltantes en {path.name}: {faltantes}")
    return df


def columnas_anio(df: pd.DataFrame, hasta: int | None = None) -> list[str]:
    """Columnas de año ('2022', '2023', ...), opcionalmente hasta un año máximo."""
    cols = [c for c in df.columns if str(c).isdigit()]
    if hasta is not None:
        cols = [c for c in cols if int(c) <= hasta]
    return cols


def columnas_valor(df: pd.DataFrame, hasta: int | None = None) -> list[str]:
    """Columnas de valor: la de tiempo-independiente (si existe) + las de año."""
    cols = [TIME_INDEP_COL] if TIME_INDEP_COL in df.columns else []
    return cols + columnas_anio(df, hasta)


def a_formato_largo(
    df: pd.DataFrame,
    id_cols: list[str],
    hasta_anio: int | None = None,
    nombre_valor: str = "VALUE",
) -> pd.DataFrame:
    """Despivota las columnas de año a formato largo (YEAR int, valor numérico, NaN->0)."""
    year_cols = columnas_anio(df, hasta_anio)
    largo = df[id_cols + year_cols].melt(
        id_vars=id_cols, value_vars=year_cols, var_name="YEAR", value_name=nombre_valor
    )
    largo["YEAR"] = largo["YEAR"].astype(int)
    largo[nombre_valor] = pd.to_numeric(largo[nombre_valor], errors="coerce").fillna(0)
    return largo


def construir_sand_salida(
    valores_ancho: pd.DataFrame,
    parametro: str,
    columnas_ref: list[str],
    region_osemosys: str,
    col_dimension: str,
) -> pd.DataFrame:
    """Arma una tabla SAND con exactamente las columnas del archivo de referencia.

    valores_ancho: index = códigos regionales (ej. AN_INDCLIM), columnas = años (str).
    col_dimension: 'FUEL' o 'TECHNOLOGY' según dónde va el código regional.
    Las demás columnas de dimensión quedan vacías; si el SAND de referencia trae
    una columna indicadora 'Tiene datos...' se recalcula (1 si algún valor != 0).
    """
    salida = pd.DataFrame(index=valores_ancho.index)
    for col in columnas_ref:
        if col == "Parameter":
            salida[col] = parametro
        elif col == "REGION":
            salida[col] = region_osemosys
        elif col == col_dimension:
            salida[col] = valores_ancho.index
        elif col in valores_ancho.columns:
            salida[col] = valores_ancho[col]
        elif str(col).startswith("Tiene datos"):
            salida[col] = (valores_ancho != 0).any(axis=1).astype(int)
        else:
            salida[col] = pd.NA
    return salida.reset_index(drop=True)


def escribir_sand(df: pd.DataFrame, path: str | Path, hoja: str = HOJA_SAND) -> Path:
    """Escribe un DataFrame en formato SAND (una hoja, sin índice)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=hoja, index=False)
    return path

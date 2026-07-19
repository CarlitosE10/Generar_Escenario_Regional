"""Comparación de un escenario SAND Nacional contra el Regional (7 regiones).

La comparación se conduce por los índices que `config_depurado.yaml` (otoole)
define para cada parámetro:

- TECHNOLOGY / FUEL: el nacional se busca en el regional con prefijo de región
  (`DEMINDAUTBOI` -> `AN_DEMINDAUTBOI`, `CA_DEMINDAUTBOI`, ...).
- EMISSION / MODE_OF_OPERATION / TIMESLICE / STORAGE: coincidencia directa.
- YEAR: columnas 2022..2055; parámetros sin YEAR usan `Time indipendent variables`.

Dos rutas de validación según config/params_config.yaml:
- INTENSIVOS: cada región debe conservar el valor nacional (se compara región
  por región).
- ADITIVOS (todo lo demás): el nacional debe ser la suma de las regiones,
  excluyendo valores centinela "sin límite" (99999 / patrón solo-9s).

Además detecta anomalías estructurales (tecnologías sin correspondencia,
parámetros ausentes o vacíos, cobertura regional incompleta...).
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from sand_io import columnas_anio
from utils import (
    TIME_INDEP_COL,
    es_centinela,
    is_trade_technology,
    norm_indice,
    normalize_text,
    split_regional_name,
)

logger = logging.getLogger(__name__)

# Índices de otoole cuya coincidencia es directa (sin prefijo regional)
INDICES_DIRECTOS = ["EMISSION", "MODE_OF_OPERATION", "TIMESLICE", "STORAGE"]
INDICES_PREFIJADOS = ["TECHNOLOGY", "FUEL"]
ORDEN_COLUMNAS_ID = ["TECHNOLOGY", "FUEL", "EMISSION", "MODE_OF_OPERATION", "TIMESLICE", "STORAGE"]

MARCADOR_VACIO = "__SIN_VALOR__"


def aplicar_filtro(df: pd.DataFrame, columna: str, valores: list[str], modo: str = "exacto") -> pd.DataFrame:
    """Filtra la columna según la lista; lista vacía = no filtrar.

    modo='exacto'   -> coincidencia exacta (.isin)
    modo='contiene' -> basta con que el valor contenga alguna subcadena
    """
    if isinstance(valores, str):
        # Un string suelto se iteraría carácter por carácter ("RES" -> R|E|S)
        valores = [valores]
    if not valores:
        return df
    if modo not in ("exacto", "contiene"):
        raise ValueError(f"MODO_FILTRO inválido: {modo!r} (usar 'exacto' o 'contiene')")
    if modo == "contiene":
        patron = "|".join(re.escape(v) for v in valores)
        return df[df[columna].astype(str).str.contains(patron, na=False)]
    return df[df[columna].isin(valores)]


def preparar_regional(df_regional: pd.DataFrame, prefijos: list[str]) -> pd.DataFrame:
    """Regional con prefijos separados: TECHNOLOGY/FUEL base + columna REGION_PREFIJO.

    Las tecnologías TRN* (comercio interregional) se marcan en IS_TRN y se
    excluyen de la comparación de valores (no existen en el nacional).
    """
    df = df_regional.copy()
    tech = df["TECHNOLOGY"].apply(lambda v: split_regional_name(v, prefijos))
    fuel = df["FUEL"].apply(lambda v: split_regional_name(v, prefijos))
    df["REGION_PREFIJO"] = [t[0] or f[0] for t, f in zip(tech, fuel)]
    df["TECHNOLOGY"] = [t[1] for t in tech]
    df["FUEL"] = [f[1] for f in fuel]
    df["IS_TRN"] = df["TECHNOLOGY"].apply(is_trade_technology)
    return df


def _a_largo(df: pd.DataFrame, id_cols: list[str], tiene_year: bool,
             hasta_anio: int | None, nombre_valor: str) -> pd.DataFrame:
    """Formato largo por id_cols; con YEAR usa las columnas de año, sin YEAR usa
    la columna de tiempo-independiente (Año queda vacío)."""
    if tiene_year:
        year_cols = columnas_anio(df, hasta_anio)
        largo = df[id_cols + year_cols].melt(
            id_vars=id_cols, value_vars=year_cols, var_name="Año", value_name=nombre_valor
        )
        largo["Año"] = largo["Año"].astype(int)
    else:
        largo = df[id_cols].copy()
        # -1 = sin dimensión temporal (NaN no cruza en un merge); se limpia al final
        largo["Año"] = -1
        largo[nombre_valor] = pd.to_numeric(df[TIME_INDEP_COL], errors="coerce")
    largo[nombre_valor] = pd.to_numeric(largo[nombre_valor], errors="coerce")
    return largo.dropna(subset=[nombre_valor])


def _normalizar_ids(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Normaliza los valores de índice para el merge (1.0 -> '1', strip, NaN -> marcador)."""
    df = df.copy()
    for col in id_cols:
        df[col] = df[col].apply(norm_indice)
        df[col] = df[col].fillna(MARCADOR_VACIO)
    return df


def _anomalia(parametro: str, variable: str, tipo: str, detalle: str) -> dict:
    return {"Parametro": parametro, "Tecnologia_Fuel": variable, "Tipo_Anomalia": tipo, "Detalle": detalle}


def comparar_parametro(
    nombre: str,
    spec: dict,
    df_nac: pd.DataFrame,
    df_reg_prep: pd.DataFrame,
    cfg: dict,
    tecnologias: list[str] | None = None,
    fuels: list[str] | None = None,
    modo_filtro: str = "exacto",
) -> tuple[pd.DataFrame, list[dict]]:
    """Compara un parámetro según sus índices de otoole.

    Returns:
        comparacion: filas [Parametro, <índices>, Año, Region, Valor_Nacional,
                     Valor_Regional, Diferencia, Diferencia_Pct]
        anomalias: lista de dicts para la hoja de anomalías.
    """
    anomalias: list[dict] = []
    indices = spec["indices"]
    id_cols = [c for c in ORDEN_COLUMNAS_ID if c in indices]
    tiene_year = "YEAR" in indices
    es_intensivo = nombre in cfg["parametros_intensivos"]
    hasta_anio = cfg.get("anio_maximo_comparacion")

    nac_p = df_nac[df_nac["Parameter"] == nombre].copy()
    # Los filtros de tecnología/fuel solo aplican si el parámetro usa ese índice
    if "TECHNOLOGY" in indices:
        nac_p = aplicar_filtro(nac_p, "TECHNOLOGY", tecnologias or [], modo_filtro)
    if "FUEL" in indices:
        nac_p = aplicar_filtro(nac_p, "FUEL", fuels or [], modo_filtro)
    if nac_p.empty:
        return pd.DataFrame(), anomalias

    reg_p = df_reg_prep[(df_reg_prep["Parameter"] == nombre) & ~df_reg_prep["IS_TRN"]].copy()

    # Anomalía 2: el parámetro no existe en el SAND regional
    if reg_p.empty:
        anomalias.append(_anomalia(nombre, "-", "PARAMETRO_NO_EN_REGIONAL",
                                   "El parámetro no existe en el archivo SAND regional"))
        return pd.DataFrame(), anomalias

    # Anomalía 3: existe pero sin ningún valor
    value_cols = columnas_anio(reg_p, hasta_anio) if tiene_year else [TIME_INDEP_COL]
    if reg_p[value_cols].apply(pd.to_numeric, errors="coerce").isna().all().all():
        anomalias.append(_anomalia(nombre, "-", "PARAMETRO_SIN_VALORES",
                                   "El parámetro existe en el regional pero no tiene valores (todo vacío)"))
        return pd.DataFrame(), anomalias

    # Mismos filtros del nacional sobre el regional (códigos base, sin prefijo):
    # sin esto, anomalías calculadas del lado regional (REGIONES_INCOMPLETAS)
    # reportarían variables fuera del filtro solicitado.
    if "TECHNOLOGY" in indices:
        reg_p = aplicar_filtro(reg_p, "TECHNOLOGY", tecnologias or [], modo_filtro)
    if "FUEL" in indices:
        reg_p = aplicar_filtro(reg_p, "FUEL", fuels or [], modo_filtro)

    nac_largo = _normalizar_ids(_a_largo(nac_p, id_cols, tiene_year, hasta_anio, "Valor_Nacional"), id_cols)
    reg_largo = _normalizar_ids(
        _a_largo(reg_p, id_cols + ["REGION_PREFIJO"], tiene_year, hasta_anio, "Valor_Regional"),
        id_cols,
    )
    llaves = id_cols + ["Año"]

    # Anomalía 5: cobertura regional incompleta por combinación de índices.
    # Solo aplica a parámetros con TECHNOLOGY/FUEL (los indexados solo por
    # REGION, como DiscountRate, no llevan prefijo regional).
    n_regiones_esperado = len(cfg["prefijo_region"])
    if id_cols and (("TECHNOLOGY" in id_cols) or ("FUEL" in id_cols)):
        regiones_por_combo = reg_largo.groupby(id_cols, dropna=False)["REGION_PREFIJO"].nunique()
        incompletas = regiones_por_combo[regiones_por_combo < n_regiones_esperado]
        for combo, n in incompletas.items():
            combo = combo if isinstance(combo, tuple) else (combo,)
            variable = " | ".join(str(v) for v in combo if v != MARCADOR_VACIO)
            anomalias.append(_anomalia(nombre, variable, "REGIONES_INCOMPLETAS",
                                       f"Presente en {n} de {n_regiones_esperado} regiones"))

    if es_intensivo:
        # Cada región debe igualar el valor nacional
        comp = nac_largo.merge(reg_largo, on=llaves, how="left")
        comp["Region"] = comp["REGION_PREFIJO"].fillna("SIN_PREFIJO")
        comp = comp.drop(columns="REGION_PREFIJO")
    else:
        # Aditivo: nacional vs suma de regiones, excluyendo centinelas nacionales
        centinela = nac_largo["Valor_Nacional"].apply(
            lambda v: es_centinela(v, cfg["valor_centinela"], cfg["centinela_patron_9s"])
        )
        nac_largo = nac_largo[~centinela]
        if nac_largo.empty:
            return pd.DataFrame(), anomalias
        reg_suma = reg_largo.groupby(llaves, dropna=False, as_index=False)["Valor_Regional"].sum()
        comp = nac_largo.merge(reg_suma, on=llaves, how="left")
        comp["Region"] = "SUMA_REGIONES"

    # Anomalía 1 / 4: combinaciones nacionales sin correspondencia regional
    if id_cols:
        sin_match = comp.groupby(id_cols, dropna=False)["Valor_Regional"].apply(lambda s: s.isna().all())
        if sin_match.any():
            techs_regional = set(reg_largo["TECHNOLOGY"].unique()) if "TECHNOLOGY" in id_cols else set()
            for combo in sin_match[sin_match].index:
                combo = combo if isinstance(combo, tuple) else (combo,)
                valores = dict(zip(id_cols, combo))
                variable = " | ".join(str(v) for v in combo if v != MARCADOR_VACIO)
                if ("TECHNOLOGY" in valores and "FUEL" in valores
                        and valores["TECHNOLOGY"] in techs_regional):
                    anomalias.append(_anomalia(nombre, variable, "FUEL_NO_COINCIDE",
                                               "La tecnología existe en el regional pero el FUEL no coincide"))
                else:
                    anomalias.append(_anomalia(nombre, variable, "SIN_CORRESPONDENCIA",
                                               "Ningún XX_<código> existe en el escenario regional"))
    elif comp["Valor_Regional"].isna().all():
        anomalias.append(_anomalia(nombre, "-", "SIN_CORRESPONDENCIA",
                                   "Sin dato regional para el parámetro"))

    comp["Valor_Regional"] = comp["Valor_Regional"].fillna(0)
    comp["Diferencia"] = comp["Valor_Nacional"] - comp["Valor_Regional"]
    comp["Diferencia_Pct"] = np.where(
        comp["Valor_Nacional"] != 0, comp["Diferencia"] / comp["Valor_Nacional"].abs(), np.nan
    )
    comp.insert(0, "Parametro", nombre)
    comp["Año"] = comp["Año"].replace(-1, pd.NA)  # -1 = tiempo-independiente
    for col in id_cols:
        comp[col] = comp[col].replace(MARCADOR_VACIO, pd.NA)
    return comp, anomalias


def anomalias_codigos_globales(
    df_nacional: pd.DataFrame,
    df_reg_prep: pd.DataFrame,
    nombres: list[str],
    params_otoole: dict[str, dict],
    tecnologias: list[str] | None = None,
    fuels: list[str] | None = None,
    modo_filtro: str = "exacto",
) -> list[dict]:
    """Anomalías globales: TECHNOLOGY/FUEL del Nacional que no existe en el
    Regional (con ningún prefijo) bajo NINGUNO de los parámetros consultados.

    Complementa a SIN_CORRESPONDENCIA (que es por parámetro): aquí solo se
    reporta el código ausente en todos los parámetros consultados que lo usan,
    una única vez, con el listado de parámetros donde aparece en el Nacional.
    """
    anomalias: list[dict] = []
    reg_sub = df_reg_prep[df_reg_prep["Parameter"].isin(nombres) & ~df_reg_prep["IS_TRN"]]
    universo_reg = {
        "TECHNOLOGY": {normalize_text(v) for v in reg_sub["TECHNOLOGY"].dropna().unique()},
        "FUEL": {normalize_text(v) for v in reg_sub["FUEL"].dropna().unique()},
    }
    filtros = {"TECHNOLOGY": tecnologias or [], "FUEL": fuels or []}

    for dimension in INDICES_PREFIJADOS:
        # Parámetros consultados que usan esta dimensión y códigos nacionales
        # que aportan (aplicando el mismo filtro que la comparación)
        presencia_nac: dict[str, list[str]] = defaultdict(list)
        for nombre in nombres:
            if dimension not in params_otoole[nombre]["indices"]:
                continue
            nac_p = df_nacional[df_nacional["Parameter"] == nombre]
            nac_p = aplicar_filtro(nac_p, dimension, filtros[dimension], modo_filtro)
            for valor in nac_p[dimension].dropna().unique():
                codigo = normalize_text(valor)
                if codigo is not None:
                    presencia_nac[codigo].append(nombre)

        for codigo in sorted(set(presencia_nac) - universo_reg[dimension]):
            parametros_usan = sorted(set(presencia_nac[codigo]))
            anomalias.append(_anomalia(
                "TODOS (consultados)", codigo, f"{dimension}_NO_EN_REGIONAL",
                f"El {dimension} nacional no existe en el regional (con ningún prefijo) "
                f"para ninguno de los parámetros consultados que lo usan: "
                f"{', '.join(parametros_usan)}",
            ))
    return anomalias


def comparar_escenarios(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    params_otoole: dict[str, dict],
    cfg: dict,
    modo: str = "general",
    parametros_filtro: list[str] | None = None,
    tecnologias: list[str] | None = None,
    fuels: list[str] | None = None,
    modo_filtro: str = "exacto",
) -> dict:
    """Ejecuta la comparación completa Nacional vs Regional.

    modo: 'general' (todos los parámetros del config presentes en el nacional),
          'parametro' o 'lista_parametros' (usa parametros_filtro).

    Returns dict con:
        comparacion: cruce completo ordenado por |Diferencia| descendente
        discrepancias: filas sobre la tolerancia
        anomalias: hoja de anomalías estructurales
        resumen: métricas globales por parámetro
    """
    if modo not in ("general", "parametro", "lista_parametros"):
        raise ValueError(f"MODO_COMPARACION inválido: {modo!r}")

    prefijos = list(cfg["prefijo_region"].values())
    tolerancia = cfg["tolerancia_comparacion"]

    if modo == "general":
        nombres = [p for p in params_otoole if (df_nacional["Parameter"] == p).any()]
    else:
        nombres = list(parametros_filtro or [])
        if not nombres:
            raise ValueError("PARAMETROS_FILTRO vacío con MODO_COMPARACION != 'general'")
        desconocidos = [p for p in nombres if p not in params_otoole]
        if desconocidos:
            raise ValueError(f"Parámetros no definidos en config_depurado.yaml: {desconocidos}")

    df_reg_prep = preparar_regional(df_regional, prefijos)

    bloques, anomalias, filas_resumen = [], [], []
    for nombre in nombres:
        logger.info("Comparando %s ...", nombre)
        comp, anom = comparar_parametro(
            nombre, params_otoole[nombre], df_nacional, df_reg_prep, cfg,
            tecnologias=tecnologias, fuels=fuels, modo_filtro=modo_filtro,
        )
        anomalias.extend(anom)
        n_disc = int((comp["Diferencia"].abs() > tolerancia).sum()) if not comp.empty else 0
        filas_resumen.append({
            "Parametro": nombre,
            "Tipo": "intensivo" if nombre in cfg["parametros_intensivos"] else "aditivo",
            "Filas_Comparadas": len(comp),
            "Filas_Con_Diferencia": n_disc,
            "Max_Diferencia_Abs": float(comp["Diferencia"].abs().max()) if len(comp) else 0.0,
            "Anomalias": len(anom),
        })
        if not comp.empty:
            bloques.append(comp)

    # Anomalías globales: códigos nacionales ausentes del regional en TODOS
    # los parámetros consultados que los usan
    globales = anomalias_codigos_globales(
        df_nacional, df_reg_prep, nombres, params_otoole,
        tecnologias=tecnologias, fuels=fuels, modo_filtro=modo_filtro,
    )
    anomalias.extend(globales)

    comparacion = pd.concat(bloques, ignore_index=True) if bloques else pd.DataFrame()
    if not comparacion.empty:
        # Columnas dinámicas: solo los índices usados por los parámetros comparados
        id_presentes = [c for c in ORDEN_COLUMNAS_ID if c in comparacion.columns]
        orden = ["Parametro"] + id_presentes + ["Año", "Region",
                                                 "Valor_Nacional", "Valor_Regional",
                                                 "Diferencia", "Diferencia_Pct"]
        comparacion = comparacion[orden].sort_values(
            "Diferencia", key=lambda s: s.abs(), ascending=False
        ).reset_index(drop=True)

    discrepancias = (
        comparacion[comparacion["Diferencia"].abs() > tolerancia].reset_index(drop=True)
        if not comparacion.empty else pd.DataFrame()
    )

    resumen = pd.DataFrame(filas_resumen)
    if not resumen.empty:
        sin_diferencias = (resumen["Filas_Con_Diferencia"] == 0) & (resumen["Filas_Comparadas"] > 0)
        fila_total = pd.DataFrame([{
            "Parametro": "== TOTAL ==", "Tipo": "-",
            "Filas_Comparadas": int(resumen["Filas_Comparadas"].sum()),
            "Filas_Con_Diferencia": int(resumen["Filas_Con_Diferencia"].sum()),
            "Max_Diferencia_Abs": float(resumen["Max_Diferencia_Abs"].max()),
            "Anomalias": int(resumen["Anomalias"].sum()) + len(globales),
            "Pct_Parametros_Sin_Diferencia": round(100 * sin_diferencias.sum() / len(resumen), 1),
        }])
        resumen = pd.concat([fila_total, resumen], ignore_index=True)

    df_anomalias = pd.DataFrame(anomalias, columns=["Parametro", "Tecnologia_Fuel", "Tipo_Anomalia", "Detalle"])
    logger.info("Comparación terminada: %s filas, %s discrepancias, %s anomalías",
                len(comparacion), len(discrepancias), len(df_anomalias))
    return {"comparacion": comparacion, "discrepancias": discrepancias,
            "anomalias": df_anomalias, "resumen": resumen}


# --- Inventario estructural (se mantiene del flujo anterior) -------------------
def comparar_parametros(df_nacional: pd.DataFrame, df_regional: pd.DataFrame) -> pd.DataFrame:
    """Qué parámetros existen solo en Nacional, solo en Regional o en ambos."""
    params_nac = set(df_nacional["Parameter"].dropna().unique())
    params_reg = set(df_regional["Parameter"].dropna().unique())
    filas = []
    for p in sorted(params_nac | params_reg):
        en_nac, en_reg = p in params_nac, p in params_reg
        filas.append({
            "Parameter": p,
            "Categoria": "En ambos" if en_nac and en_reg else ("Solo Nacional" if en_nac else "Solo Regional"),
            "Filas_Nacional": int((df_nacional["Parameter"] == p).sum()) if en_nac else 0,
            "Filas_Regional": int((df_regional["Parameter"] == p).sum()) if en_reg else 0,
        })
    return pd.DataFrame(filas)


def comparar_dimension(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    dimension: str,
    regiones: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara los valores de una dimensión (TECHNOLOGY o FUEL) entre archivos.

    Returns:
        detalle: cada variable base con su categoría (solo Nacional / solo
                 Regional / TRN / en ambos) y las regiones donde aparece.
        cobertura: por variable base regional, en cuántas regiones existe y si
                 cubre las 7 (para detectar desagregaciones incompletas).
    """
    nac_values = {normalize_text(v) for v in df_nacional[dimension].dropna().unique()}
    presencia: dict[str, set] = defaultdict(set)
    reg_base, reg_trade, reg_sin_prefijo = set(), set(), set()
    for value in df_regional[dimension].dropna().unique():
        if dimension == "TECHNOLOGY" and is_trade_technology(value):
            reg_trade.add(normalize_text(value))
            continue
        region, base = split_regional_name(value, regiones)
        if base is None:
            continue
        presencia[base].add(region if region else "SIN_PREFIJO")
        (reg_base if region else reg_sin_prefijo).add(base)

    reg_all = reg_base | reg_sin_prefijo
    categorias = [
        ("Solo Nacional", sorted(nac_values - reg_all)),
        ("Solo Regional (base)", sorted(reg_base - nac_values)),
        ("Solo Regional (sin prefijo)", sorted(reg_sin_prefijo - nac_values)),
        ("Solo Regional (TRN)", sorted(reg_trade)),
        ("En ambos", sorted(nac_values & reg_all)),
    ]
    detalle = pd.DataFrame([
        {"Dimension": dimension, "Categoria": cat, "Variable": var,
         "Regiones": ", ".join(sorted(presencia.get(var, []))),
         "Num_Regiones": len(presencia.get(var, []))}
        for cat, variables in categorias for var in variables
    ])
    cobertura = pd.DataFrame([
        {"Variable_Base": base, "Regiones": ", ".join(sorted(regs)),
         "Num_Regiones": len(regs), "Cubre_Todas": len(regs & set(regiones)) == len(regiones)}
        for base, regs in sorted(presencia.items())
    ])
    return detalle, cobertura

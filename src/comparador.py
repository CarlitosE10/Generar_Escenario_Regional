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

import difflib
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sand_io import columnas_anio
from utils import (
    TIME_INDEP_COL,
    es_centinela,
    is_trade_technology,
    norm_indice,
    normalize_text,
    safe_float,
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


# --- Inventario enriquecido, análisis por sector y contextualización ----------
# Sectores del modelo detectados por contains() sobre TECHNOLOGY y FUEL
SECTORES = ["TRA", "IND", "RES", "TER", "AGF", "PWR"]

# Parámetros clave que se validan por sector (los vacíos en el SAND se saltan)
PARAMS_CLAVE_SECTOR = [
    "AccumulatedAnnualDemand",
    "TotalAnnualMaxCapacity",
    "TotalAnnualMinCapacity",
    "TotalTechnologyAnnualActivityUpperLimit",
    "TotalTechnologyAnnualActivityLowerLimit",
    "TotalTechnologyModelPeriodActivityUpperLimit",
    "TotalTechnologyModelPeriodActivityLowerLimit",
    "TotalAnnualMaxCapacityInvestment",
]

# Modos de transporte (códigos del notebook validacion_nacional_vs_regional)
MODOS_TRA = {
    "TRAAVI": "Aviación", "TRABUS": "Buses", "TRAFWD": "4x4/FWD",
    "TRALDV": "Veh. livianos", "TRAMIC": "Motocicletas", "TRAMRT": "Metro/MRT",
    "TRAMTC": "MTC", "TRAMET": "Metro eléctrico", "TRARVT": "Tren ligero",
    "TRASTT": "Barcos/STT", "TRATAX": "Taxis", "TRATCK": "Camiones",
    "TRAMAS": "MAS", "TRARES": "Resto", "TRAMAR": "Marítimo",
}

PARAM_DESCRIPCIONES = {
    "AccumulatedAnnualDemand": "Demanda acumulada anual por combustible",
    "AnnualEmissionLimit": "Límite anual de emisiones",
    "AnnualExogenousEmission": "Emisiones exógenas anuales",
    "AvailabilityFactor": "Factor de disponibilidad de tecnología",
    "CapacityFactor": "Factor de capacidad por timeslice",
    "CapacityOfOneTechnologyUnit": "Capacidad de una unidad tecnológica",
    "CapacityToActivityUnit": "Factor de conversión capacidad-actividad",
    "CapitalCost": "Costo de capital ($/kW)",
    "DepreciationMethod": "Método de depreciación",
    "DiscountRate": "Tasa de descuento global",
    "DiscountRateIdv": "Tasa de descuento por tecnología",
    "EmissionActivityRatio": "Ratio de emisión por actividad",
    "EmissionsPenalty": "Penalización por emisiones",
    "FixedCost": "Costo fijo anual",
    "InputActivityRatio": "Ratio de entrada (combustible -> tecnología)",
    "ModelPeriodEmissionLimit": "Límite de emisión del periodo del modelo",
    "ModelPeriodExogenousEmission": "Emisión exógena del periodo del modelo",
    "OperationalLife": "Vida operativa de la tecnología (años)",
    "OutputActivityRatio": "Ratio de salida (tecnología -> producto)",
    "REMinProductionTarget": "Meta mínima de producción renovable",
    "RETagFuel": "Etiqueta de combustible renovable",
    "RETagTechnology": "Etiqueta de tecnología renovable",
    "ReserveMargin": "Margen de reserva del sistema",
    "ReserveMarginTagFuel": "Etiqueta de combustible para margen de reserva",
    "ReserveMarginTagTechnology": "Etiqueta de tecnología para margen de reserva",
    "ResidualCapacity": "Capacidad residual instalada (MW)",
    "SpecifiedAnnualDemand": "Demanda anual especificada",
    "SpecifiedDemandProfile": "Perfil de demanda especificada",
    "TotalAnnualMaxCapacity": "Capacidad máxima anual total",
    "TotalAnnualMaxCapacityInvestment": "Límite máximo de inversión anual",
    "TotalAnnualMinCapacity": "Capacidad mínima anual total",
    "TotalAnnualMinCapacityInvestment": "Límite mínimo de inversión anual",
    "TotalTechnologyAnnualActivityLowerLimit": "Límite inferior de actividad anual",
    "TotalTechnologyAnnualActivityUpperLimit": "Límite superior de actividad anual",
    "TotalTechnologyModelPeriodActivityLowerLimit": "Límite inferior de actividad del periodo",
    "TotalTechnologyModelPeriodActivityUpperLimit": "Límite superior de actividad del periodo",
    "TradeRoute": "Ruta de comercio entre regiones",
    "VariableCost": "Costo variable ($/GWh)",
    "YearSplit": "Distribución temporal del año (timeslices)",
}

# Anomalías que se pueden explicar con Insumos/Mapeo/mapeo_tech_fuel.xlsx
TIPOS_ANOMALIA_CONTEXTUALIZABLES = [
    "SIN_CORRESPONDENCIA", "TECHNOLOGY_NO_EN_REGIONAL", "FUEL_NO_EN_REGIONAL",
]

_ARCHIVOS_MAPEO = {
    "mapeo": "mapeo_tech_fuel.xlsx",
    "diccionario_tech": "diccionario_tech.xlsx",
    "diccionario_fuel": "diccionario_fuel.xlsx",
}


def cargar_mapeos(carpeta: str | Path) -> dict[str, pd.DataFrame | None]:
    """Carga los archivos de Insumos/Mapeo/ que existan; los ausentes quedan en None.

    Claves: 'mapeo' (mapeo_tech_fuel.xlsx: regiones esperadas 0/1 por código
    nacional), 'diccionario_tech' y 'diccionario_fuel' (equivalencias y
    renombramientos Nacional <-> Regional).
    """
    carpeta = Path(carpeta)
    mapeos: dict[str, pd.DataFrame | None] = {}
    for clave, nombre in _ARCHIVOS_MAPEO.items():
        ruta = carpeta / nombre
        if ruta.exists():
            mapeos[clave] = pd.read_excel(ruta)
            logger.info("Mapeo cargado: %s (%s filas)", nombre, len(mapeos[clave]))
        else:
            mapeos[clave] = None
            logger.warning("Mapeo no encontrado (el análisis degrada sin él): %s", ruta)
    return mapeos


def renombramientos_fuel(diccionario_fuel: pd.DataFrame | None,
                         regiones: list[str] | None = None) -> pd.DataFrame:
    """Pares de renombramiento FUEL nacional -> base regional (ELC -> ELC001...).

    Sale de diccionario_fuel.xlsx: filas donde la base regional (sin prefijo)
    difiere del FUEL_NACIONAL declarado.
    """
    columnas = ["Fuel_Nacional", "Fuel_Regional", "Fuente"]
    if diccionario_fuel is None or diccionario_fuel.empty:
        return pd.DataFrame(columns=columnas)
    base = diccionario_fuel["FUEL_REGIONAL"].apply(lambda v: split_regional_name(v, regiones)[1])
    nacional = diccionario_fuel["FUEL_NACIONAL"].apply(normalize_text)
    pares = pd.DataFrame({"Fuel_Nacional": nacional, "Fuel_Regional": base})
    pares = pares.dropna()
    pares = pares[pares["Fuel_Nacional"] != pares["Fuel_Regional"]].drop_duplicates()
    pares["Fuente"] = "diccionario_fuel"
    return pares.reset_index(drop=True)


def renombramientos_fuzzy(solo_nacional: list[str], solo_regional: list[str],
                          umbral: float = 0.8) -> pd.DataFrame:
    """Renombramientos probables por similitud de nombre (difflib), sin diccionario.

    Para cada código solo-nacional busca el solo-regional más parecido con
    ratio >= umbral. Es una heurística: revisar antes de dar por buena la pareja.
    """
    filas = []
    for nac in solo_nacional:
        mejor, mejor_ratio = None, umbral
        for reg in solo_regional:
            ratio = difflib.SequenceMatcher(None, nac, reg).ratio()
            if ratio >= mejor_ratio:
                mejor, mejor_ratio = reg, ratio
        if mejor is not None:
            filas.append({"Fuel_Nacional": nac, "Fuel_Regional": mejor,
                          "Similitud": round(mejor_ratio, 3), "Fuente": "fuzzy"})
    return pd.DataFrame(filas, columns=["Fuel_Nacional", "Fuel_Regional", "Similitud", "Fuente"])


def _enriquecer_tech_con_diccionario(detalle: pd.DataFrame,
                                     diccionario_tech: pd.DataFrame | None) -> pd.DataFrame:
    """Añade Estado_Mapeo / Nombre_Equivalente al detalle TECHNOLOGY.

    - Solo Nacional  -> SIN_CORRESPONDENCIA (ausencia real declarada en el
      diccionario), RENOMBRADO (existe con TECHNOLOGY_REGIONAL distinto) o SIN_INFO.
    - Solo Regional  -> RENOMBRADO_DE si el diccionario le asigna un nacional.
    """
    detalle = detalle.copy()
    detalle["Estado_Mapeo"] = pd.NA
    detalle["Nombre_Equivalente"] = pd.NA
    if diccionario_tech is None or diccionario_tech.empty or detalle.empty:
        return detalle

    por_nacional: dict[str, tuple] = {}
    por_regional: dict[str, str] = {}
    for _, fila in diccionario_tech.iterrows():
        nac = normalize_text(fila.get("TECHNOLOGY_NACIONAL"))
        reg = normalize_text(fila.get("TECHNOLOGY_REGIONAL"))
        if nac is not None:
            por_nacional[nac] = (fila.get("EQUIVALENCIA"), reg)
        if reg is not None and nac is not None and reg != nac:
            por_regional[reg] = nac

    for idx, fila in detalle.iterrows():
        var = fila["Variable"]
        if fila["Categoria"] == "Solo Nacional":
            equiv, reg = por_nacional.get(var, (None, None))
            if reg is not None and reg != var:
                detalle.at[idx, "Estado_Mapeo"] = "RENOMBRADO"
                detalle.at[idx, "Nombre_Equivalente"] = reg
            elif isinstance(equiv, str) and equiv == "SIN_CORRESPONDENCIA":
                detalle.at[idx, "Estado_Mapeo"] = "SIN_CORRESPONDENCIA"
            else:
                detalle.at[idx, "Estado_Mapeo"] = "SIN_INFO"
        elif str(fila["Categoria"]).startswith("Solo Regional") and var in por_regional:
            detalle.at[idx, "Estado_Mapeo"] = "RENOMBRADO_DE"
            detalle.at[idx, "Nombre_Equivalente"] = por_regional[var]
    return detalle


def _marcar_renombramientos_fuel(detalle: pd.DataFrame, pares: pd.DataFrame) -> pd.DataFrame:
    """Añade Estado_Mapeo / Nombre_Equivalente al detalle FUEL usando los pares
    de renombramiento (diccionario o fuzzy)."""
    detalle = detalle.copy()
    detalle["Estado_Mapeo"] = pd.NA
    detalle["Nombre_Equivalente"] = pd.NA
    if detalle.empty:
        return detalle
    nac_a_reg = pares.groupby("Fuel_Nacional")["Fuel_Regional"].agg(lambda s: ", ".join(sorted(set(s)))) \
        if not pares.empty else pd.Series(dtype=object)
    reg_a_nac = pares.groupby("Fuel_Regional")["Fuel_Nacional"].agg(lambda s: ", ".join(sorted(set(s)))) \
        if not pares.empty else pd.Series(dtype=object)
    for idx, fila in detalle.iterrows():
        var = fila["Variable"]
        if fila["Categoria"] == "Solo Nacional":
            if var in nac_a_reg.index:
                detalle.at[idx, "Estado_Mapeo"] = "RENOMBRADO"
                detalle.at[idx, "Nombre_Equivalente"] = nac_a_reg[var]
            else:
                detalle.at[idx, "Estado_Mapeo"] = "SIN_INFO"
        elif str(fila["Categoria"]).startswith("Solo Regional") and var in reg_a_nac.index:
            detalle.at[idx, "Estado_Mapeo"] = "RENOMBRADO_DE"
            detalle.at[idx, "Nombre_Equivalente"] = reg_a_nac[var]
    return detalle


def _grupo_tematico(variable: str) -> str:
    """Prefijo temático de un código (para agrupar las variables solo-regionales)."""
    var = str(variable)
    for prefijo in ("BACKSTOP", "TRN", "ELCEV"):
        if var.startswith(prefijo):
            return prefijo
    for sector in SECTORES:
        if sector in var:
            return sector
    letras = re.match(r"^[A-Z]+", var)
    return letras.group(0)[:3] if letras else var[:3]


def inventario_tech_fuel(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    regiones: list[str],
    diccionario_tech: pd.DataFrame | None = None,
    diccionario_fuel: pd.DataFrame | None = None,
    umbral_fuzzy: float = 0.8,
) -> dict:
    """Inventario estructural completo: parámetros, TECHNOLOGY y FUEL.

    Enriquece los detalles de comparar_dimension con los diccionarios de mapeo
    (renombramientos y ausencias declaradas); sin diccionario de FUEL cae a
    detección fuzzy (difflib, ratio >= umbral_fuzzy).

    Returns dict: parametros, tech, fuel, cobertura_tech, cobertura_fuel,
    renombramientos_fuel, resumen (conteos por dimensión y categoría).
    """
    parametros = comparar_parametros(df_nacional, df_regional)
    parametros["Descripcion"] = parametros["Parameter"].map(PARAM_DESCRIPCIONES).fillna("")
    parametros["En_Nacional"] = parametros["Filas_Nacional"] > 0
    parametros["En_Regional"] = parametros["Filas_Regional"] > 0

    tech_det, tech_cob = comparar_dimension(df_nacional, df_regional, "TECHNOLOGY", regiones)
    fuel_det, fuel_cob = comparar_dimension(df_nacional, df_regional, "FUEL", regiones)

    tech_det = _enriquecer_tech_con_diccionario(tech_det, diccionario_tech)
    pares_fuel = renombramientos_fuel(diccionario_fuel, regiones)
    if pares_fuel.empty and not fuel_det.empty:
        pares_fuel = renombramientos_fuzzy(
            fuel_det.loc[fuel_det["Categoria"] == "Solo Nacional", "Variable"].tolist(),
            fuel_det.loc[fuel_det["Categoria"].str.startswith("Solo Regional"), "Variable"].tolist(),
            umbral_fuzzy,
        )
    fuel_det = _marcar_renombramientos_fuel(fuel_det, pares_fuel)

    for det in (tech_det, fuel_det):
        if not det.empty:
            det["Grupo"] = det["Variable"].apply(_grupo_tematico)

    resumen = (
        pd.concat([tech_det, fuel_det], ignore_index=True)
        .groupby(["Dimension", "Categoria"], as_index=False).size()
        .rename(columns={"size": "Cantidad"})
        if not (tech_det.empty and fuel_det.empty) else pd.DataFrame()
    )
    return {"parametros": parametros, "tech": tech_det, "fuel": fuel_det,
            "cobertura_tech": tech_cob, "cobertura_fuel": fuel_cob,
            "renombramientos_fuel": pares_fuel, "resumen": resumen}


def _mascara_sector(df: pd.DataFrame, sector: str) -> pd.Series:
    """True donde TECHNOLOGY o FUEL contienen el código del sector."""
    mascara = pd.Series(False, index=df.index)
    for col in INDICES_PREFIJADOS:
        if col in df.columns:
            mascara |= df[col].astype(str).str.contains(sector, na=False)
    return mascara


def detectar_sectores(df: pd.DataFrame, sectores: list[str] | None = None) -> list[str]:
    """Sectores presentes en el DataFrame (contains sobre TECHNOLOGY/FUEL)."""
    return [s for s in (sectores or SECTORES) if _mascara_sector(df, s).any()]


def _desglose_modos_tra(nac_s: pd.DataFrame, reg_s: pd.DataFrame,
                        regiones: list[str]) -> pd.DataFrame:
    """Desglose del sector transporte por modo (MODOS_TRA): presencia en
    Nacional/Regional y cobertura de regiones."""
    presencia: dict[str, set] = defaultdict(set)
    bases_reg: dict[str, set] = defaultdict(set)
    for col in INDICES_PREFIJADOS:
        for valor in reg_s[col].dropna().unique():
            region, base = split_regional_name(valor, regiones)
            if base is None:
                continue
            for modo in MODOS_TRA:
                if modo in base:
                    bases_reg[modo].add(base)
                    if region:
                        presencia[modo].add(region)

    filas = []
    for modo, descripcion in MODOS_TRA.items():
        codigos_nac = {
            normalize_text(v)
            for col in INDICES_PREFIJADOS for v in nac_s[col].dropna().unique()
            if modo in str(v)
        }
        if not codigos_nac and not bases_reg[modo]:
            continue
        regs = sorted(presencia[modo])
        filas.append({
            "Modo": modo, "Descripcion": descripcion,
            "Codigos_Nacional": len(codigos_nac),
            "Codigos_Regional_Base": len(bases_reg[modo]),
            "Regiones": ", ".join(regs), "Num_Regiones": len(regs),
            "Cubre_Todas": len(set(regs) & set(regiones)) == len(regiones),
        })
    return pd.DataFrame(filas)


def _desglose_fuel_region(
    df_nacional: pd.DataFrame,
    df_reg_prep: pd.DataFrame,
    nombre: str,
    spec: dict,
    sector: str,
    regiones: list[str],
    hasta_anio: int | None,
    anio: int | str | None,
    pares_fuel: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """Pivot variable base × región para un parámetro y sector, con columnas
    Nacional / Suma_Regional / Diferencia_Pct y advertencia de renombramiento."""
    indices = spec["indices"]
    dim = "FUEL" if ("FUEL" in indices and "TECHNOLOGY" not in indices) else "TECHNOLOGY"
    tiene_year = "YEAR" in indices

    nac_p = aplicar_filtro(df_nacional[df_nacional["Parameter"] == nombre], dim, [sector], "contiene")
    reg_p = aplicar_filtro(
        df_reg_prep[(df_reg_prep["Parameter"] == nombre) & ~df_reg_prep["IS_TRN"]],
        dim, [sector], "contiene",
    )
    if nac_p.empty and reg_p.empty:
        return pd.DataFrame(), None

    if tiene_year:
        anios = columnas_anio(nac_p if not nac_p.empty else reg_p, hasta_anio)
        col_valor = str(anio) if anio is not None and str(anio) in anios else anios[0]
    else:
        col_valor = TIME_INDEP_COL

    reg_v = reg_p.copy()
    reg_v["_valor"] = pd.to_numeric(reg_v[col_valor], errors="coerce")
    pivote = reg_v.pivot_table(index=dim, columns="REGION_PREFIJO", values="_valor", aggfunc="sum")
    pivote = pivote.reindex(columns=[r for r in regiones if r in pivote.columns])
    pivote["Suma_Regional"] = pivote.sum(axis=1)

    nacional = pd.to_numeric(nac_p[col_valor], errors="coerce").groupby(nac_p[dim]).sum()
    pivote = pivote.join(nacional.rename("Nacional"), how="outer")
    pivote["Diferencia_Pct"] = np.where(
        pivote["Nacional"].fillna(0) != 0,
        (pivote["Nacional"] - pivote["Suma_Regional"]) / pivote["Nacional"].abs(),
        np.nan,
    )

    pivote = pivote.reset_index().rename(columns={dim: "Variable"})
    orden = ["Variable", "Nacional"] + [r for r in regiones if r in pivote.columns] + \
            ["Suma_Regional", "Diferencia_Pct"]
    pivote = pivote[orden]

    if dim == "FUEL" and not pares_fuel.empty:
        nac_a_reg = pares_fuel.groupby("Fuel_Nacional")["Fuel_Regional"].agg(
            lambda s: ", ".join(sorted(set(s))))
        reg_a_nac = pares_fuel.groupby("Fuel_Regional")["Fuel_Nacional"].agg(
            lambda s: ", ".join(sorted(set(s))))

        def advertencia(var):
            if var in nac_a_reg.index:
                return f"renombrado en regional a: {nac_a_reg[var]}"
            if var in reg_a_nac.index:
                return f"renombramiento del nacional: {reg_a_nac[var]}"
            return ""

        pivote["Advertencia_Renombramiento"] = pivote["Variable"].apply(advertencia)
    return pivote, col_valor


def analisis_por_sector(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    sector: str,
    otoole_params: dict[str, dict],
    params_cfg: dict,
    diccionario_tech: pd.DataFrame | None = None,
    diccionario_fuel: pd.DataFrame | None = None,
    parametros: list[str] | None = None,
    df_reg_prep: pd.DataFrame | None = None,
    anio_desglose: int | str | None = None,
) -> dict:
    """Análisis profundo de un sector (TRA, IND, RES, TER, AGF, PWR).

    Combina el inventario estructural del sector (comparar_dimension filtrado),
    la validación de PARAMS_CLAVE_SECTOR vía comparar_parametro (misma ruta
    aditiva/intensiva del flujo general, filtro contains=sector) y el desglose
    variable × región del parámetro con más datos.

    parametros: si se pasa (modo 'parametro'/'lista_parametros'), solo se validan
    los parámetros clave incluidos en esa lista.
    df_reg_prep: regional ya preparado (preparar_regional) para reutilizar entre
    sectores; si es None se calcula aquí.

    Returns dict: sector, filas_nacional, filas_regional, inventario_tech,
    inventario_fuel, cobertura_tech, cobertura_fuel, modos_tra (solo TRA),
    validacion (detalle con columna Coincide), resumen_validacion,
    parametro_desglose, anio_desglose, desglose, anomalias.
    """
    regiones = list(params_cfg["prefijo_region"].values())
    tolerancia = params_cfg["tolerancia_comparacion"]
    hasta_anio = params_cfg.get("anio_maximo_comparacion")

    nac_s = df_nacional[_mascara_sector(df_nacional, sector)]
    reg_s = df_regional[_mascara_sector(df_regional, sector)]
    if df_reg_prep is None:
        df_reg_prep = preparar_regional(df_regional, regiones)

    # 1) Inventario del sector (misma lógica que el inventario global, filtrado)
    inv_tech, cob_tech = comparar_dimension(nac_s, reg_s, "TECHNOLOGY", regiones)
    inv_fuel, cob_fuel = comparar_dimension(nac_s, reg_s, "FUEL", regiones)
    inv_tech = _enriquecer_tech_con_diccionario(inv_tech, diccionario_tech)
    pares_fuel = renombramientos_fuel(diccionario_fuel, regiones)
    if pares_fuel.empty and not inv_fuel.empty:
        pares_fuel = renombramientos_fuzzy(
            inv_fuel.loc[inv_fuel["Categoria"] == "Solo Nacional", "Variable"].tolist(),
            inv_fuel.loc[inv_fuel["Categoria"].str.startswith("Solo Regional"), "Variable"].tolist(),
        )
    inv_fuel = _marcar_renombramientos_fuel(inv_fuel, pares_fuel)
    modos_tra = _desglose_modos_tra(nac_s, reg_s, regiones) if sector == "TRA" else None

    # 2) Validación de los parámetros clave con datos (ruta aditiva/intensiva)
    nombres = [
        p for p in PARAMS_CLAVE_SECTOR
        if (parametros is None or p in parametros)
        and p in otoole_params
        and (df_nacional["Parameter"] == p).any()
    ]
    bloques, filas_resumen, anomalias = [], [], []
    for nombre in nombres:
        comp, anom = comparar_parametro(
            nombre, otoole_params[nombre], df_nacional, df_reg_prep, params_cfg,
            tecnologias=[sector], fuels=[sector], modo_filtro="contiene",
        )
        anomalias.extend(anom)
        if comp.empty:
            continue
        comp = comp.copy()
        comp["Coincide"] = comp["Diferencia"].abs() <= tolerancia
        n_disc = int((~comp["Coincide"]).sum())
        filas_resumen.append({
            "Sector": sector, "Parametro": nombre,
            "Tipo": "intensivo" if nombre in params_cfg["parametros_intensivos"] else "aditivo",
            "Filas": len(comp), "Filas_OK": len(comp) - n_disc,
            "Filas_Discrepancia": n_disc,
            "Pct_Discrepancia": round(100 * n_disc / len(comp), 1),
            "Max_Diferencia_Abs": float(comp["Diferencia"].abs().max()),
        })
        bloques.append(comp)

    validacion = pd.concat(bloques, ignore_index=True) if bloques else pd.DataFrame()
    resumen_validacion = pd.DataFrame(filas_resumen)

    # 3) Desglose variable × región del parámetro con más datos. Si el candidato
    # no tiene valores regionales en el año de referencia se prueba el siguiente.
    parametro_desglose, desglose, anio_usado = None, pd.DataFrame(), None
    for fila in sorted(filas_resumen, key=lambda f: f["Filas"], reverse=True):
        candidato = fila["Parametro"]
        desg, anio = _desglose_fuel_region(
            df_nacional, df_reg_prep, candidato, otoole_params[candidato],
            sector, regiones, hasta_anio, anio_desglose, pares_fuel,
        )
        if not desg.empty and desg["Suma_Regional"].notna().any():
            parametro_desglose, desglose, anio_usado = candidato, desg, anio
            break

    logger.info("Sector %s: %s filas nac / %s filas reg, %s parámetros clave validados",
                sector, len(nac_s), len(reg_s), len(filas_resumen))
    return {
        "sector": sector, "filas_nacional": len(nac_s), "filas_regional": len(reg_s),
        "inventario_tech": inv_tech, "inventario_fuel": inv_fuel,
        "cobertura_tech": cob_tech, "cobertura_fuel": cob_fuel,
        "modos_tra": modos_tra,
        "validacion": validacion, "resumen_validacion": resumen_validacion,
        "parametro_desglose": parametro_desglose, "anio_desglose": anio_usado,
        "desglose": desglose,
        "anomalias": pd.DataFrame(anomalias, columns=["Parametro", "Tecnologia_Fuel",
                                                      "Tipo_Anomalia", "Detalle"]),
    }


def tabla_por_sector(analisis_sectores: dict[str, dict]) -> pd.DataFrame:
    """Pivot sector × parámetro con el % de filas con discrepancia (hoja Por_Sector)."""
    frames = [a["resumen_validacion"] for a in analisis_sectores.values()
              if not a["resumen_validacion"].empty]
    if not frames:
        return pd.DataFrame()
    todo = pd.concat(frames, ignore_index=True)
    pivote = todo.pivot_table(index="Sector", columns="Parametro", values="Pct_Discrepancia")
    return pivote.reset_index().rename_axis(None, axis=1)


def contextualizar_anomalias(anomalias: pd.DataFrame, mapeo: pd.DataFrame,
                             regiones: list[str] | None = None) -> pd.DataFrame:
    """Explica las anomalías estructurales con mapeo_tech_fuel.xlsx.

    Para SIN_CORRESPONDENCIA / TECHNOLOGY_NO_EN_REGIONAL / FUEL_NO_EN_REGIONAL
    agrega TIENE_MAPEO, REGIONES_ESPERADAS (columnas 0/1 del mapeo),
    NOMBRE_REGIONAL (TECHNOLOGY_REGIONAL/FUEL_REGIONAL si está definido) y
    ACCION_SUGERIDA: cambio_de_nombre | no_existe_en_region |
    verificar_creacion | sin_info.
    """
    from utils import REGIONES as _REGIONES_DEFAULT
    regiones = regiones or _REGIONES_DEFAULT
    cols_region = [c for c in regiones if c in mapeo.columns]

    indice_tech: dict[str, pd.Series] = {}
    indice_fuel: dict[str, pd.Series] = {}
    for _, fila in mapeo.iterrows():
        tech = normalize_text(fila.get("TECHNOLOGY"))
        fuel = normalize_text(fila.get("FUEL"))
        if tech is not None:
            indice_tech[tech] = fila
        if fuel is not None:
            indice_fuel[fuel] = fila

    out = anomalias[anomalias["Tipo_Anomalia"].isin(TIPOS_ANOMALIA_CONTEXTUALIZABLES)].copy()
    contexto = []
    for _, an in out.iterrows():
        # Tecnologia_Fuel puede traer varios índices ("TECH | FUEL | modo")
        partes = [p.strip() for p in str(an["Tecnologia_Fuel"]).split("|")]
        es_fuel = an["Tipo_Anomalia"] == "FUEL_NO_EN_REGIONAL"
        orden_busqueda = (indice_fuel, indice_tech) if es_fuel else (indice_tech, indice_fuel)
        fila_mapeo = None
        for indice in orden_busqueda:
            for parte in partes:
                if parte in indice:
                    fila_mapeo = indice[parte]
                    break
            if fila_mapeo is not None:
                break

        if fila_mapeo is None:
            contexto.append({"TIENE_MAPEO": False, "REGIONES_ESPERADAS": "",
                             "NOMBRE_REGIONAL": "", "ACCION_SUGERIDA": "sin_info"})
            continue
        regiones_esperadas = [c for c in cols_region if safe_float(fila_mapeo.get(c)) == 1]
        nombre_regional = (normalize_text(fila_mapeo.get("TECHNOLOGY_REGIONAL"))
                           or normalize_text(fila_mapeo.get("FUEL_REGIONAL")) or "")
        if nombre_regional:
            accion = "cambio_de_nombre"
        elif not regiones_esperadas:
            accion = "no_existe_en_region"
        else:
            accion = "verificar_creacion"
        contexto.append({"TIENE_MAPEO": True,
                         "REGIONES_ESPERADAS": ", ".join(regiones_esperadas),
                         "NOMBRE_REGIONAL": nombre_regional,
                         "ACCION_SUGERIDA": accion})

    for columna in ["TIENE_MAPEO", "REGIONES_ESPERADAS", "NOMBRE_REGIONAL", "ACCION_SUGERIDA"]:
        out[columna] = [c[columna] for c in contexto] if contexto else pd.Series(dtype=object)
    return out.reset_index(drop=True)

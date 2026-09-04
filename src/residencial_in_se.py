"""Corrección del sector RESIDENCIAL para las regiones aisladas Insular (IN) y
Sureste (SE) — caso especial, NO integrado al flujo general de regionalización.

En el escenario regional las 5 regiones conectadas conservan el patrón nacional
`..._URB` / `..._RUR` (tecnologías `DEMRES*` y fuels `RES*`). En IN y SE, por ser
aisladas, el modelo se simplificó: se sumó URB+RUR y se quitó el sufijo, y algunas
tecnologías no existen. Este módulo recompone la demanda/actividad residencial de
IN y SE a partir del SAND nacional y de `Insumos/Mapeo/participaciones.xlsx` (el
único insumo de participaciones del flujo; los `Insumos/Participacion_*.xlsx` son
la materia prima de `00_Generar_Mapeo.ipynb`, no entrada de aquí):

    valor_IN(base) = Σ_sufijo  nac(base_sufijo) × participación_IN(base_sufijo)

colapsando URB+RUR sobre el valor ABSOLUTO del nacional (sumar fracciones sería
incorrecto porque URB y RUR tienen demandas distintas). Reglas de destino, en orden:

  1a. Si el código SIN sufijo existe en la región  -> se acumula ahí (colapso).
  1b. Si NO existe sin sufijo:
      - tecnología eléctrica (contiene 'DEMRESELC') -> DEMRESELCOTH conservando la
        eficiencia (_HIG/_LOW; cualquier otra -> _MID).
      - fuel (RESILU/RESTV/RESWHT/RESWSH)           -> RESOTH.
      - cualquier otro caso                          -> queda sin regla (se registra).

Genera: SAND Reducido con las filas corregidas de IN/SE, un log de auditoría de cada
reasignación y la participación efectiva (fracción ponderada por demanda) derivada.
El nacional y las participaciones originales NO se modifican.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from sand_io import columnas_anio
from utils import TIME_INDEP_COL, normalize_text, safe_float

log = logging.getLogger("residencial_in_se")

# Nombre de región en los archivos de participación -> prefijo del modelo.
REGIONES_AISLADAS = {"Insular": "IN", "Este": "SE"}
# Fuels de uso residencial que no existen en IN/SE y se agregan a RESOTH.
FUELS_A_RESOTH = {"RESILU", "RESTV", "RESWHT", "RESWSH"}
_SUFIJO = re.compile(r"_(URB|RUR)$")

# Etiquetas de regla (para el log de auditoría).
REGLA_COLAPSO = "colapso_sin_sufijo"
REGLA_ELEC_OTH = "redirigir_electrica_DEMRESELCOTH"
REGLA_FUEL_OTH = "agregar_fuel_RESOTH"
REGLA_SIN_REGLA = "sin_regla"


def quitar_sufijo(codigo: str) -> str:
    """'DEMRESELCOTH_HIG_URB' -> 'DEMRESELCOTH_HIG'. Sin sufijo, igual."""
    return _SUFIJO.sub("", str(codigo))


def _eficiencia_oth(base: str) -> str:
    """Eficiencia de destino en DEMRESELCOTH: conserva _HIG/_LOW; el resto -> _MID."""
    tok = base.rsplit("_", 1)[-1]
    return tok if tok in ("HIG", "MID", "LOW") else "MID"


def clasificar_destino(base: str, dim: str, prefijo: str, existentes: set[str]) -> tuple[str | None, str]:
    """Código base de destino y regla aplicada para un código residencial sin sufijo.

    `existentes`: códigos regionales (con prefijo) que existen en el escenario.
    """
    if f"{prefijo}_{base}" in existentes:
        return base, REGLA_COLAPSO
    if dim == "TECHNOLOGY" and "DEMRESELC" in base:
        destino = f"DEMRESELCOTH_{_eficiencia_oth(base)}"
        if f"{prefijo}_{destino}" in existentes:
            return destino, REGLA_ELEC_OTH
        return None, REGLA_SIN_REGLA
    if dim == "FUEL" and base in FUELS_A_RESOTH:
        if f"{prefijo}_RESOTH" in existentes:
            return "RESOTH", REGLA_FUEL_OTH
        return None, REGLA_SIN_REGLA
    return None, REGLA_SIN_REGLA


def dimension_parametro(parametro: str, params_otoole: dict[str, dict]) -> str:
    """Dimensión de código del parámetro: TECHNOLOGY si se indexa por tecnología,
    si no FUEL. Es la misma convención que usa `participaciones.xlsx` (una columna
    por dimensión, y `cargar_participaciones` descarta la que no aplica)."""
    spec = params_otoole.get(parametro)
    if spec is None:
        raise ValueError(f"Parámetro no definido en el config de otoole: {parametro}")
    indices = spec["indices"]
    if "TECHNOLOGY" in indices:
        return "TECHNOLOGY"
    if "FUEL" in indices:
        return "FUEL"
    raise ValueError(f"{parametro} no se indexa por TECHNOLOGY ni por FUEL: "
                     "no tiene código residencial que corregir")


def cargar_participacion(ruta, parametro: str, dim: str,
                         params_otoole: dict[str, dict] | None = None,
                         anios: list[int] | None = None) -> pd.DataFrame:
    """Participaciones de IN/SE para un parámetro, desde `participaciones.xlsx`.

    `Insumos/Mapeo/participaciones.xlsx` (salida de `00_Generar_Mapeo.ipynb`) es el
    ÚNICO insumo de participaciones del flujo; los `Insumos/Participacion_*.xlsx`
    son la materia prima de ese notebook, no entrada de este módulo. Se reutiliza
    `regionalizador.cargar_participaciones`, que estandariza el formato ancho y
    descarta las filas cuya dimensión no corresponde al parámetro.

    Devuelve el largo restringido a las regiones aisladas:
    columns [codigo, prefijo, Anio, Participacion].
    `anios` es obligatorio si el archivo trae participación constante (sin año):
    esa fila se replica a cada año pedido.
    """
    from regionalizador import ANIO_CONSTANTE, cargar_participaciones

    df = cargar_participaciones(ruta, parametros=[parametro], params_otoole=params_otoole)
    df = df[df["Parametro"] == parametro]
    # El archivo puede traer el nombre de la región o su prefijo.
    prefijos = {**REGIONES_AISLADAS, **{p: p for p in REGIONES_AISLADAS.values()}}
    df = df[df["Region"].isin(prefijos)]

    largo = pd.DataFrame({
        "codigo": df[dim].map(normalize_text),
        "prefijo": df["Region"].map(prefijos),
        "Anio": df["Año"],
        "Participacion": df["Participacion"].map(safe_float),
    }).dropna(subset=["codigo"])

    constante = largo["Anio"] == ANIO_CONSTANTE
    if constante.any():
        if not anios:
            raise ValueError(f"{parametro}: participación constante (sin año) y no se "
                             "indicaron los años a los que replicarla")
        largo = pd.concat(
            [largo[~constante],
             largo[constante].drop(columns="Anio").merge(
                 pd.DataFrame({"Anio": list(anios)}), how="cross")],
            ignore_index=True)
    largo["Anio"] = largo["Anio"].astype(int)
    log.info("%s: %d participaciones IN/SE (%d códigos)", parametro, len(largo),
             largo["codigo"].nunique())
    return largo[["codigo", "prefijo", "Anio", "Participacion"]]


def corregir_parametro(sand_nac: pd.DataFrame, participacion: pd.DataFrame, parametro: str,
                       dim: str, existentes: set[str], anio_max: int = 2054):
    """Recompone las filas de IN/SE para un parámetro residencial.

    dim: 'TECHNOLOGY' o 'FUEL' (la columna de código del parámetro).
    Devuelve (df_sand, df_log, df_participacion_efectiva).
    """
    cols_anio = [c for c in columnas_anio(sand_nac) if int(c) <= anio_max]
    anios = [int(c) for c in cols_anio]

    nac_p = sand_nac[sand_nac["Parameter"] == parametro]
    # valor nacional por código y año (suma si hubiera más de una fila por código)
    val_nac = (nac_p.assign(_cod=nac_p[dim].map(normalize_text))
               .dropna(subset=["_cod"])
               .groupby("_cod")[cols_anio]
               .apply(lambda g: g.apply(pd.to_numeric, errors="coerce").sum(min_count=1)))

    pct = participacion.copy()
    pct["codigo"] = pct["codigo"].map(normalize_text)
    pct = pct[pct["Participacion"].map(safe_float).fillna(0) > 0]

    # acumuladores: (prefijo, base_destino) -> {anio: valor}
    acum: dict[tuple[str, str], dict[int, float]] = {}
    fuente_nac: dict[tuple[str, str], dict[int, float]] = {}   # demanda nacional combinada (para % efectiva)
    log_filas: list[dict] = []

    for (codigo, prefijo), grupo in pct.groupby(["codigo", "prefijo"]):
        if codigo not in val_nac.index:
            continue                       # sin fila nacional para este parámetro
        base = quitar_sufijo(codigo)
        destino, regla = clasificar_destino(base, dim, prefijo, existentes)
        share_por_anio = grupo.set_index("Anio")["Participacion"].to_dict()
        nac_row = val_nac.loc[codigo]

        aporte = {}
        for col, anio in zip(cols_anio, anios):
            v = safe_float(nac_row[col])
            s = safe_float(share_por_anio.get(anio))
            if v is None or s is None or s == 0:
                continue
            aporte[anio] = v * s
        if not aporte:
            continue

        if destino is None:                # sin regla: no se puede reubicar
            log_filas.append({"Region": prefijo, "Parametro": parametro, "Dimension": dim,
                              "Codigo_Origen": codigo, "Codigo_Destino": "(sin regla)",
                              "Regla": regla, "Valor_Sumado": round(sum(aporte.values()), 6),
                              "Anios": ";".join(str(a) for a in sorted(aporte))})
            log.warning("Sin regla de destino: %s (%s) en %s / %s", codigo, base, prefijo, parametro)
            continue

        clave = (prefijo, destino)
        dst = acum.setdefault(clave, {})
        src = fuente_nac.setdefault(clave, {})
        for anio, val in aporte.items():
            dst[anio] = dst.get(anio, 0.0) + val
            src[anio] = src.get(anio, 0.0) + safe_float(nac_row[str(anio)])
        log_filas.append({"Region": prefijo, "Parametro": parametro, "Dimension": dim,
                          "Codigo_Origen": codigo, "Codigo_Destino": f"{prefijo}_{destino}",
                          "Regla": regla, "Valor_Sumado": round(sum(aporte.values()), 6),
                          "Anios": ";".join(str(a) for a in sorted(aporte))})

    # SAND Reducido: una fila por (parametro, código destino)
    filas_sand, filas_pct = [], []
    for (prefijo, destino), valores in sorted(acum.items()):
        cod_reg = f"{prefijo}_{destino}"
        fila = {"Parameter": parametro, "REGION": "RE1", "TECHNOLOGY": pd.NA, "EMISSION": pd.NA,
                "MODE_OF_OPERATION": pd.NA, "FUEL": pd.NA, "TIMESLICE": pd.NA, "STORAGE": pd.NA,
                "REGION2": pd.NA, TIME_INDEP_COL: pd.NA}
        fila[dim] = cod_reg
        for col, anio in zip(cols_anio, anios):
            if anio in valores:
                fila[col] = round(valores[anio], 6)
        filas_sand.append(fila)
        # participación efectiva = aporte / demanda nacional combinada de las fuentes
        for anio in anios:
            base_nac = fuente_nac[(prefijo, destino)].get(anio)
            if base_nac:
                filas_pct.append({"Parametro": parametro, "Dimension": dim,
                                  "Codigo_Destino": cod_reg, "Region": prefijo, "Anio": anio,
                                  "Participacion_Efectiva": round(valores.get(anio, 0.0) / base_nac, 6),
                                  "Valor_Absoluto": round(valores.get(anio, 0.0), 6)})

    cols_sand = ["Parameter", "REGION", "TECHNOLOGY", "EMISSION", "MODE_OF_OPERATION", "FUEL",
                 "TIMESLICE", "STORAGE", "REGION2", TIME_INDEP_COL] + cols_anio
    df_sand = pd.DataFrame(filas_sand, columns=cols_sand)
    df_log = pd.DataFrame(log_filas, columns=["Region", "Parametro", "Dimension", "Codigo_Origen",
                                              "Codigo_Destino", "Regla", "Valor_Sumado", "Anios"])
    df_pct = pd.DataFrame(filas_pct, columns=["Parametro", "Dimension", "Codigo_Destino", "Region",
                                              "Anio", "Participacion_Efectiva", "Valor_Absoluto"])
    log.info("%s: %d filas SAND para IN/SE, %d reasignaciones (%s sin regla)", parametro,
             len(df_sand), len(df_log), (df_log["Regla"] == REGLA_SIN_REGLA).sum() if not df_log.empty else 0)
    return df_sand, df_log, df_pct

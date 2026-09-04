"""Sincronización de dos escenarios SAND (Nacional nuevo vs Regional nuevo).

Caso de uso distinto al de `regionalizador`: aquí **ya existen** los dos
escenarios y no se parte de cero. El modelo regional no es el nacional con
prefijos — tiene tecnologías nuevas (TRN*, carga EV), fuels renombrados,
códigos que no existen en todas las regiones y parámetros intencionalmente
distintos. El objetivo no es que sean idénticos, sino que **los parámetros
equivalentes tengan valores coherentes y que las diferencias residuales sean
intencionales y estén documentadas**.

Flujo (ver `notebooks/03_Flujo_Integrado.ipynb`):

1. `comparador.comparar_escenarios` produce el diagnóstico crudo.
2. `construir_decisiones` lo colapsa a una tabla editable, una fila por
   (Parámetro, TECHNOLOGY, FUEL), con `TIPO_DIFERENCIA` y una `ACCION`
   **inferida** desde `Insumos/Mapeo/mapeo_tech_fuel.xlsx`.
3. El usuario edita las ACCIONes en Excel; `leer_decisiones` las revalida.
4. `aplicar_decisiones` ejecuta `regionalizador.regionalizar` solo sobre las
   filas marcadas `regionalizar`; las marcadas `crear_en_regional` NO se crean
   automáticamente, se devuelven como pendientes para intervención manual.
5. `integrar_sands` arma el SAND regional sincronizado (sin tocar el original)
   y `comparar_resolucion` mide qué se resolvió y qué quedó como residual.

Las cuatro ACCIONes son las únicas admitidas:

| ACCION              | Efecto                                                    |
|---------------------|-----------------------------------------------------------|
| `regionalizar`      | reparte el valor del nacional nuevo hacia el regional      |
| `mantener_regional` | la diferencia es intencional: no se toca                   |
| `crear_en_regional` | falta el código en el regional: se alerta, no se crea      |
| `ignorar`           | no equivalente o bajo el umbral: no entra al reporte final |
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import regionalizador
from utils import normalize_text, split_regional_name

logger = logging.getLogger(__name__)

# --- Vocabulario de la tabla de decisiones -----------------------------------
TIPO_VALOR_DIFERENTE = "valor_diferente"
TIPO_TECH_FALTANTE = "tecnologia_faltante"
TIPO_FUEL_FALTANTE = "fuel_faltante"
TIPO_TECH_RENOMBRADA = "tecnologia_renombrada"
TIPO_FUEL_RENOMBRADO = "fuel_renombrado"
TIPO_REGIONES_INCOMPLETAS = "regiones_incompletas"
TIPO_FUEL_NO_COINCIDE = "fuel_no_coincide"
TIPO_PARAMETRO_AUSENTE = "parametro_ausente"
# El código SÍ existe en el regional, pero este parámetro no tiene filas para él.
# `comparador` lo reporta como SIN_CORRESPONDENCIA (que es por parámetro); no es
# una tecnología faltante, es un parámetro sin poblar para un código que ya existe.
TIPO_SIN_DATOS_PARAMETRO = "sin_datos_para_el_codigo"

ACCION_REGIONALIZAR = "regionalizar"
ACCION_MANTENER = "mantener_regional"
ACCION_CREAR = "crear_en_regional"
ACCION_IGNORAR = "ignorar"
ACCIONES_VALIDAS = frozenset({ACCION_REGIONALIZAR, ACCION_MANTENER,
                              ACCION_CREAR, ACCION_IGNORAR})

# Anomalías de comparador.comparar_escenarios -> TIPO_DIFERENCIA de esta tabla
_TIPO_POR_ANOMALIA = {
    "SIN_CORRESPONDENCIA": TIPO_TECH_FALTANTE,          # se afina con la dimensión real
    "TECHNOLOGY_NO_EN_REGIONAL": TIPO_TECH_FALTANTE,
    "FUEL_NO_EN_REGIONAL": TIPO_FUEL_FALTANTE,
    "REGIONES_INCOMPLETAS": TIPO_REGIONES_INCOMPLETAS,
    "FUEL_NO_COINCIDE": TIPO_FUEL_NO_COINCIDE,
    "PARAMETRO_NO_EN_REGIONAL": TIPO_PARAMETRO_AUSENTE,
    "PARAMETRO_SIN_VALORES": TIPO_PARAMETRO_AUSENTE,
}

COLUMNAS_DECISIONES = [
    "Parametro", "TECHNOLOGY", "FUEL", "TIPO_DIFERENCIA", "Origen",
    "N_Filas", "Max_Diferencia_Abs", "Max_Diferencia_Pct",
    "Valor_Nacional_Total", "Valor_Regional_Total",
    "Regiones_Existentes", "Regiones_Esperadas", "Regiones_Faltantes",
    "Nombre_Regional", "Tiene_Participacion", "ACCION", "Motivo_Accion", "Detalle",
]


# --- Diagnóstico -------------------------------------------------------------
def resumen_diferencias(comparacion: pd.DataFrame, umbral_pct: float = 1.0) -> pd.DataFrame:
    """Resumen por parámetro: cuántas filas superan el umbral de diferencia.

    `umbral_pct` va en puntos porcentuales (1.0 = 1%); `Diferencia_Pct` de
    `comparador` es una fracción, así que se compara contra umbral_pct/100.
    Estado = OK si ninguna fila del parámetro supera el umbral.
    """
    if comparacion.empty:
        return pd.DataFrame(columns=["Parametro", "Filas", "Filas_Sobre_Umbral",
                                     "Max_Diferencia_Abs", "Max_Diferencia_Pct", "Estado"])
    df = comparacion.copy()
    df["_pct_abs"] = pd.to_numeric(df["Diferencia_Pct"], errors="coerce").abs()
    df["_sobre"] = df["_pct_abs"] > (umbral_pct / 100.0)
    resumen = (
        df.groupby("Parametro", as_index=False)
        .agg(Filas=("Diferencia", "size"),
             Filas_Sobre_Umbral=("_sobre", "sum"),
             Max_Diferencia_Abs=("Diferencia", lambda s: float(s.abs().max())),
             Max_Diferencia_Pct=("_pct_abs", "max"))
    )
    resumen["Max_Diferencia_Pct"] = resumen["Max_Diferencia_Pct"] * 100.0
    resumen["Estado"] = np.where(resumen["Filas_Sobre_Umbral"] > 0, "REVISAR", "OK")
    return resumen.sort_values("Filas_Sobre_Umbral", ascending=False).reset_index(drop=True)


# --- Contraste contra el Nacional base ----------------------------------------
_LLAVES_SAND = ["Parameter", "TECHNOLOGY", "FUEL", "EMISSION",
                "MODE_OF_OPERATION", "TIMESLICE", "STORAGE"]


def _a_largo_sand(df: pd.DataFrame, hasta_anio: int | None, nombre: str) -> pd.DataFrame:
    """SAND a formato largo [llaves, Año, <nombre>] para diferenciar escenarios."""
    from sand_io import columnas_valor
    from utils import TIME_INDEP_COL

    llaves = [c for c in _LLAVES_SAND if c in df.columns]
    cols = columnas_valor(df, hasta_anio)
    largo = df[llaves + cols].melt(id_vars=llaves, value_vars=cols,
                                   var_name="Año", value_name=nombre)
    largo["Año"] = largo["Año"].replace(TIME_INDEP_COL, "-1")
    largo[nombre] = pd.to_numeric(largo[nombre], errors="coerce")
    for col in llaves:
        largo[col] = largo[col].apply(_llave)
    return largo.dropna(subset=[nombre])


def cambios_vs_base(
    df_escenario: pd.DataFrame,
    df_base: pd.DataFrame,
    parametros: list[str] | None = None,
    hasta_anio: int | None = None,
    tolerancia: float = 1e-4,
) -> pd.DataFrame:
    """Qué cambió en el Nacional del nuevo escenario respecto del Nacional base.

    Es el dato que desempata al decidir: si el combo **cambió** en el nacional
    nuevo, la diferencia contra el regional viene del escenario y `regionalizar`
    suele ser lo correcto; si el nacional **no cambió**, la diferencia nació del
    lado regional y probablemente sea intencional.

    Returns una fila por (Parametro, TECHNOLOGY, FUEL) con Estado_Vs_Base:
    NUEVO (no existía en el base), CAMBIADO, ELIMINADO o IGUAL.
    """
    esc = df_escenario if parametros is None else \
        df_escenario[df_escenario["Parameter"].isin(parametros)]
    base = df_base if parametros is None else \
        df_base[df_base["Parameter"].isin(parametros)]

    largo_esc = _a_largo_sand(esc, hasta_anio, "Valor_Escenario")
    largo_base = _a_largo_sand(base, hasta_anio, "Valor_Base")
    llaves = [c for c in _LLAVES_SAND if c in largo_esc.columns] + ["Año"]
    cruce = largo_esc.merge(largo_base, on=llaves, how="outer")

    cruce["_dif"] = (cruce["Valor_Escenario"].fillna(0) - cruce["Valor_Base"].fillna(0)).abs()
    cruce["_nuevo"] = cruce["Valor_Base"].isna()
    cruce["_eliminado"] = cruce["Valor_Escenario"].isna()

    agrupado = (
        cruce.groupby(["Parameter", "TECHNOLOGY", "FUEL"], dropna=False, as_index=False)
        .agg(Filas_Comparadas=("_dif", "size"),
             Max_Diferencia_Vs_Base=("_dif", "max"),
             _todo_nuevo=("_nuevo", "all"),
             _todo_eliminado=("_eliminado", "all"))
        .rename(columns={"Parameter": "Parametro"})
    )
    agrupado["Estado_Vs_Base"] = np.select(
        [agrupado["_todo_nuevo"], agrupado["_todo_eliminado"],
         agrupado["Max_Diferencia_Vs_Base"] > tolerancia],
        ["NUEVO", "ELIMINADO", "CAMBIADO"], default="IGUAL")
    agrupado = agrupado.drop(columns=["_todo_nuevo", "_todo_eliminado"])
    logger.info("Cambios vs base: %s", agrupado["Estado_Vs_Base"].value_counts().to_dict())
    return agrupado


def anotar_cambios_vs_base(decisiones: pd.DataFrame, cambios: pd.DataFrame) -> pd.DataFrame:
    """Añade `Estado_Vs_Base` y `Max_Diferencia_Vs_Base` a la tabla de decisiones.

    Solo informa: no reescribe ninguna ACCION — la decisión sigue siendo del
    usuario, esto le da el contexto para tomarla.
    """
    if decisiones.empty or cambios.empty:
        salida = decisiones.copy()
        salida["Estado_Vs_Base"] = "(sin base)"
        return salida
    llaves = ["Parametro", "TECHNOLOGY", "FUEL"]
    dec = decisiones.copy()
    cam = cambios.copy()
    for df in (dec, cam):
        for col in llaves:
            df[col] = df[col].apply(_llave)
    salida = dec.merge(cam[llaves + ["Estado_Vs_Base", "Max_Diferencia_Vs_Base"]],
                       on=llaves, how="left")
    salida["Estado_Vs_Base"] = salida["Estado_Vs_Base"].fillna("(no está en el nacional)")
    return salida


# --- Estado de un código contra el mapeo y el SAND regional -------------------
def _llave(valor) -> str:
    """Valor de índice normalizado a texto, con '' para los nulos.

    Ojo: `Series.apply(normalize_text)` reconvierte los None a NaN al inferir el
    dtype, así que nunca se puede confiar en `is None` sobre una columna ya
    guardada — se re-normaliza siempre al leerla. El '' evita además que dos
    NaN distintos no crucen entre sí al comparar tuplas.
    """
    return normalize_text(valor) or ""


def _universos(df_regional: pd.DataFrame) -> dict[str, set[str]]:
    return {
        "TECHNOLOGY": set(df_regional["TECHNOLOGY"].dropna().astype(str).str.strip()),
        "FUEL": set(df_regional["FUEL"].dropna().astype(str).str.strip()),
    }


def _estado_codigo(codigo: str | None, dimension: str, mapeo: dict,
                   universos: dict[str, set[str]], prefijos: list[str]) -> dict:
    """Regiones donde el código existe / debe existir, y su nombre regional.

    `dimension` es 'TECHNOLOGY' o 'FUEL'. Si `mapeo['disponible']` es False se
    asumen las 7 regiones como esperadas (mismo degradado que el regionalizador).
    """
    vacio = {"existentes": [], "esperadas": [], "faltantes": [], "nombre_regional": ""}
    if codigo is None:
        return vacio
    es_tech = dimension == "TECHNOLOGY"
    renames = mapeo["rename_tech"] if es_tech else mapeo["rename_fuel"]
    regs_map = mapeo["regiones_tech"] if es_tech else mapeo["regiones_fuel"]
    base_regional = renames.get(codigo, codigo)

    existentes = [p for p in prefijos if f"{p}_{base_regional}" in universos[dimension]]
    esperadas = regs_map.get(codigo, []) if mapeo["disponible"] else list(prefijos)
    return {
        "existentes": existentes,
        "esperadas": esperadas,
        "faltantes": sorted(set(esperadas) - set(existentes)),
        "nombre_regional": base_regional if base_regional != codigo else "",
        # Distingue "el mapeo lo declara con todas las regiones en 0" (decisión
        # tomada) de "el código ni siquiera aparece en el mapeo" (falta info).
        "en_mapeo": codigo in regs_map,
    }


def _tiene_participacion(df_pct: pd.DataFrame | None, parametro: str,
                         tech: str | None, fuel: str | None) -> bool:
    """True si el archivo de participaciones cubre el combo (o trae comodín '*').

    Solo mira la existencia de la fila; el reparto real lo resuelve el
    regionalizador. Con `df_pct=None` devuelve True (no se puede descartar).
    """
    if df_pct is None or df_pct.empty:
        return True
    del_param = df_pct[df_pct["Parametro"] == parametro]
    if del_param.empty:
        return False
    if (del_param["TECHNOLOGY"] == regionalizador.COMODIN).any():
        return True
    if tech is not None and (del_param["TECHNOLOGY"] == tech).any():
        return True
    if fuel is not None and (del_param["FUEL"] == fuel).any():
        return True
    return False


# --- Inferencia de la acción por defecto -------------------------------------
def _inferir_accion(tipo: str, estado: dict, es_intensivo: bool,
                    tiene_pct: bool, mapeo_disponible: bool) -> tuple[str, str]:
    """ACCION por defecto + motivo, a partir del mapeo y del estado del código.

    Reglas (en orden):
    - renombrado conocido -> `ignorar`: la diferencia es de nomenclatura, no de valor.
    - el mapeo declara que no debe existir en ninguna región -> `mantener_regional`.
    - falta en regiones que el mapeo espera -> `crear_en_regional` (solo alerta).
    - el código existe y el valor difiere (o el parámetro no tiene datos para él)
      -> `regionalizar`, si hay participación (o el parámetro es intensivo, que
      no la necesita).
    - sin información en el mapeo -> `mantener_regional`, marcado para revisión
      manual: no se actúa automáticamente sobre lo que no se puede explicar.
    """
    if estado["nombre_regional"]:
        return (ACCION_IGNORAR,
                f"Renombrado en el regional a '{estado['nombre_regional']}': "
                f"la diferencia es de nomenclatura, no de valor")

    if tipo == TIPO_PARAMETRO_AUSENTE:
        return (ACCION_REGIONALIZAR,
                "El parámetro no existe (o está vacío) en el regional: se regionaliza completo")

    if not estado["existentes"]:
        if estado["esperadas"]:
            return (ACCION_CREAR,
                    f"Falta en el regional; el mapeo lo espera en "
                    f"{', '.join(estado['esperadas'])}")
        if mapeo_disponible and estado["en_mapeo"]:
            return (ACCION_MANTENER,
                    "El mapeo declara que el código no debe existir en ninguna región")
        return (ACCION_MANTENER,
                "Sin correspondencia regional y el código no aparece en el mapeo: REVISAR a mano")

    if estado["faltantes"]:
        return (ACCION_CREAR,
                f"Existe en {', '.join(estado['existentes'])} pero el mapeo lo espera "
                f"también en {', '.join(estado['faltantes'])}")

    if tipo == TIPO_REGIONES_INCOMPLETAS:
        # `comparador` compara contra las 7 regiones; el mapeo puede esperar menos.
        # Sin faltantes, la cobertura parcial es exactamente la esperada.
        return (ACCION_IGNORAR,
                f"Cobertura parcial esperada: existe en las "
                f"{len(estado['existentes'])} regiones que el mapeo declara "
                f"({', '.join(estado['existentes'])})")

    if tipo in (TIPO_VALOR_DIFERENTE, TIPO_SIN_DATOS_PARAMETRO):
        motivo_base = (
            "Código presente en ambos escenarios"
            if tipo == TIPO_VALOR_DIFERENTE else
            f"El código ya existe en {', '.join(estado['existentes'])} pero este parámetro "
            f"no tiene valores para él en el regional")
        if es_intensivo or tiene_pct:
            return (ACCION_REGIONALIZAR, f"{motivo_base}: se aplica el valor del nacional nuevo")
        return (ACCION_MANTENER,
                f"{motivo_base}, pero no hay participación definida para repartirlo: "
                f"REVISAR a mano")

    if tipo == TIPO_FUEL_NO_COINCIDE:
        return (ACCION_MANTENER,
                "La tecnología existe pero el FUEL no coincide: revisar el mapeo tech-fuel")

    return (ACCION_MANTENER, "Sin regla aplicable: REVISAR a mano")


# --- Construcción de la tabla de decisiones ----------------------------------
def _dimension_de(codigo: str, universos_nac: dict[str, set[str]]) -> str:
    """Decide si un código de anomalía es TECHNOLOGY o FUEL mirando el nacional."""
    if codigo in universos_nac["TECHNOLOGY"]:
        return "TECHNOLOGY"
    if codigo in universos_nac["FUEL"]:
        return "FUEL"
    return "TECHNOLOGY"


def _filas_desde_comparacion(comparacion: pd.DataFrame, umbral_pct: float) -> pd.DataFrame:
    """Colapsa la comparación a una fila por (Parametro, TECHNOLOGY, FUEL).

    Solo entran las combinaciones con alguna fila sobre el umbral; el resto
    queda fuera por definición (`ignorar` implícito).
    """
    if comparacion.empty:
        return pd.DataFrame()
    df = comparacion.copy()
    for col in ("TECHNOLOGY", "FUEL"):
        if col not in df.columns:
            df[col] = pd.NA
    df["_pct_abs"] = pd.to_numeric(df["Diferencia_Pct"], errors="coerce").abs()
    df = df[df["_pct_abs"] > (umbral_pct / 100.0)]
    if df.empty:
        return pd.DataFrame()

    df["_dif_abs"] = df["Diferencia"].abs()
    agrupado = (
        df.groupby(["Parametro", "TECHNOLOGY", "FUEL"], dropna=False, as_index=False)
        .agg(N_Filas=("_dif_abs", "size"),
             Max_Diferencia_Abs=("_dif_abs", "max"),
             Max_Diferencia_Pct=("_pct_abs", "max"),
             Valor_Nacional_Total=("Valor_Nacional", "sum"),
             Valor_Regional_Total=("Valor_Regional", "sum"))
    )
    agrupado["Max_Diferencia_Pct"] = agrupado["Max_Diferencia_Pct"] * 100.0
    agrupado["TIPO_DIFERENCIA"] = TIPO_VALOR_DIFERENTE
    agrupado["Origen"] = "comparacion"
    agrupado["Detalle"] = ""
    return agrupado


def _filas_desde_anomalias(anomalias: pd.DataFrame,
                           universos_nac: dict[str, set[str]]) -> pd.DataFrame:
    """Una fila de decisión por anomalía estructural contextualizable."""
    if anomalias.empty:
        return pd.DataFrame()
    filas = []
    for _, an in anomalias.iterrows():
        tipo = _TIPO_POR_ANOMALIA.get(an["Tipo_Anomalia"])
        if tipo is None:
            continue
        etiqueta = normalize_text(an["Tecnologia_Fuel"])
        tech = fuel = None
        if etiqueta not in (None, "-"):
            for parte in (p.strip() for p in str(etiqueta).split("|")):
                if not parte:
                    continue
                if an["Tipo_Anomalia"] == "FUEL_NO_EN_REGIONAL":
                    dimension = "FUEL"
                elif an["Tipo_Anomalia"] == "TECHNOLOGY_NO_EN_REGIONAL":
                    dimension = "TECHNOLOGY"
                else:
                    dimension = _dimension_de(parte, universos_nac)
                if dimension == "TECHNOLOGY" and tech is None:
                    tech = parte
                elif dimension == "FUEL" and fuel is None:
                    fuel = parte
        # SIN_CORRESPONDENCIA sirve a ambas dimensiones: afinar el tipo
        if tipo == TIPO_TECH_FALTANTE and tech is None and fuel is not None:
            tipo = TIPO_FUEL_FALTANTE
        filas.append({
            "Parametro": an["Parametro"], "TECHNOLOGY": tech, "FUEL": fuel,
            "TIPO_DIFERENCIA": tipo, "Origen": f"anomalia:{an['Tipo_Anomalia']}",
            "N_Filas": 0, "Max_Diferencia_Abs": np.nan, "Max_Diferencia_Pct": np.nan,
            "Valor_Nacional_Total": np.nan, "Valor_Regional_Total": np.nan,
            "Detalle": an["Detalle"],
        })
    return pd.DataFrame(filas)


def construir_decisiones(
    comparacion: pd.DataFrame,
    anomalias: pd.DataFrame,
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    mapeo: dict,
    cfg: dict,
    umbral_pct: float = 1.0,
    df_pct: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tabla editable de decisiones, una fila por (Parámetro, TECHNOLOGY, FUEL).

    Junta las diferencias de valor sobre el umbral (`comparacion`) con las
    anomalías estructurales (`anomalias`), las contextualiza con el mapeo
    (`regionalizador.cargar_mapeo_regional`) y propone una `ACCION` por defecto
    — ver `_inferir_accion` para las reglas.

    `umbral_pct` en puntos porcentuales (1.0 = 1%). `df_pct` (participaciones)
    es opcional: si se pasa, un combo sin participación no se propone para
    `regionalizar`.
    """
    prefijos = list(cfg["prefijo_region"].values())
    universos_reg = _universos(df_regional)
    universos_nac = _universos(df_nacional)
    intensivos = set(cfg["parametros_intensivos"])

    partes = [df for df in (_filas_desde_comparacion(comparacion, umbral_pct),
                            _filas_desde_anomalias(anomalias, universos_nac))
              if not df.empty]
    if not partes:
        logger.info("Sin diferencias sobre el umbral (%s%%) ni anomalías", umbral_pct)
        return pd.DataFrame(columns=COLUMNAS_DECISIONES)

    dec = pd.concat(partes, ignore_index=True)
    for col in ("TECHNOLOGY", "FUEL"):
        dec[col] = dec[col].apply(normalize_text)
    # Una anomalía puede repetir un combo ya presente por diferencia de valor:
    # gana la fila estructural, que es la que explica la causa raíz.
    dec = dec.sort_values("Origen", ascending=False).drop_duplicates(
        subset=["Parametro", "TECHNOLOGY", "FUEL"], keep="first")

    contexto = []
    for _, fila in dec.iterrows():
        tech = normalize_text(fila["TECHNOLOGY"])
        fuel = normalize_text(fila["FUEL"])
        dimension = "TECHNOLOGY" if tech is not None else "FUEL"
        codigo = tech if tech is not None else fuel
        estado = _estado_codigo(codigo, dimension, mapeo, universos_reg, prefijos)
        tiene_pct = _tiene_participacion(df_pct, fila["Parametro"], tech, fuel)

        tipo = fila["TIPO_DIFERENCIA"]
        # `comparador` marca SIN_CORRESPONDENCIA por parámetro: si el código sí
        # existe en el regional, lo que falta son los valores de ESTE parámetro,
        # no la tecnología/fuel. Reclasificar antes de inferir la acción.
        if tipo in (TIPO_TECH_FALTANTE, TIPO_FUEL_FALTANTE) and estado["existentes"]:
            tipo = TIPO_SIN_DATOS_PARAMETRO

        accion, motivo = _inferir_accion(
            tipo, estado,
            es_intensivo=fila["Parametro"] in intensivos,
            tiene_pct=tiene_pct, mapeo_disponible=mapeo["disponible"],
        )
        if estado["nombre_regional"]:   # el rename manda sobre el tipo genérico
            tipo = TIPO_TECH_RENOMBRADA if dimension == "TECHNOLOGY" else TIPO_FUEL_RENOMBRADO
        contexto.append({
            "TIPO_DIFERENCIA": tipo,
            "Regiones_Existentes": ",".join(estado["existentes"]),
            "Regiones_Esperadas": ",".join(estado["esperadas"]),
            "Regiones_Faltantes": ",".join(estado["faltantes"]),
            "Nombre_Regional": estado["nombre_regional"],
            "Tiene_Participacion": tiene_pct,
            "ACCION": accion, "Motivo_Accion": motivo,
        })

    ctx = pd.DataFrame(contexto, index=dec.index)
    dec["TIPO_DIFERENCIA"] = ctx["TIPO_DIFERENCIA"]
    for col in ("Regiones_Existentes", "Regiones_Esperadas", "Regiones_Faltantes",
                "Nombre_Regional", "Tiene_Participacion", "ACCION", "Motivo_Accion"):
        dec[col] = ctx[col]

    dec = dec[COLUMNAS_DECISIONES].sort_values(
        ["ACCION", "Max_Diferencia_Abs"], ascending=[True, False]
    ).reset_index(drop=True)
    logger.info("Decisiones construidas: %s filas (%s)", len(dec),
                dec["ACCION"].value_counts().to_dict())
    return dec


# --- Ida y vuelta por Excel ---------------------------------------------------
_HOJA_DECISIONES = "Decisiones"

_INSTRUCCIONES = pd.DataFrame([
    {"ACCION": ACCION_REGIONALIZAR,
     "Significado": "Aplicar el valor del nacional nuevo al regional (reparte por participación; "
                    "los intensivos se copian igual a cada región)"},
    {"ACCION": ACCION_MANTENER,
     "Significado": "La diferencia es intencional: el regional se deja como está. "
                    "Queda listada como diferencia residual en el reporte final"},
    {"ACCION": ACCION_CREAR,
     "Significado": "El código debe añadirse al regional. NO se crea automáticamente: "
                    "el notebook alerta con qué crear y en qué regiones"},
    {"ACCION": ACCION_IGNORAR,
     "Significado": "No equivalente (renombramiento, código nuevo del regional) o bajo el "
                    "umbral: no entra al reporte de residuales"},
])


def exportar_decisiones(decisiones: pd.DataFrame, ruta: str | Path) -> Path:
    """Escribe `decisiones` a Excel para que el usuario edite la columna ACCION.

    Además de la hoja `Decisiones` incluye una hoja `Instrucciones` con las
    cuatro ACCIONes admitidas, para no tener que volver al notebook.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        decisiones.to_excel(writer, sheet_name=_HOJA_DECISIONES, index=False)
        _INSTRUCCIONES.to_excel(writer, sheet_name="Instrucciones", index=False)
    logger.info("Decisiones exportadas para edición manual: %s (%s filas)", ruta, len(decisiones))
    return ruta


def leer_decisiones(ruta: str | Path, estricto: bool = True) -> pd.DataFrame:
    """Relee el Excel de decisiones después de la edición manual y lo valida.

    Con `estricto=True` una ACCION no reconocida aborta; con `False` se degrada
    a `mantener_regional` (la opción que no modifica nada) y se avisa por log.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecutar primero la Sección 2 para generarlo.")
    dec = pd.read_excel(ruta, sheet_name=_HOJA_DECISIONES)
    faltantes = [c for c in ("Parametro", "TECHNOLOGY", "FUEL", "ACCION") if c not in dec.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en {ruta.name}: {faltantes}")

    dec["ACCION"] = dec["ACCION"].apply(lambda v: (normalize_text(v) or "").lower())
    invalidas = dec[~dec["ACCION"].isin(ACCIONES_VALIDAS)]
    if not invalidas.empty:
        detalle = sorted(set(invalidas["ACCION"]))
        if estricto:
            raise ValueError(
                f"{len(invalidas)} filas de {ruta.name} tienen una ACCION no reconocida "
                f"{detalle}. Válidas: {sorted(ACCIONES_VALIDAS)}")
        logger.warning("%s filas con ACCION inválida %s -> se tratan como %s",
                       len(invalidas), detalle, ACCION_MANTENER)
        dec.loc[invalidas.index, "ACCION"] = ACCION_MANTENER

    for col in ("TECHNOLOGY", "FUEL"):
        dec[col] = dec[col].apply(normalize_text)
    logger.info("Decisiones releídas de %s: %s", ruta.name, dec["ACCION"].value_counts().to_dict())
    return dec


# --- Aplicación de las decisiones ---------------------------------------------
def _renombrar_para_regional(df: pd.DataFrame, mapeo: dict) -> pd.DataFrame:
    """Reescribe los códigos nacionales con su nombre base regional (ELC -> ELC003).

    `regionalizador.regionalizar` (flujo sin mapeo) busca el código nacional tal
    cual dentro del universo regional; sin este paso, un código renombrado no se
    encontraría en ninguna región y se omitiría en silencio.
    """
    if not mapeo["rename_tech"] and not mapeo["rename_fuel"]:
        return df
    out = df.copy()
    for col, renames in (("TECHNOLOGY", mapeo["rename_tech"]), ("FUEL", mapeo["rename_fuel"])):
        if renames and col in out.columns:
            out[col] = out[col].apply(
                lambda v: renames.get(normalize_text(v), v) if pd.notna(v) else v)
    return out


def _filtrar_combos_decididos(df_sand: pd.DataFrame, combos: set[tuple],
                              usa_tech: bool, usa_fuel: bool,
                              prefijos: list[str]) -> pd.DataFrame:
    """Deja solo las filas SAND cuyo (TECHNOLOGY, FUEL) base fue decidido.

    Necesario porque `regionalizar` recibe las listas de TECHNOLOGY y de FUEL
    por separado y las cruza (AND), lo que puede producir combinaciones que
    nadie marcó `regionalizar`.
    """
    if df_sand.empty or not combos:
        return df_sand
    def base(valor):
        return split_regional_name(valor, prefijos)[1] or ""
    vacio = pd.Series("", index=df_sand.index)
    tech = df_sand["TECHNOLOGY"].apply(base) if usa_tech else vacio
    fuel = df_sand["FUEL"].apply(base) if usa_fuel else vacio
    return df_sand[[k in combos for k in zip(tech, fuel)]]


def aplicar_decisiones(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    df_pct: pd.DataFrame,
    decisiones: pd.DataFrame,
    params_otoole: dict[str, dict],
    cfg: dict,
    mapeo: dict,
    years_filtro: list[int] | None = None,
) -> dict:
    """Ejecuta las decisiones: regionaliza lo marcado y lista lo que falta crear.

    Solo las filas `ACCION == 'regionalizar'` producen datos, vía
    `regionalizador.regionalizar` acotado a los códigos decididos de cada
    parámetro. Las de `crear_en_regional` se devuelven en `pendientes_creacion`
    (con las regiones a crear) — **no se crea nada automáticamente**, requiere
    intervención manual. `mantener_regional` e `ignorar` no hacen nada.

    Returns dict: sands, log, resumen, pendientes_creacion.
    """
    prefijos = list(cfg["prefijo_region"].values())
    a_regionalizar = decisiones[decisiones["ACCION"] == ACCION_REGIONALIZAR]

    nac_renombrado = _renombrar_para_regional(df_nacional, mapeo)
    pct_renombrado = _renombrar_para_regional(df_pct, mapeo)

    sands: dict[str, pd.DataFrame] = {}
    logs: list[pd.DataFrame] = []
    resumen: list[dict] = []

    for parametro, grupo in a_regionalizar.groupby("Parametro"):
        spec = params_otoole.get(parametro)
        if spec is None:
            logger.warning("Parámetro %r no está en config_depurado.yaml: se salta", parametro)
            continue
        indices = spec["indices"]
        usa_tech, usa_fuel = "TECHNOLOGY" in indices, "FUEL" in indices

        # Códigos ya con el nombre regional, igual que el nacional renombrado
        def base_reg(valor, renames):
            cod = _llave(valor)
            return renames.get(cod, cod)

        techs = sorted({b for b in (base_reg(v, mapeo["rename_tech"])
                                    for v in grupo["TECHNOLOGY"]) if b}) if usa_tech else []
        fuels = sorted({b for b in (base_reg(v, mapeo["rename_fuel"])
                                    for v in grupo["FUEL"]) if b}) if usa_fuel else []
        combos = {(base_reg(f["TECHNOLOGY"], mapeo["rename_tech"]) if usa_tech else "",
                   base_reg(f["FUEL"], mapeo["rename_fuel"]) if usa_fuel else "")
                  for _, f in grupo.iterrows()}

        resultado = regionalizador.regionalizar(
            nac_renombrado, df_regional, pct_renombrado,
            parametros=[parametro], params_otoole=params_otoole, cfg=cfg,
            tecnologias_filtro=techs, fuels_filtro=fuels, modo_filtro="exacto",
            years_filtro=years_filtro,
        )
        logs.append(resultado["log"])
        df_sand = resultado["sands"].get(parametro, pd.DataFrame())
        # Un combo con TECHNOLOGY y FUEL ambos vacíos = "todo el parámetro"
        if not df_sand.empty and combos != {("", "")}:
            df_sand = _filtrar_combos_decididos(df_sand, combos, usa_tech, usa_fuel, prefijos)
        if not df_sand.empty:
            sands[parametro] = df_sand.reset_index(drop=True)
        resumen.append({
            "Parametro": parametro,
            "Combos_Decididos": len(grupo),
            "Filas_SAND": len(df_sand),
            "Regionalizado": not df_sand.empty,
        })

    pendientes = decisiones[decisiones["ACCION"] == ACCION_CREAR].copy()
    if not pendientes.empty:
        columnas = [c for c in ("Parametro", "TECHNOLOGY", "FUEL", "TIPO_DIFERENCIA",
                                "Regiones_Existentes", "Regiones_Faltantes",
                                "Nombre_Regional", "Motivo_Accion")
                    if c in pendientes.columns]
        pendientes = pendientes[columnas].reset_index(drop=True)

    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    logger.info("Decisiones aplicadas: %s parámetros regionalizados, %s pendientes de creación",
                len(sands), len(pendientes))
    return {"sands": sands, "log": log,
            "resumen": pd.DataFrame(resumen), "pendientes_creacion": pendientes}


def integrar_sands(df_regional: pd.DataFrame, sands: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """SAND regional con las filas sincronizadas sustituidas (copia, no in-place).

    Reemplaza únicamente las filas cuyo (Parameter, TECHNOLOGY, FUEL) coincide
    con alguna de las generadas: todo lo demás del regional — tecnologías nuevas,
    diferencias intencionales — se conserva intacto.
    """
    if not sands:
        return df_regional.copy()

    nuevas = pd.concat(sands.values(), ignore_index=True)
    claves = set(zip(nuevas["Parameter"].apply(_llave),
                     nuevas["TECHNOLOGY"].apply(_llave),
                     nuevas["FUEL"].apply(_llave)))
    actuales = list(zip(df_regional["Parameter"].apply(_llave),
                        df_regional["TECHNOLOGY"].apply(_llave),
                        df_regional["FUEL"].apply(_llave)))
    conserva = df_regional[[k not in claves for k in actuales]]
    salida = pd.concat([conserva, nuevas.reindex(columns=df_regional.columns)],
                       ignore_index=True)
    logger.info("SAND sincronizado: %s filas conservadas + %s sustituidas = %s",
                len(conserva), len(nuevas), len(salida))
    return salida


# --- Validación post-sincronización -------------------------------------------
def combos_sobre_umbral(comparacion: pd.DataFrame, umbral_pct: float) -> pd.DataFrame:
    """Combos (Parametro, TECHNOLOGY, FUEL) con diferencia sobre el umbral."""
    filas = _filas_desde_comparacion(comparacion, umbral_pct)
    if filas.empty:
        return pd.DataFrame(columns=["Parametro", "TECHNOLOGY", "FUEL",
                                     "Max_Diferencia_Abs", "Max_Diferencia_Pct"])
    return filas[["Parametro", "TECHNOLOGY", "FUEL",
                  "Max_Diferencia_Abs", "Max_Diferencia_Pct"]]


def comparar_resolucion(
    comparacion_antes: pd.DataFrame,
    comparacion_despues: pd.DataFrame,
    decisiones: pd.DataFrame,
    umbral_pct: float = 1.0,
) -> dict:
    """Qué diferencias se resolvieron y cuáles quedan, cruzando antes vs después.

    Estados por combo: RESUELTA (estaba sobre el umbral y ya no),
    PERSISTE (sigue sobre el umbral) y NUEVA (aparece después — señal de que la
    sincronización rompió algo, hay que mirarla).

    Las diferencias residuales son las que PERSISTEN cuya decisión fue
    `mantener_regional`: intencionales y documentadas por su `Motivo_Accion`.
    """
    antes = combos_sobre_umbral(comparacion_antes, umbral_pct)
    despues = combos_sobre_umbral(comparacion_despues, umbral_pct)

    llaves = ["Parametro", "TECHNOLOGY", "FUEL"]
    cruce = antes.merge(despues, on=llaves, how="outer",
                        suffixes=("_Antes", "_Despues"), indicator=True)
    cruce["Estado"] = cruce["_merge"].map({
        "left_only": "RESUELTA", "both": "PERSISTE", "right_only": "NUEVA"})
    cruce = cruce.drop(columns="_merge")

    if not decisiones.empty:
        cols_dec = [c for c in llaves + ["TIPO_DIFERENCIA", "ACCION", "Motivo_Accion"]
                    if c in decisiones.columns]
        cruce = cruce.merge(decisiones[cols_dec].drop_duplicates(subset=llaves),
                            on=llaves, how="left")
        cruce["ACCION"] = cruce["ACCION"].fillna("(sin decisión)")

    residuales = cruce[(cruce["Estado"] == "PERSISTE")
                       & (cruce.get("ACCION", pd.Series(dtype=object)) == ACCION_MANTENER)]
    conteo = cruce["Estado"].value_counts()
    metricas = {
        "combos_antes": len(antes),
        "combos_despues": len(despues),
        "resueltas": int(conteo.get("RESUELTA", 0)),
        "persisten": int(conteo.get("PERSISTE", 0)),
        "nuevas": int(conteo.get("NUEVA", 0)),
        "residuales_intencionales": len(residuales),
        "umbral_pct": umbral_pct,
    }
    logger.info("Resolución: %s resueltas, %s persisten, %s nuevas",
                metricas["resueltas"], metricas["persisten"], metricas["nuevas"])
    return {"detalle": cruce.reset_index(drop=True),
            "residuales": residuales.reset_index(drop=True),
            "metricas": metricas}

"""Regionalización de parámetros nacionales SAND hacia las 7 regiones.

El reparto usa un archivo de participaciones. El formato canónico es:
    Parámetro | TECHNOLOGY | FUEL | Región | 2022 | ... | 2055
o, si el porcentaje es constante en el tiempo, una única columna `Participacion`.

`cargar_participaciones` también estandariza automáticamente variantes con las
regiones en formato ancho (columnas Antioquia..Suroccidente o AN..SO):
- `Fuel|Anio|<regiones>` o `Technology|Anio|<regiones>` sin columna Parámetro:
  se infiere que aplica a la lista `parametros` (PARAMETROS_A_REGIONALIZAR);
  si un parámetro de la lista no se indexa por esa dimensión (según
  config_depurado.yaml) se advierte y sus filas se descartan.
- `Parametro|Fuel/Tecnologia|<regiones>` constante en los años.
- Solo `<regiones>` (sin parámetro ni tecnología/fuel): participación comodín
  `*` que aplica a cualquier combo de los parámetros de la lista, manteniendo
  los códigos TECHNOLOGY/FUEL del nacional.

Reglas por clasificación (config/params_config.yaml):
- ADITIVOS: valor_regional = valor_nacional × participación. Los valores
  centinela "sin límite" (99999 / solo-9s) se copian tal cual, no se reparten.
- INTENSIVOS: valor_regional = valor_nacional (mismo valor en cada región).
- Parámetros sin YEAR: misma lógica sobre `Time indipendent variables`
  (requieren participación constante).

Los índices del parámetro se leen de config_depurado.yaml para saber si va
prefijado por TECHNOLOGY, FUEL o ambos. Todo se registra en un log estructurado.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sand_io import columnas_anio, escribir_sand
from utils import (REGIONES, TIME_INDEP_COL, es_centinela, is_trade_technology,
                   normalize_text, safe_float)

logger = logging.getLogger(__name__)

ANIO_CONSTANTE = -1  # marcador interno: participación sin dimensión temporal
COMODIN = "*"        # participación global: aplica a cualquier TECHNOLOGY/FUEL del parámetro

_ALIAS_COLUMNAS = {
    "parámetro": "Parametro", "parametro": "Parametro", "parameter": "Parametro",
    "región": "Region", "region": "Region",
    "participación": "Participacion", "participacion": "Participacion",
    "technology": "TECHNOLOGY", "tecnología": "TECHNOLOGY", "tecnologia": "TECHNOLOGY",
    "fuel": "FUEL",
    "año": "Año", "anio": "Año", "year": "Año",
}

# Encabezados que identifican una región en los formatos anchos
_NOMBRES_REGION = ["Antioquia", "Caribe", "Este", "Insular", "Nordeste", "Oriente", "Suroccidente"]


def _descartar_dimensiones_invalidas(df: pd.DataFrame, params_otoole: dict[str, dict],
                                     nombre_archivo: str) -> pd.DataFrame:
    """Advierte y descarta filas cuya dimensión clave (TECHNOLOGY/FUEL) no está
    entre los índices del parámetro según config_depurado.yaml. El comodín '*'
    no exige el índice (aplica sin importar la indexación)."""
    mascara = pd.Series(True, index=df.index)
    for parametro, grupo in df.groupby("Parametro"):
        spec = params_otoole.get(parametro)
        if spec is None:
            logger.warning("%s: parámetro %r no existe en config_depurado.yaml; se descartan %s filas",
                           nombre_archivo, parametro, len(grupo))
            mascara &= df["Parametro"] != parametro
            continue
        indices = spec.get("indices", [])
        for col in ("TECHNOLOGY", "FUEL"):
            con_clave = grupo[col].notna() & (grupo[col] != COMODIN)
            if con_clave.any() and col not in indices:
                logger.warning("%s: %s no se indexa por %s; se descartan %s filas de participación",
                               nombre_archivo, parametro, col, int(con_clave.sum()))
                mascara &= ~((df["Parametro"] == parametro)
                             & df[col].notna() & (df[col] != COMODIN))
    return df[mascara]


def cargar_participaciones(
    path: str | Path,
    parametros: list[str] | None = None,
    params_otoole: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Lee el archivo de participaciones y lo devuelve en formato largo estándar:
    columnas [Parametro, TECHNOLOGY, FUEL, Region, Año, Participacion].

    Estandariza automáticamente los formatos admitidos:
    - Canónico: `Parámetro | TECHNOLOGY | FUEL | Región` + columnas de año
      (2022...2055) y/o columna única `Participacion` (constante, Año = -1).
    - Regiones anchas: columnas Antioquia..Suroccidente o AN..SO (se despivotan
      a la columna Región), con años en columna larga `Anio` o constantes.
    - Sin columna `Parámetro`: se replica para cada parámetro de `parametros`
      (PARAMETROS_A_REGIONALIZAR); obligatorio pasar la lista en ese caso.
    - Sin TECHNOLOGY ni FUEL: participación comodín `*` (aplica a cualquier
      combo del parámetro, manteniendo los códigos del nacional).

    Si se pasa `params_otoole` (config_depurado.yaml), se advierte y descarta
    toda fila cuya dimensión clave no esté en los índices del parámetro.
    Alias de encabezados admitidos: Parámetro/Parameter, TECHNOLOGY/Tecnología/
    Tecnologia/Technology, FUEL/Fuel, Región/Region, Año/Anio/Year.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de participaciones: {path}")
    df = pd.read_excel(path)
    df.columns = [_ALIAS_COLUMNAS.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]

    # Columna 'Año' presente pero toda vacía = sin dimensión de año (participación
    # constante en el tiempo): se trata como ausente. Si no, la rama de "formato
    # largo por año" filtraría por Año.notna() y descartaría TODAS las filas
    # (participaciones cargadas = 0 -> todo cae en SIN_PARTICIPACION). Mismo criterio
    # que con la columna 'Parámetro' vacía de abajo.
    if "Año" in df.columns and df["Año"].map(normalize_text).isna().all():
        df = df.drop(columns="Año")
        logger.info("%s: columna 'Año' vacía; participación tratada como constante", path.name)

    # Regiones en formato ancho: despivotar columnas de región -> Region/Participacion
    cols_region = [c for c in df.columns if c in _NOMBRES_REGION or c in REGIONES]
    if cols_region and "Region" not in df.columns:
        id_vars = [c for c in df.columns if c not in cols_region]
        df = df.melt(id_vars=id_vars, value_vars=cols_region,
                     var_name="Region", value_name="Participacion")
        logger.info("%s: formato ancho en regiones detectado (%s columnas de región)",
                    path.name, len(cols_region))

    # Columna 'Parámetro' presente pero toda vacía (p. ej. la genera
    # 00_Generar_Mapeo sin rellenarla): se trata como ausente para poder inferir
    # los parámetros de la lista, en vez de dejar todo con Parametro=None.
    if "Parametro" in df.columns and df["Parametro"].map(normalize_text).isna().all():
        df = df.drop(columns="Parametro")
        logger.info("%s: columna 'Parámetro' vacía; se infiere de la lista de parámetros",
                    path.name)

    # Sin columna Parámetro: se infiere que aplica a todos los parámetros pedidos
    if "Parametro" not in df.columns:
        if not parametros:
            raise ValueError(
                f"{path.name} no tiene columna 'Parámetro': pase la lista de parámetros "
                f"(PARAMETROS_A_REGIONALIZAR) en cargar_participaciones(..., parametros=...) "
                f"para inferir a cuáles aplica")
        df = pd.concat([df.assign(Parametro=p) for p in parametros], ignore_index=True)
        logger.info("%s: sin columna 'Parámetro'; se aplica a %s", path.name, list(parametros))

    # Sin TECHNOLOGY ni FUEL: participación comodín (aplica a cualquier combo)
    if "TECHNOLOGY" not in df.columns and "FUEL" not in df.columns:
        df["TECHNOLOGY"] = COMODIN
        df["FUEL"] = COMODIN
        logger.info("%s: sin TECHNOLOGY/FUEL; participación comodín '%s' "
                    "(aplica a cualquier combo manteniendo los códigos del nacional)",
                    path.name, COMODIN)
    for col in ("TECHNOLOGY", "FUEL"):
        if col not in df.columns:
            df[col] = pd.NA

    if "Region" not in df.columns:
        raise ValueError(f"Columnas faltantes en {path.name}: ['Region'] "
                         f"(se esperan Parámetro, TECHNOLOGY, FUEL, Región y años o Participacion; "
                         f"o regiones anchas Antioquia..Suroccidente / AN..SO)")

    id_cols = ["Parametro", "TECHNOLOGY", "FUEL", "Region"]
    for col in id_cols:
        df[col] = df[col].apply(normalize_text)

    if params_otoole is not None:
        df = _descartar_dimensiones_invalidas(df, params_otoole, path.name)

    year_cols = [c for c in df.columns if str(c).isdigit()]
    bloques = []
    if year_cols:
        largo = df.melt(id_vars=id_cols, value_vars=year_cols,
                        var_name="Año", value_name="Participacion").dropna(subset=["Participacion"])
        largo["Año"] = largo["Año"].astype(int)
        bloques.append(largo)
    if "Participacion" in df.columns:
        con_valor = df.loc[df["Participacion"].notna()]
        if "Año" in df.columns:  # formato largo por año (columna Anio)
            largo = con_valor.loc[con_valor["Año"].notna(), id_cols + ["Año", "Participacion"]].copy()
            largo["Año"] = pd.to_numeric(largo["Año"], errors="coerce").astype(int)
            bloques.append(largo)
        else:  # constante en el tiempo
            const = con_valor[id_cols + ["Participacion"]].copy()
            const["Año"] = ANIO_CONSTANTE
            bloques.append(const)
    if not bloques:
        raise ValueError(f"{path.name} no tiene columnas de año ni columna 'Participacion'")

    largo = pd.concat(bloques, ignore_index=True)
    largo["Participacion"] = pd.to_numeric(largo["Participacion"], errors="coerce")
    largo = largo.dropna(subset=["Participacion"])
    # Filas duplicadas exactas (p. ej. si PARAMETROS_A_REGIONALIZAR trae un parámetro
    # repetido, cada réplica genera la misma fila): se descartan para que un combo no
    # tenga (Region, Año) repetido — si no, `_participacion_combo` devolvería varias
    # filas y `pct_combo.get((prefijo, anio))` daría una Series ("truth value of a
    # Series is ambiguous" al regionalizar).
    n_antes = len(largo)
    largo = largo.drop_duplicates(ignore_index=True)
    if len(largo) < n_antes:
        logger.info("%s: se descartaron %s filas de participación duplicadas", path.name, n_antes - len(largo))
    logger.info("Participaciones cargadas: %s filas (%s combos)", len(largo),
                largo.groupby(["Parametro", "TECHNOLOGY", "FUEL"], dropna=False).ngroups)
    return largo[id_cols + ["Año", "Participacion"]]


def validar_participaciones(df_pct: pd.DataFrame, tolerancia: float = 1e-3) -> pd.DataFrame:
    """Grupos Parametro+TECHNOLOGY+FUEL+Año cuya suma entre regiones no es ≈ 1.

    Devuelve el detalle y emite warnings; el llamador decide si continuar.
    """
    sumas = df_pct.groupby(["Parametro", "TECHNOLOGY", "FUEL", "Año"], dropna=False)["Participacion"].sum()
    malas = sumas[(sumas - 1).abs() > tolerancia].reset_index().rename(
        columns={"Participacion": "Suma_Participacion"})
    if not malas.empty:
        logger.warning("%s grupos de participación no suman 1.0 (tolerancia %s)", len(malas), tolerancia)
    return malas


def _participacion_combo(
    df_pct: pd.DataFrame,
    parametro: str,
    tech: str | None,
    fuel: str | None,
    anios: list[int],
) -> pd.DataFrame:
    """Participaciones [Region, Año, Participacion] para un combo, expandidas a `anios`.

    Busca primero el combo exacto (param, tech, fuel); admite filas donde el
    combo se definió solo por TECHNOLOGY o solo por FUEL, y como último recurso
    la participación comodín '*' del parámetro (aplica a cualquier combo). Las
    participaciones constantes (Año=-1) se difunden a todos los años; en las
    por-año, los años faltantes heredan el último año disponible (ffill).

    Un año cuya participación suma 0 entre todas las regiones se considera
    **sin dato** (así deja las filas-plantilla `00_Generar_Mapeo.ipynb`) y se
    descarta: repartir por 0 escribiría ceros donde en realidad no hay reparto
    definido — en un upper-limit eso convierte "sin dato" en un límite duro de
    0. Si todos los años quedan descartados, el combo cuenta como sin
    participación (ValueError, igual que si no existiera en el archivo).
    Lanza ValueError descriptivo si el combo no existe en el archivo.
    """
    m = df_pct["Parametro"] == parametro

    def _match(tv, fv):
        mt = df_pct["TECHNOLOGY"].isna() if tv is None else (df_pct["TECHNOLOGY"] == tv)
        mf = df_pct["FUEL"].isna() if fv is None else (df_pct["FUEL"] == fv)
        return df_pct[m & mt & mf]

    filas = _match(tech, fuel)
    if filas.empty and tech is not None and fuel is not None:
        # parámetro indexado por ambos, pero el archivo definió el combo por una sola dimensión
        filas = _match(None, fuel)
        if filas.empty:
            filas = _match(tech, None)
    if filas.empty:
        filas = _match(COMODIN, COMODIN)
    if filas.empty:
        raise ValueError(
            f"Participación no encontrada para Parametro={parametro!r}, "
            f"TECHNOLOGY={tech!r}, FUEL={fuel!r} en el archivo de participaciones. "
            f"Agregue la fila (regiones deben sumar 1.0) o excluya el combo del filtro."
        )

    constantes = filas[filas["Año"] == ANIO_CONSTANTE]
    por_anio = filas[filas["Año"] != ANIO_CONSTANTE]
    if not por_anio.empty:
        pivote = por_anio.pivot_table(index="Region", columns="Año",
                                      values="Participacion", aggfunc="first")
        pivote = pivote.reindex(columns=sorted(set(pivote.columns) | set(anios)))
        pivote = pivote.ffill(axis=1)[list(anios)]
        salida = pivote.reset_index().melt(id_vars="Region", var_name="Año",
                                           value_name="Participacion")
        salida = salida.dropna(subset=["Participacion"])
    else:
        # Solo constantes: difundir a todos los años solicitados
        salida = pd.concat(
            [constantes.assign(Año=a)[["Region", "Año", "Participacion"]] for a in anios],
            ignore_index=True,
        )

    # Años-plantilla (participación toda en 0) = sin dato: se descartan
    suma_anio = salida.groupby("Año")["Participacion"].transform(lambda s: s.abs().sum())
    salida = salida[suma_anio > 1e-12]
    if salida.empty:
        raise ValueError(
            f"Participación definida pero toda en 0 (fila-plantilla sin datos) para "
            f"Parametro={parametro!r}, TECHNOLOGY={tech!r}, FUEL={fuel!r}: "
            f"se trata como sin participación."
        )
    return salida


def _log_entry(nivel: str, parametro: str, tech, fuel, mensaje: str) -> dict:
    return {"Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Nivel": nivel,
            "Parametro": parametro, "TECHNOLOGY": tech, "FUEL": fuel, "Mensaje": mensaje}


def regionalizar(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    df_pct: pd.DataFrame,
    parametros: list[str],
    params_otoole: dict[str, dict],
    cfg: dict,
    tecnologias_filtro: list[str] | None = None,
    fuels_filtro: list[str] | None = None,
    modo_filtro: str = "exacto",
    years_filtro: list[int] | None = None,
) -> dict:
    """Regionaliza los parámetros solicitados del SAND Nacional.

    Para cada fila nacional que pase los filtros:
    1. Verifica que el código exista en el regional con algún prefijo
       (si no, se omite y queda en el log).
    2. Aditivos: multiplica por la participación de cada región (los valores
       centinela se copian). Intensivos: copia el valor nacional a cada región.
    3. Construye filas SAND con los códigos prefijados (TECHNOLOGY y/o FUEL
       según los índices del parámetro en config_depurado.yaml).

    Returns dict:
        sands: {parametro: DataFrame en formato SAND}
        log: DataFrame del log estructurado (éxitos, omitidos, advertencias)
        participaciones_invalidas: grupos cuya suma no es ~1
    """
    from comparador import aplicar_filtro  # evita duplicar la lógica de filtros

    prefijo_region = cfg["prefijo_region"]
    tolerancia_pct = cfg.get("tolerancia_participacion", 1e-3)
    log: list[dict] = []

    # La columna Región del archivo puede traer nombres (Antioquia) o prefijos (AN)
    mapa_region = dict(prefijo_region)
    mapa_region.update({v: v for v in prefijo_region.values()})
    df_pct = df_pct.copy()
    sin_mapa = ~df_pct["Region"].isin(mapa_region)
    if sin_mapa.any():
        desconocidas = sorted(df_pct.loc[sin_mapa, "Region"].unique())
        raise ValueError(f"Regiones del archivo de participaciones sin prefijo definido: {desconocidas}")
    df_pct["Region"] = df_pct["Region"].map(mapa_region)

    participaciones_invalidas = validar_participaciones(df_pct, tolerancia_pct)
    for _, fila in participaciones_invalidas.iterrows():
        log.append(_log_entry("ADVERTENCIA", str(fila["Parametro"]), fila["TECHNOLOGY"], fila["FUEL"],
                              f"Suma de participaciones = {fila['Suma_Participacion']:.6f} (≠ 1.0) "
                              f"en Año={fila['Año']}"))

    anios_sand = columnas_anio(df_nacional)
    columnas_ref = df_nacional.columns.tolist()
    # Códigos regionales existentes (para la verificación de correspondencia)
    codigos_reg_tech = set(df_regional["TECHNOLOGY"].dropna().astype(str).str.strip())
    codigos_reg_fuel = set(df_regional["FUEL"].dropna().astype(str).str.strip())

    desconocidos = [p for p in parametros if p not in params_otoole]
    if desconocidos:
        raise ValueError(f"Parámetros no definidos en config_depurado.yaml: {desconocidos}")

    sands: dict[str, pd.DataFrame] = {}
    for parametro in parametros:
        spec = params_otoole[parametro]
        indices = spec["indices"]
        tiene_year = "YEAR" in indices
        usa_tech = "TECHNOLOGY" in indices
        usa_fuel = "FUEL" in indices
        es_intensivo = parametro in cfg["parametros_intensivos"]

        nac_p = df_nacional[df_nacional["Parameter"] == parametro].copy()
        if usa_tech:
            nac_p = aplicar_filtro(nac_p, "TECHNOLOGY", tecnologias_filtro or [], modo_filtro)
        if usa_fuel:
            nac_p = aplicar_filtro(nac_p, "FUEL", fuels_filtro or [], modo_filtro)
        if nac_p.empty:
            log.append(_log_entry("ADVERTENCIA", parametro, None, None,
                                  "Ninguna fila nacional pasa los filtros"))
            continue

        if tiene_year:
            anios_out = [c for c in anios_sand
                         if years_filtro is None or int(c) in years_filtro]
            anios_int = [int(c) for c in anios_out]
        else:
            anios_out, anios_int = [TIME_INDEP_COL], [ANIO_CONSTANTE]

        filas_out: list[dict] = []
        n_ok = n_omitidos = 0
        for _, row in nac_p.iterrows():
            tech = normalize_text(row["TECHNOLOGY"]) if usa_tech else None
            fuel = normalize_text(row["FUEL"]) if usa_fuel else None

            # 1. El código debe existir regionalizado con al menos un prefijo
            codigo, universo = (tech, codigos_reg_tech) if usa_tech else (fuel, codigos_reg_fuel)
            regiones_presentes = [pref for pref in prefijo_region.values()
                                  if f"{pref}_{codigo}" in universo]
            if not regiones_presentes:
                log.append(_log_entry("OMITIDO", parametro, tech, fuel,
                                      "No existe con ningún prefijo de región en el escenario regional"))
                n_omitidos += 1
                continue

            # 2/3/4. Valores por región.
            # Intensivos: se copia a las regiones donde el código ya existe.
            # Aditivos: se emite una fila por región del archivo de participaciones.
            if es_intensivo:
                pct_combo = None
                regiones_emitir = regiones_presentes
            else:
                try:
                    pct_combo = _participacion_combo(df_pct, parametro, tech, fuel, anios_int)
                except ValueError as exc:
                    log.append(_log_entry("OMITIDO", parametro, tech, fuel, str(exc)))
                    n_omitidos += 1
                    continue
                pct_combo = pct_combo.set_index(["Region", "Año"])["Participacion"]
                regiones_emitir = sorted(set(pct_combo.index.get_level_values("Region")))

            for prefijo in regiones_emitir:
                fila = {c: pd.NA for c in columnas_ref}
                fila["Parameter"] = parametro
                fila["REGION"] = cfg["region_osemosys"]
                for col in ("EMISSION", "MODE_OF_OPERATION", "TIMESLICE", "STORAGE", "REGION2"):
                    if col in indices or normalize_text(row.get(col)) is not None:
                        fila[col] = row.get(col)
                if usa_tech:
                    fila["TECHNOLOGY"] = f"{prefijo}_{tech}"
                if usa_fuel:
                    fila["FUEL"] = f"{prefijo}_{fuel}"

                for col, anio in zip(anios_out, anios_int):
                    val = safe_float(row[col])
                    if val is None:
                        continue
                    if es_intensivo or es_centinela(val, cfg["valor_centinela"], cfg["centinela_patron_9s"]):
                        fila[col] = val  # intensivo o "sin límite": mismo valor por región
                    else:
                        pct = pct_combo.get((prefijo, anio))
                        if pct is None or pd.isna(pct):
                            continue  # región sin participación para este año -> sin dato
                        fila[col] = val * float(pct)
                filas_out.append(fila)
            n_ok += 1

        if filas_out:
            df_sand = pd.DataFrame(filas_out, columns=columnas_ref)
            # Recalcular la columna indicadora del SAND base, si existe
            for col in columnas_ref:
                if str(col).startswith("Tiene datos"):
                    vals = df_sand[anios_sand].apply(pd.to_numeric, errors="coerce")
                    df_sand[col] = (vals.notna() & (vals != 0)).any(axis=1).astype(int)
            sands[parametro] = df_sand
        log.append(_log_entry("OK", parametro, None, None,
                              f"{n_ok} combinaciones regionalizadas ({len(filas_out)} filas SAND), "
                              f"{n_omitidos} omitidas"))
        logger.info("%s: %s combos regionalizados, %s omitidos", parametro, n_ok, n_omitidos)

    return {"sands": sands, "log": pd.DataFrame(log),
            "participaciones_invalidas": participaciones_invalidas}


def escribir_sands(
    sands: dict[str, pd.DataFrame],
    carpeta: str | Path,
    descripcion: str,
    consolidado: bool = False,
) -> list[Path]:
    """Escribe los SAND reducidos: `SAND_{Parametro}_{descripcion}.xlsx` por
    parámetro, o un único `SAND_Consolidado_{descripcion}.xlsx`."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []
    if consolidado:
        df = pd.concat(sands.values(), ignore_index=True)
        rutas.append(escribir_sand(df, carpeta / f"SAND_Consolidado_{descripcion}.xlsx"))
    else:
        for parametro, df in sands.items():
            rutas.append(escribir_sand(df, carpeta / f"SAND_{parametro}_{descripcion}.xlsx"))
    for ruta in rutas:
        logger.info("SAND escrito: %s", ruta)
    return rutas


# ---------------------------------------------------------------------------
# Flujo con archivos de mapeo (Insumos/Mapeo/)
#
# Añade sobre `regionalizar`: (1) las regiones esperadas por código salen de
# mapeo_tech_fuel.xlsx en vez de asumirse las 7; (2) los renombramientos
# nacional -> regional (FUEL_REGIONAL, p. ej. ELC -> ELC003) se aplican al
# construir el código prefijado; (3) los códigos sin correspondencia o sin
# participación no abortan: se detectan antes (`detectar_omisiones`), el usuario
# decide qué hacer (`resolver_omisiones`) y `regionalizar_con_mapeo` aplica esa
# decisión. `regionalizar` se mantiene intacta para el flujo anterior.
# ---------------------------------------------------------------------------

# Si la fila nacional es toda ceros / solo-9s se copia (intensivo); si trae
# valores reales se reparte (aditivo) y exige participación.
PARAMS_CONDICIONALES = frozenset({
    "TotalAnnualMaxCapacityInvestment",
    "TotalTechnologyAnnualActivityUpperLimit",
    "TotalTechnologyModelPeriodActivityUpperLimit",
})

MOTIVO_SIN_CORRESPONDENCIA = "SIN_CORRESPONDENCIA_REGIONAL"
MOTIVO_SIN_PARTICIPACION = "SIN_PARTICIPACION"
MOTIVO_REGIONES_INCOMPLETAS = "REGIONES_INCOMPLETAS"
# Aditivo cuyo archivo de participaciones asigna % > 0 a una región donde el código
# NO existe en el regional. Sin control, la regionalización aditiva emitiría una fila
# para esa región inexistente (bug de sector residencial 2026-07). Por defecto se
# omite esa región (la fracción se pierde, como crear_existentes) y se reporta aquí.
MOTIVO_PARTICIPACION_REGION_INEXISTENTE = "PARTICIPACION_REGION_INEXISTENTE"
MOTIVO_EXCLUIDO_POR_MAPEO = "EXCLUIDO_POR_MAPEO"

DECISION_OMITIR = "omitir"
DECISION_CREAR_TODAS = "crear_todas"
DECISION_CREAR_REGIONES = "crear_regiones:"   # + "AN,CA,..."
DECISION_CREAR_EXISTENTES = "crear_existentes"
# Como crear_existentes pero repartiendo uniforme 1/N entre las regiones donde el
# código ya existe. Pensado para SIN_PARTICIPACION: ahí no hay % en el archivo, así
# que 'crear_existentes' (que nunca inventa reparto) no emitiría nada.
DECISION_CREAR_EXISTENTES_UNIF = "crear_existentes_uniforme"
DECISION_EXCLUIDO = "excluido_por_mapeo"


def cargar_mapeo_regional(carpeta: str | Path) -> dict:
    """Lee Insumos/Mapeo/ y arma los diccionarios de consulta del flujo.

    Devuelve, para TECHNOLOGY y FUEL por separado:
      - `regiones_*`: {código nacional: [prefijos donde debe existir]} según las
        columnas 0/1 de mapeo_tech_fuel.xlsx.
      - `rename_*`: {código nacional: código base regional} (TECHNOLOGY_REGIONAL /
        FUEL_REGIONAL); p. ej. ELC -> ELC003, y se escribirá `AN_ELC003`.
      - `obs_*`: texto de diccionario_tech/fuel.xlsx para mostrar al decidir.
    `disponible` es False si no hay mapeo_tech_fuel.xlsx (todo degrada a las 7
    regiones, igual que el flujo anterior).
    """
    from comparador import cargar_mapeos

    mapeos = cargar_mapeos(carpeta)
    mapa = mapeos.get("mapeo")
    salida = {
        "regiones_tech": {}, "regiones_fuel": {},
        "rename_tech": {}, "rename_fuel": {},
        "obs_tech": {}, "obs_fuel": {},
        "disponible": mapa is not None,
    }

    if mapa is not None:
        cols_reg = [c for c in REGIONES if c in mapa.columns]
        for _, fila in mapa.iterrows():
            regiones = [c for c in cols_reg if safe_float(fila.get(c)) == 1.0]
            tech = normalize_text(fila.get("TECHNOLOGY"))
            fuel = normalize_text(fila.get("FUEL"))
            if tech is not None:
                salida["regiones_tech"][tech] = regiones
                if normalize_text(fila.get("TECHNOLOGY_REGIONAL")) is not None:
                    salida["rename_tech"][tech] = normalize_text(fila["TECHNOLOGY_REGIONAL"])
            if fuel is not None:
                salida["regiones_fuel"][fuel] = regiones
                if normalize_text(fila.get("FUEL_REGIONAL")) is not None:
                    salida["rename_fuel"][fuel] = normalize_text(fila["FUEL_REGIONAL"])
        logger.info("Mapeo regional: %s tecnologías, %s fuels, %s renombramientos",
                    len(salida["regiones_tech"]), len(salida["regiones_fuel"]),
                    len(salida["rename_tech"]) + len(salida["rename_fuel"]))
    else:
        logger.warning("Sin mapeo_tech_fuel.xlsx: se asumen las 7 regiones para todo código")

    # Observaciones de los diccionarios (solo informativas, para MODO_INTERACTIVO)
    for clave, col_nac, destino in (("diccionario_tech", "TECHNOLOGY_NACIONAL", "obs_tech"),
                                    ("diccionario_fuel", "FUEL_NACIONAL", "obs_fuel")):
        dicc = mapeos.get(clave)
        if dicc is None or col_nac not in dicc.columns:
            continue
        cols_txt = [c for c in ("EQUIVALENCIA", "OBSERVACION") if c in dicc.columns]
        for _, fila in dicc.iterrows():
            cod = normalize_text(fila.get(col_nac))
            if cod is None or cod in salida[destino]:
                continue
            texto = " | ".join(str(fila[c]) for c in cols_txt
                               if normalize_text(fila.get(c)) is not None)
            if texto:
                salida[destino][cod] = texto
    return salida


def _regiones_pct(df_pct: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Normaliza la columna Region del archivo de participaciones a prefijos."""
    mapa = dict(cfg["prefijo_region"])
    mapa.update({v: v for v in cfg["prefijo_region"].values()})
    df = df_pct.copy()
    sin_mapa = ~df["Region"].isin(mapa)
    if sin_mapa.any():
        raise ValueError("Regiones del archivo de participaciones sin prefijo definido: "
                         f"{sorted(df.loc[sin_mapa, 'Region'].unique())}")
    df["Region"] = df["Region"].map(mapa)
    return df


def _buscar_participacion(df_pct, parametro, tech, fuel, anios):
    """`_participacion_combo` sin excepción: None si el combo no está definido."""
    try:
        return _participacion_combo(df_pct, parametro, tech, fuel, anios)
    except ValueError:
        return None


def detectar_omisiones(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    df_pct: pd.DataFrame,
    parametros: list[str],
    params_otoole: dict[str, dict],
    cfg: dict,
    mapeo: dict,
    sectores_a_excluir: list[str] | None = None,
    tecnologias_filtro: list[str] | None = None,
    fuels_filtro: list[str] | None = None,
    modo_filtro: str = "exacto",
) -> pd.DataFrame:
    """Casos problemáticos antes de regionalizar, uno por (Parámetro, código).

    Motivos: SIN_CORRESPONDENCIA_REGIONAL (el código nacional no aparece en el
    regional con ningún prefijo), SIN_PARTICIPACION (existe pero no hay % para
    repartirlo), REGIONES_INCOMPLETAS (existe solo en algunas de las regiones
    que el mapeo espera) y EXCLUIDO_POR_MAPEO (el mapeo dice que no debe existir
    en ninguna región — esperado, no se pregunta). Las tecnologías TRN* se
    saltan en silencio: son nuevas por diseño del modelo regional.

    `tecnologias_filtro` / `fuels_filtro` (con `modo_filtro` 'exacto' | 'contiene')
    acotan qué códigos se revisan; cada uno solo aplica a los parámetros
    indexados por esa dimensión (un filtro de FUEL no toca un parámetro que solo
    se indexa por TECHNOLOGY). En modo 'contiene' sirven además para filtrar por
    sector (ej. tecnologias_filtro=['IND']).

    `Decision` sale con el valor por defecto ('omitir', o 'excluido_por_mapeo');
    `resolver_omisiones` la sobreescribe.
    """
    from comparador import aplicar_filtro  # filtros por dimensión (mismo que comparación)

    sectores_a_excluir = sectores_a_excluir or []
    parametros = list(dict.fromkeys(parametros))   # dedup preservando orden (evita casos duplicados)
    df_pct = _regiones_pct(df_pct, cfg)
    prefijos = list(cfg["prefijo_region"].values())
    cod_reg_tech = set(df_regional["TECHNOLOGY"].dropna().astype(str).str.strip())
    cod_reg_fuel = set(df_regional["FUEL"].dropna().astype(str).str.strip())
    cols_anio_str = columnas_anio(df_nacional)
    anios_sand = [int(c) for c in cols_anio_str]

    filas: list[dict] = []
    for parametro in parametros:
        spec = params_otoole.get(parametro)
        if spec is None:
            continue
        indices = spec["indices"]
        usa_tech, usa_fuel = "TECHNOLOGY" in indices, "FUEL" in indices
        anios = anios_sand if "YEAR" in indices else [ANIO_CONSTANTE]
        cols_valor = cols_anio_str if "YEAR" in indices else [TIME_INDEP_COL]
        es_intensivo = parametro in cfg["parametros_intensivos"]
        es_condicional = parametro in PARAMS_CONDICIONALES

        nac_p = df_nacional[df_nacional["Parameter"] == parametro]
        if usa_tech:
            nac_p = aplicar_filtro(nac_p, "TECHNOLOGY", tecnologias_filtro or [], modo_filtro)
        if usa_fuel:
            nac_p = aplicar_filtro(nac_p, "FUEL", fuels_filtro or [], modo_filtro)
        vistos: set[tuple] = set()
        for _, row in nac_p.iterrows():
            tech = normalize_text(row["TECHNOLOGY"]) if usa_tech else None
            fuel = normalize_text(row["FUEL"]) if usa_fuel else None
            if (tech, fuel) in vistos:
                continue
            vistos.add((tech, fuel))

            codigo = tech if usa_tech else fuel
            if codigo is None:
                continue
            if is_trade_technology(codigo):      # TRN*: nuevas por diseño
                continue
            if any(s in codigo for s in sectores_a_excluir):
                continue

            universo = cod_reg_tech if usa_tech else cod_reg_fuel
            renames = mapeo["rename_tech"] if usa_tech else mapeo["rename_fuel"]
            regs_map = mapeo["regiones_tech"] if usa_tech else mapeo["regiones_fuel"]
            base_regional = renames.get(codigo, codigo)

            existentes = [p for p in prefijos if f"{p}_{base_regional}" in universo]
            if mapeo["disponible"]:
                esperadas = regs_map.get(codigo, [])
            else:
                esperadas = prefijos
            faltantes = sorted(set(esperadas) - set(existentes))

            # Un condicional (upper/max) cuya fila nacional es toda 0/centinela
            # NO requiere participación: se copia (0 se reparte como 0). Solo
            # exige % si tiene límites reales que haya que repartir.
            pct_df = _buscar_participacion(df_pct, parametro, tech, fuel, anios)
            solo_ceros = es_condicional and _es_fila_condicional_intensiva(row, cols_valor, cfg)
            tiene_pct = es_intensivo or solo_ceros or pct_df is not None

            # Participación asignada a regiones donde el código NO existe. Solo
            # afecta a los ADITIVOS: la rama aditiva de regionalizar_con_mapeo emite
            # según las regiones del archivo de participaciones, así que sin este
            # control crearía filas para regiones inexistentes. Intensivos y
            # condicionales ya acotan a las regiones existentes, no aplica.
            regiones_pct_fuera: list[str] = []
            if pct_df is not None and existentes and not es_intensivo and not es_condicional:
                pos = pct_df[pct_df["Participacion"].map(safe_float).fillna(0) > 0]
                regiones_pct_fuera = sorted(set(pos["Region"]) - set(existentes))

            if not esperadas and not existentes:
                motivo, decision, faltantes_rep = MOTIVO_EXCLUIDO_POR_MAPEO, DECISION_EXCLUIDO, faltantes
            elif not existentes:
                motivo, decision, faltantes_rep = MOTIVO_SIN_CORRESPONDENCIA, DECISION_OMITIR, faltantes
            elif not tiene_pct:
                motivo, decision, faltantes_rep = MOTIVO_SIN_PARTICIPACION, DECISION_OMITIR, faltantes
            elif regiones_pct_fuera:
                # Fase 2: existe en algunas regiones, pero hay % para una donde no existe.
                # Default = crear_existentes: emitir en las regiones donde SÍ existe y
                # soltar la fracción de las inexistentes (no 'omitir', que saltaría el
                # código entero). El usuario puede decidir crearlo en esas regiones.
                motivo, decision, faltantes_rep = (MOTIVO_PARTICIPACION_REGION_INEXISTENTE,
                                                   DECISION_CREAR_EXISTENTES, regiones_pct_fuera)
            elif faltantes:
                motivo, decision, faltantes_rep = MOTIVO_REGIONES_INCOMPLETAS, DECISION_OMITIR, faltantes
            else:
                continue   # sin problema: no entra al reporte de omisiones

            obs = (mapeo["obs_tech"] if usa_tech else mapeo["obs_fuel"]).get(codigo, "")
            filas.append({
                "Parametro": parametro, "TECHNOLOGY": tech, "FUEL": fuel,
                "Motivo": motivo,
                "Regiones_Existentes": ",".join(existentes),
                "Regiones_Faltantes": ",".join(faltantes_rep),
                "Nombre_Regional": base_regional if base_regional != codigo else "",
                "Observacion": obs,
                "Decision": decision,
            })

    cols = ["Parametro", "TECHNOLOGY", "FUEL", "Motivo", "Regiones_Existentes",
            "Regiones_Faltantes", "Nombre_Regional", "Observacion", "Decision"]
    om = pd.DataFrame(filas, columns=cols)
    if not om.empty:
        logger.info("Omisiones potenciales: %s (%s)", len(om),
                    om["Motivo"].value_counts().to_dict())
    return om


def resolver_omisiones(omisiones: pd.DataFrame, interactivo: bool = False,
                       entrada=input, salida=print) -> pd.DataFrame:
    """Fija la columna `Decision` de cada omisión.

    Con `interactivo=False` deja los valores por defecto (todo 'omitir', salvo
    los EXCLUIDO_POR_MAPEO). Con `interactivo=True` pregunta caso por caso —
    los EXCLUIDO_POR_MAPEO nunca se preguntan, son esperados:
      [1] crear en todas las regiones faltantes (participación uniforme 1/N)
      [2] crear solo en las regiones que se indiquen (ej. `AN,CA`)
      [3] omitir (en PARTICIPACION_REGION_INEXISTENTE: omitir solo esas regiones)
      [4] omitir todos los casos restantes de este mismo Motivo
      [5] crear en las regiones existentes; el reparto depende del Motivo:
          - REGIONES_INCOMPLETAS: participaciones del archivo tal cual (sin
            renormalizar) — la fracción de las faltantes se pierde y sin %
            aplicable no se inventa reparto (queda vacío con ADVERTENCIA).
          - SIN_PARTICIPACION: uniforme 1/N entre las existentes (no hay % en el
            archivo que reutilizar).

    Tras cualquier opción explícita ([1]/[2]/[3]/[5]) se ofrece aplicarla a
    todos los casos restantes del mismo Motivo (s/N); la [4] es el atajo directo
    para el 'no crear' del motivo. El lote se guarda en `accion_por_motivo` y los
    casos que lo heredan ya no se preguntan.
    """
    om = omisiones.copy()
    if om.empty or not interactivo:
        return om

    accion_por_motivo: dict[str, str] = {}
    pendientes = list(om.index[om["Motivo"] != MOTIVO_EXCLUIDO_POR_MAPEO])
    for pos, idx in enumerate(pendientes, start=1):
        fila = om.loc[idx]
        if fila["Motivo"] in accion_por_motivo:
            om.at[idx, "Decision"] = accion_por_motivo[fila["Motivo"]]
            continue
        es_pct_fuera = fila["Motivo"] == MOTIVO_PARTICIPACION_REGION_INEXISTENTE
        salida(f"\n[{pos}/{len(pendientes)}] {fila['Motivo']} — {fila['Parametro']}")
        salida(f"  TECHNOLOGY={fila['TECHNOLOGY']}  FUEL={fila['FUEL']}")
        salida(f"  Existe en : {fila['Regiones_Existentes'] or '(ninguna)'}")
        etiqueta_falta = "% en regiones sin el código" if es_pct_fuera else "Falta en "
        salida(f"  {etiqueta_falta}: {fila['Regiones_Faltantes'] or '(ninguna)'}")
        if fila["Nombre_Regional"]:
            salida(f"  Nombre regional: {fila['Nombre_Regional']}")
        if fila["Observacion"]:
            salida(f"  Diccionario: {fila['Observacion']}")
        if es_pct_fuera:
            # Aquí la participación de esas regiones SÍ existe en el archivo, así que
            # "crear" usa ese % real (no 1/N); "omitir" pierde esa fracción.
            menu = ("  [1] crear el código también en esas regiones (usa el % del archivo)  "
                    "[2] crear en regiones...  [3] omitir esas regiones (se pierde su %)  "
                    "[4] omitir todos los de este motivo")
        else:
            menu = ("  [1] crear en todas las faltantes (1/N)  [2] crear en regiones...  "
                    "[3] omitir  [4] omitir todos los de este motivo")
            if fila["Motivo"] == MOTIVO_REGIONES_INCOMPLETAS:
                menu += "  [5] crear solo en existentes (% tal cual)"
            elif fila["Motivo"] == MOTIVO_SIN_PARTICIPACION:
                # Sin % en el archivo, "tal cual" no emitiría nada: aquí el reparto es 1/N.
                menu += "  [5] crear en todas las existentes (1/N)"
        salida(menu)
        # Para el motivo de participación en región inexistente el "no crear" NO es
        # omitir el código entero, sino crear_existentes (emitir donde sí existe y
        # soltar la fracción de las regiones inexistentes).
        default_no_crear = DECISION_CREAR_EXISTENTES if es_pct_fuera else DECISION_OMITIR
        decision = default_no_crear
        # El atajo por lotes se ofrece tras cualquier opción explícita, incluida la [3]
        # ("no crear"), que en PARTICIPACION_REGION_INEXISTENTE es la respuesta habitual
        # y antes solo se podía aplicar en lote pasando por la [4].
        ofrecer_lote = False
        try:
            resp = str(entrada("  Opción [3]: ")).strip()
            if resp == "1":
                decision, ofrecer_lote = DECISION_CREAR_TODAS, True
            elif resp == "2":
                crudo = str(entrada("  Regiones (ej. AN,CA): ")).strip().upper()
                regs = ",".join(r.strip() for r in crudo.split(",") if r.strip() in REGIONES)
                if regs:
                    decision, ofrecer_lote = DECISION_CREAR_REGIONES + regs, True
            elif resp == "3":
                ofrecer_lote = True
            elif resp == "4":
                accion_por_motivo[fila["Motivo"]] = default_no_crear
            elif resp == "5" and fila["Motivo"] == MOTIVO_REGIONES_INCOMPLETAS:
                decision, ofrecer_lote = DECISION_CREAR_EXISTENTES, True
            elif resp == "5" and fila["Motivo"] == MOTIVO_SIN_PARTICIPACION:
                decision, ofrecer_lote = DECISION_CREAR_EXISTENTES_UNIF, True
            if ofrecer_lote:
                lote = str(entrada("  ¿Aplicar a todos los restantes de este motivo? (s/N): ")).strip().lower()
                if lote == "s":
                    accion_por_motivo[fila["Motivo"]] = decision
        except (EOFError, KeyboardInterrupt):
            salida("\n  Entrada no disponible; se omiten los casos restantes.")
            om.at[idx, "Decision"] = default_no_crear
            break
        om.at[idx, "Decision"] = decision
    return om


def _decisiones_dict(omisiones: pd.DataFrame) -> dict:
    if omisiones is None or omisiones.empty:
        return {}
    # normalize_text: iterrows convierte None -> NaN y la clave (param, NaN, fuel)
    # jamás casaría con la (param, None, fuel) que consulta regionalizar_con_mapeo
    # — las decisiones se ignorarían en silencio.
    return {(f["Parametro"], normalize_text(f["TECHNOLOGY"]), normalize_text(f["FUEL"])):
            f["Decision"] for _, f in omisiones.iterrows()}


def _regiones_de_decision(decision, faltantes: list[str], existentes: list[str]) -> list[str]:
    """Regiones a crear según la decisión tomada en la pre-validación.

    `crear_existentes` devuelve [] a propósito: no crea regiones nuevas, solo
    autoriza emitir en las que ya existen (se maneja aparte en
    `regionalizar_con_mapeo`). `crear_existentes_uniforme` sí devuelve las
    existentes: se tratan como "creadas" para que hereden el reparto 1/N cuando
    no hay participación en el archivo.
    """
    if decision == DECISION_CREAR_TODAS:
        return faltantes or existentes
    if decision == DECISION_CREAR_EXISTENTES_UNIF:
        return list(existentes)
    if isinstance(decision, str) and decision.startswith(DECISION_CREAR_REGIONES):
        pedidas = [r.strip() for r in decision[len(DECISION_CREAR_REGIONES):].split(",") if r.strip()]
        return [r for r in pedidas if r in REGIONES]
    return []


def _es_fila_condicional_intensiva(row, cols_valor, cfg) -> bool:
    """Fila condicional que se copia en vez de repartirse: todo ceros o solo-9s."""
    valores = [safe_float(row[c]) for c in cols_valor]
    valores = [v for v in valores if v is not None]
    if not valores:
        return True
    return all(v == 0 or es_centinela(v, cfg["valor_centinela"], cfg["centinela_patron_9s"])
               for v in valores)


def regionalizar_con_mapeo(
    df_nacional: pd.DataFrame,
    df_regional: pd.DataFrame,
    df_pct: pd.DataFrame,
    parametros: list[str],
    params_otoole: dict[str, dict],
    cfg: dict,
    mapeo: dict,
    omisiones: pd.DataFrame | None = None,
    years_filtro: list[int] | None = None,
    sectores_a_excluir: list[str] | None = None,
    tecnologias_filtro: list[str] | None = None,
    fuels_filtro: list[str] | None = None,
    modo_filtro: str = "exacto",
) -> dict:
    """Regionaliza usando los archivos de mapeo y las decisiones de omisión.

    Clasificación por parámetro: intensivo (copia el valor nacional a cada
    región donde el código exista según el mapeo), aditivo (nacional ×
    participación) y condicional / upper-max (híbrido **por año**): los años en
    0 o centinela se copian tal cual a las regiones existentes —un límite de 0
    significa "0 en todas las regiones" y no requiere participación— y solo los
    años con valor real se reparten. Por eso un condicional NUNCA se omite por
    falta de participación mientras existan regiones equivalentes: sus años en 0
    quedan fijados y los años reales sin participación se dejan vacíos (con
    ADVERTENCIA en el log).

    Los códigos con problema consultan su `Decision` en `omisiones`: 'omitir'
    los deja fuera (queda en el log), 'crear_todas' / 'crear_regiones:AN,CA' los
    genera con participación uniforme 1/N sobre esas regiones, y
    'crear_existentes' (piloto de REGIONES_INCOMPLETAS) emite solo en las
    regiones donde el código ya existe con las participaciones del archivo
    **tal cual, sin renormalizar** — la fracción de las regiones faltantes se
    pierde (la suma regional queda por debajo del nacional, a propósito) y sin
    % aplicable NO se inventa reparto uniforme: se omite con ADVERTENCIA
    (aditivos) o solo se fijan los años 0/centinela (condicionales).
    'crear_existentes_uniforme' (piloto de SIN_PARTICIPACION) también acota a las
    regiones existentes, pero las trata como creadas: sin % en el archivo reparte
    uniforme 1/N entre ellas. Los renombramientos del mapeo (ELC -> ELC003) se
    aplican al prefijar.

    `tecnologias_filtro` / `fuels_filtro` (con `modo_filtro` 'exacto' | 'contiene')
    acotan qué códigos se regionalizan; cada uno solo aplica a los parámetros
    indexados por esa dimensión. En modo 'contiene' filtran también por sector
    (ej. tecnologias_filtro=['IND']). Deben coincidir con los que se pasaron a
    `detectar_omisiones` para que las decisiones cuadren con lo regionalizado.

    Returns dict: sands, log, resumen (por parámetro) y participaciones_invalidas.
    """
    from comparador import aplicar_filtro  # filtros por dimensión (mismo que comparación)

    sectores_a_excluir = sectores_a_excluir or []
    parametros = list(dict.fromkeys(parametros))   # dedup preservando orden (evita reprocesar)
    df_pct = _regiones_pct(df_pct, cfg)
    prefijos = list(cfg["prefijo_region"].values())
    decisiones = _decisiones_dict(omisiones)

    log: list[dict] = []
    participaciones_invalidas = validar_participaciones(
        df_pct, cfg.get("tolerancia_participacion", 1e-3))
    for _, fila in participaciones_invalidas.iterrows():
        log.append(_log_entry("ADVERTENCIA", str(fila["Parametro"]), fila["TECHNOLOGY"],
                              fila["FUEL"],
                              f"Suma de participaciones = {fila['Suma_Participacion']:.6f} "
                              f"(distinta de 1.0) en Año={fila['Año']}"))

    anios_sand = columnas_anio(df_nacional)
    columnas_ref = df_nacional.columns.tolist()
    # Columnas de año a conservar en el SAND de salida: se EXCLUYEN (no solo se dejan
    # vacías) las que quedan fuera de years_filtro — p. ej. 2055 cuando ANIO_MAX=2054.
    # Las columnas de dimensión (no-año) se mantienen todas.
    anios_salida = [c for c in anios_sand if years_filtro is None or int(c) in years_filtro]
    columnas_salida = [c for c in columnas_ref if c not in anios_sand or c in set(anios_salida)]
    cod_reg_tech = set(df_regional["TECHNOLOGY"].dropna().astype(str).str.strip())
    cod_reg_fuel = set(df_regional["FUEL"].dropna().astype(str).str.strip())

    desconocidos = [p for p in parametros if p not in params_otoole]
    if desconocidos:
        raise ValueError(f"Parámetros no definidos en config_depurado.yaml: {desconocidos}")

    sands: dict[str, pd.DataFrame] = {}
    resumen: list[dict] = []
    for parametro in parametros:
        spec = params_otoole[parametro]
        indices = spec["indices"]
        tiene_year = "YEAR" in indices
        usa_tech, usa_fuel = "TECHNOLOGY" in indices, "FUEL" in indices
        es_intensivo = parametro in cfg["parametros_intensivos"]
        es_condicional = parametro in PARAMS_CONDICIONALES

        nac_p = df_nacional[df_nacional["Parameter"] == parametro].copy()
        if usa_tech:
            nac_p = aplicar_filtro(nac_p, "TECHNOLOGY", tecnologias_filtro or [], modo_filtro)
        if usa_fuel:
            nac_p = aplicar_filtro(nac_p, "FUEL", fuels_filtro or [], modo_filtro)
        if nac_p.empty:
            log.append(_log_entry("ADVERTENCIA", parametro, None, None,
                                  "Sin filas nacionales que pasen los filtros"))
            resumen.append({"Parametro": parametro, "Filas_SAND": 0, "Combos_OK": 0,
                            "Combos_Omitidos": 0, "Combos_Creados": 0})
            continue

        if tiene_year:
            anios_out = [c for c in anios_sand if years_filtro is None or int(c) in years_filtro]
            anios_int = [int(c) for c in anios_out]
        else:
            anios_out, anios_int = [TIME_INDEP_COL], [ANIO_CONSTANTE]
        if not anios_out:
            log.append(_log_entry("ADVERTENCIA", parametro, None, None,
                                  "Ningún año pasa YEARS_FILTRO"))
            resumen.append({"Parametro": parametro, "Filas_SAND": 0, "Combos_OK": 0,
                            "Combos_Omitidos": 0, "Combos_Creados": 0})
            continue

        filas_out: list[dict] = []
        n_ok = n_omitidos = n_creados = 0
        for _, row in nac_p.iterrows():
            tech = normalize_text(row["TECHNOLOGY"]) if usa_tech else None
            fuel = normalize_text(row["FUEL"]) if usa_fuel else None
            codigo = tech if usa_tech else fuel
            if codigo is None:
                continue
            if is_trade_technology(codigo):
                continue          # TRN*: nuevas por diseño, no son omisiones
            if any(s in codigo for s in sectores_a_excluir):
                continue

            universo = cod_reg_tech if usa_tech else cod_reg_fuel
            renames = mapeo["rename_tech"] if usa_tech else mapeo["rename_fuel"]
            regs_map = mapeo["regiones_tech"] if usa_tech else mapeo["regiones_fuel"]
            base_regional = renames.get(codigo, codigo)

            existentes = [p for p in prefijos if f"{p}_{base_regional}" in universo]
            esperadas = regs_map.get(codigo, []) if mapeo["disponible"] else prefijos
            faltantes = sorted(set(esperadas) - set(existentes))

            decision = decisiones.get((parametro, tech, fuel))
            if decision == DECISION_EXCLUIDO:
                continue          # el mapeo dice que no debe existir: esperado
            creadas = _regiones_de_decision(decision, faltantes, existentes) if decision else []
            # crear_existentes: emitir solo en las regiones donde ya existe,
            # con las participaciones del archivo tal cual (sin renormalizar).
            crear_exist = decision == DECISION_CREAR_EXISTENTES
            # Los condicionales (upper/max) no se omiten aunque falte la
            # participación: sus años en 0/centinela deben quedar fijados en las
            # regiones existentes (0 no requiere reparto). El resto sí se omite.
            if decision == DECISION_OMITIR and not creadas and not es_condicional:
                log.append(_log_entry("OMITIDO", parametro, tech, fuel,
                                      "Decisión de la pre-validación: omitir"))
                n_omitidos += 1
                continue

            # Clasificar la fila:
            # - Intensivo: copia el valor nacional a cada región existente.
            # - Condicional: híbrido por año — los años en 0/centinela se copian
            #   tal cual a las regiones existentes (un límite de 0 significa "0
            #   en todas las regiones", sin participación); solo los años con
            #   valor real se reparten con participación (si la hay).
            # - Aditivo: nacional × participación (se omite si falta).
            copia_valor = es_intensivo
            if es_intensivo:
                pct_combo = None
                regiones_emitir = sorted(set(existentes) | set(creadas))
            elif es_condicional:
                pct = _buscar_participacion(df_pct, parametro, tech, fuel, anios_int)
                if crear_exist and pct is not None:
                    # tal cual, sin renormalizar; sin % aplicable no se inventa
                    pct = pct[pct["Region"].isin(existentes)]
                    if pct.empty:
                        pct = None
                if pct is None and creadas:
                    n = len(creadas)   # crear sin % en el archivo: uniforme 1/N
                    pct = pd.DataFrame([{"Region": r, "Año": a, "Participacion": 1.0 / n}
                                        for r in creadas for a in anios_int])
                    log.append(_log_entry("CREADO", parametro, tech, fuel,
                                          f"Participación uniforme 1/{n} en {','.join(creadas)}"))
                elif pct is None and not _es_fila_condicional_intensiva(row, anios_out, cfg):
                    log.append(_log_entry("ADVERTENCIA", parametro, tech, fuel,
                                          "Condicional con límites reales sin participación: se "
                                          "fijan solo los años en 0/centinela en las regiones existentes"))
                pct_combo = (pct.set_index(["Region", "Año"])["Participacion"]
                             if pct is not None else None)
                regiones_emitir = sorted(set(existentes) | set(creadas))
            else:
                pct = _buscar_participacion(df_pct, parametro, tech, fuel, anios_int)
                if crear_exist:
                    # Solo regiones existentes, % del archivo tal cual (sin
                    # renormalizar): la fracción de las faltantes se pierde.
                    # Sin % aplicable no se inventa reparto uniforme.
                    if pct is not None:
                        pct = pct[pct["Region"].isin(existentes)]
                    if pct is None or pct.empty:
                        log.append(_log_entry("ADVERTENCIA", parametro, tech, fuel,
                                              "crear_existentes sin participación aplicable a las "
                                              "regiones existentes: no se emite"))
                        n_omitidos += 1
                        continue
                elif pct is None and not creadas:
                    log.append(_log_entry("OMITIDO", parametro, tech, fuel,
                                          "Sin participación definida y sin decisión de creación"))
                    n_omitidos += 1
                    continue
                elif pct is None:
                    # Decisión de crear sin % en el archivo: uniforme 1/N
                    n = len(creadas)
                    pct = pd.DataFrame([{"Region": r, "Año": a, "Participacion": 1.0 / n}
                                        for r in creadas for a in anios_int])
                    log.append(_log_entry("CREADO", parametro, tech, fuel,
                                          f"Participación uniforme 1/{n} en {','.join(creadas)}"))
                pct_combo = pct.set_index(["Region", "Año"])["Participacion"]
                regiones_pct = set(pct_combo.index.get_level_values("Region"))
                # Emitir solo donde el código EXISTE, más las regiones que una
                # decisión de creación haya autorizado. La participación asignada a
                # regiones inexistentes no se emite (crearía un código ausente del
                # modelo); esa fracción se pierde a propósito y se registra (motivo
                # PARTICIPACION_REGION_INEXISTENTE en la pre-validación).
                regiones_emitir = sorted((regiones_pct & set(existentes)) | set(creadas))
                descartadas = sorted(regiones_pct - set(regiones_emitir))
                if descartadas:
                    log.append(_log_entry("ADVERTENCIA", parametro, tech, fuel,
                                          f"Participación en regiones sin el código, descartada: "
                                          f"{','.join(descartadas)} (fracción perdida; decidir "
                                          "'crear' en la pre-validación para incluirlas)"))

            if not regiones_emitir:
                log.append(_log_entry("OMITIDO", parametro, tech, fuel,
                                      "Sin regiones destino (no existe y no se decidió crearlo)"))
                n_omitidos += 1
                continue
            if creadas:
                n_creados += 1
            if crear_exist:   # excluyente con `creadas` (siempre [] aquí)
                n_creados += 1
                log.append(_log_entry("CREADO", parametro, tech, fuel,
                                      f"Emitido solo en regiones existentes ({','.join(regiones_emitir)}): "
                                      "participaciones tal cual, sin renormalizar"))

            for prefijo in regiones_emitir:
                fila = {c: pd.NA for c in columnas_ref}
                fila["Parameter"] = parametro
                fila["REGION"] = cfg["region_osemosys"]
                for col in ("EMISSION", "MODE_OF_OPERATION", "TIMESLICE", "STORAGE", "REGION2"):
                    if col in indices or normalize_text(row.get(col)) is not None:
                        fila[col] = row.get(col)
                if usa_tech:
                    fila["TECHNOLOGY"] = f"{prefijo}_{mapeo['rename_tech'].get(tech, tech)}"
                if usa_fuel:
                    fila["FUEL"] = f"{prefijo}_{mapeo['rename_fuel'].get(fuel, fuel)}"

                for col, anio in zip(anios_out, anios_int):
                    val = safe_float(row[col])
                    if val is None:
                        continue
                    if (copia_valor
                            or es_centinela(val, cfg["valor_centinela"], cfg["centinela_patron_9s"])
                            or (es_condicional and val == 0)):
                        fila[col] = val   # intensivo, "sin límite", o límite 0: mismo valor
                    else:
                        if pct_combo is None:
                            continue      # condicional con valor real sin participación
                        p = pct_combo.get((prefijo, anio))
                        if p is None or pd.isna(p):
                            continue      # región sin participación ese año
                        fila[col] = val * float(p)
                filas_out.append(fila)
            n_ok += 1

        if filas_out:
            df_sand = pd.DataFrame(filas_out, columns=columnas_salida)
            for col in columnas_salida:   # recalcular la indicadora del SAND base
                if str(col).startswith("Tiene datos"):
                    vals = df_sand[anios_salida].apply(pd.to_numeric, errors="coerce")
                    df_sand[col] = (vals.notna() & (vals != 0)).any(axis=1).astype(int)
            sands[parametro] = df_sand
        else:
            log.append(_log_entry("ADVERTENCIA", parametro, None, None,
                                  "Regionalizado sin filas: todos los combos omitidos"))

        log.append(_log_entry("OK", parametro, None, None,
                              f"{n_ok} combos regionalizados ({len(filas_out)} filas SAND), "
                              f"{n_omitidos} omitidos, {n_creados} creados por decisión"))
        resumen.append({"Parametro": parametro, "Filas_SAND": len(filas_out), "Combos_OK": n_ok,
                        "Combos_Omitidos": n_omitidos, "Combos_Creados": n_creados})
        logger.info("%s: %s combos, %s filas, %s omitidos, %s creados",
                    parametro, n_ok, len(filas_out), n_omitidos, n_creados)

    return {"sands": sands, "log": pd.DataFrame(log),
            "resumen": pd.DataFrame(resumen),
            "participaciones_invalidas": participaciones_invalidas}

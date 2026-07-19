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
from utils import REGIONES, TIME_INDEP_COL, es_centinela, normalize_text, safe_float

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

    # Regiones en formato ancho: despivotar columnas de región -> Region/Participacion
    cols_region = [c for c in df.columns if c in _NOMBRES_REGION or c in REGIONES]
    if cols_region and "Region" not in df.columns:
        id_vars = [c for c in df.columns if c not in cols_region]
        df = df.melt(id_vars=id_vars, value_vars=cols_region,
                     var_name="Region", value_name="Participacion")
        logger.info("%s: formato ancho en regiones detectado (%s columnas de región)",
                    path.name, len(cols_region))

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
        return salida.dropna(subset=["Participacion"])
    # Solo constantes: difundir a todos los años solicitados
    salida = pd.concat(
        [constantes.assign(Año=a)[["Region", "Año", "Participacion"]] for a in anios],
        ignore_index=True,
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

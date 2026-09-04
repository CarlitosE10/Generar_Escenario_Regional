"""Construye la tabla maestra de mapeo entre el escenario NACIONAL y el REGIONAL.

Fuente de verdad autocontenida del pipeline de regionalización: se genera SOLO a
partir de los CSV de los escenarios (sin graphml, sin diccionario.xlsx, sin archivo
guía), de modo que sea reproducible y el equipo pueda re-ejecutarla cada vez que
actualice los inputs.

Insumos (CSV_Nacional/ y CSV_Regional/):
    TECHNOLOGY.csv, FUEL.csv, InputActivityRatio.csv, OutputActivityRatio.csv

Lógica:
 1. A cada código regional se le quita el prefijo de región (XX_) -> (región, base).
 2. La base se cruza contra el universo nacional (TECHNOLOGY/FUEL).
    - Tecnologías: por identidad de código base.
    - Fuels: por identidad y, para los que no casan, por inferencia de
      RENOMBRAMIENTO usando evidencia de conexión (Jaccard de las tecnologías
      conectadas vía IAR+OAR), anclada por RENOMBRES_FUEL_MANUALES. Sin evidencia
      no se inventa el renombramiento (un split 1->N como OIL no es expresable).
 3. Clasificación de cada entrada:
    - "mapeo_directo": el código regional corresponde a un código nacional
      (por identidad o por renombramiento inferido).
    - "sin_match": código regional huérfano (su base no existe en el nacional).
    - "nacional_sin_regionalizar": código nacional que no aparece en NINGUNA región.
 4. Para las tecnologías se resuelven además las conexiones IAR/OAR (fuels de
    entrada/salida, en códigos regionales y su equivalente nacional) para poder
    regionalizar esos parámetros correctamente.

Salidas (Insumos/Mapeo/, conviven con los xlsx del notebook 00):
    mapeo_maestro.csv    - tabla maestra, una fila por código (fuente de verdad).
    mapeo_pendientes.csv - solo sin_match + nacional_sin_regionalizar, para revisión.
    mapeo_alertas.csv    - casos especiales que el equipo debe refinar
                           (renombramientos de baja confianza, splits 1->N como
                           OIL, colisiones, códigos sin prefijo, topología).

Ejecutar desde la raíz del proyecto:  python scripts/mapeo_maestro.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from utils import is_trade_technology, split_regional_name  # noqa: E402

DIR_NACIONAL = RAIZ / "CSV_Nacional"
DIR_REGIONAL = RAIZ / "CSV_Regional"
DIR_SALIDA = RAIZ / "Insumos" / "Mapeo"

# Renombramientos FUEL nacional -> base regional fijados a mano. Mandan sobre la
# inferencia automática. La familia ELC es decisión UPME (2026-07-21): diccionario
# .xlsx la contradecía, pero el criterio por Jaccard coincide con esta decisión, así
# que quedan como anclaje explícito y documentado. Edítese aquí para añadir anclas.
RENOMBRES_FUEL_MANUALES = {
    "ELC": "ELC001",
    "ELC002": "ELC003",
    "ELC003": "ELCEV001",
    "ELC004": "ELCEV002",
}

# Por debajo de este solape (Jaccard) un renombramiento inferido se marca como de
# baja confianza en las alertas para que el equipo lo revise.
UMBRAL_JACCARD_ALERTA = 0.30


# --- Carga de insumos --------------------------------------------------------
def cargar_set(path: Path) -> set[str]:
    return set(pd.read_csv(path)["VALUE"].astype(str))


def cargar_pares(path: Path) -> dict[str, set[str]]:
    """IAR/OAR csv de otoole -> {tech: {fuels}}. Descarta las filas con VALUE=0
    (relleno sin conexión real: el nacional trae ~10.5k aristas OAR de ese tipo)."""
    df = pd.read_csv(path, usecols=["TECHNOLOGY", "FUEL", "VALUE"])
    df = df[df["VALUE"] != 0][["TECHNOLOGY", "FUEL"]].drop_duplicates()
    pares: dict[str, set[str]] = defaultdict(set)
    for tech, fuel in df.itertuples(index=False):
        pares[str(tech)].add(str(fuel))
    return dict(pares)


def existencia_por_region(codigos: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    """{base: {regiones donde existe}} y el conjunto de códigos sin prefijo regional."""
    existencia: dict[str, set[str]] = defaultdict(set)
    sin_prefijo: set[str] = set()
    for cod in codigos:
        region, base = split_regional_name(cod)
        if region:
            existencia[base].add(region)
        else:
            sin_prefijo.add(cod)
    return dict(existencia), sin_prefijo


def fuels_a_techs(*pares_por_tech: dict[str, set[str]]) -> dict[str, set[str]]:
    """{tech: {fuels}} -> {fuel base: {tech base}}, ambos sin prefijo de región."""
    out: dict[str, set[str]] = defaultdict(set)
    for pares in pares_por_tech:
        for tech, fuels in pares.items():
            _, base_tech = split_regional_name(tech)
            for fuel in fuels:
                _, base_fuel = split_regional_name(fuel)
                out[base_fuel].add(base_tech)
    return dict(out)


# --- Inferencia de correspondencia de FUEL -----------------------------------
def inferir_correspondencia_fuel(fuels_nac, exist_fuel, techs_de_fuel_nac, techs_de_fuel_reg):
    """Resuelve la correspondencia FUEL nacional -> base regional 1 a 1.

    Devuelve:
      mapa      : {fuel_nacional: base_regional}   (incluye identidades)
      fuente    : {fuel_nacional: 'manual'|'identidad'|'jaccard'}
      score     : {fuel_nacional: Jaccard del par elegido}
      candidatos: {fuel_nacional: {bases regionales con Jaccard>0}}
      sin_corr  : [fuels nacionales sin base regional]
    """
    bases_reg = set(exist_fuel)

    def jaccard(nac: str, reg: str) -> float:
        a = techs_de_fuel_nac.get(nac, set())
        b = techs_de_fuel_reg.get(reg, set())
        return len(a & b) / len(a | b) if (a or b) else 0.0

    candidatos: dict[str, set[str]] = defaultdict(set)
    for nac in fuels_nac:
        if nac in bases_reg:
            candidatos[nac].add(nac)                 # identidad
        for reg in bases_reg:
            if jaccard(nac, reg) > 0:
                candidatos[nac].add(reg)             # posible renombramiento

    # 1) Anclas manuales primero: sacan a ambos lados del reparto.
    mapa = {n: r for n, r in RENOMBRES_FUEL_MANUALES.items() if n in fuels_nac}
    fuente = {n: "manual" for n in mapa}
    ocupadas = set(mapa.values())

    # 2) Reparto greedy. Orden: (1) que la base exista con prefijo de región
    #    -mandar un fuel a una base solo-global equivale a no regionalizarlo-,
    #    (2) mayor solape, (3) alfabético (determinista).
    puntuadas = sorted(
        ((1 if exist_fuel.get(reg) else 0, jaccard(nac, reg), nac, reg)
         for nac, regs in candidatos.items() for reg in regs),
        key=lambda t: (-t[0], -t[1], t[2], t[3]))

    for _, sc, nac, reg in puntuadas:
        if nac in mapa or reg in ocupadas:
            continue
        if sc == 0 and nac != reg:
            continue                                  # sin evidencia: no se inventa
        mapa[nac] = reg
        ocupadas.add(reg)
        fuente[nac] = "identidad" if nac == reg else "jaccard"

    score = {n: jaccard(n, r) for n, r in mapa.items()}
    sin_corr = sorted(fuels_nac - set(mapa))
    return mapa, fuente, score, dict(candidatos), sin_corr


# --- Construcción de la tabla maestra ----------------------------------------
def _nota_tech_sin_match(base: str) -> str:
    if base.startswith("BACKSTOP"):
        return "BACKSTOP (tecnología de respaldo, nueva por diseño)"
    if is_trade_technology(base):
        return "TRN (comercio inter-regional, nueva por diseño)"
    return "revisar: tecnología regional sin equivalente nacional"


def main() -> None:
    # 1. Insumos
    techs_nac = cargar_set(DIR_NACIONAL / "TECHNOLOGY.csv")
    fuels_nac = cargar_set(DIR_NACIONAL / "FUEL.csv")
    techs_reg = cargar_set(DIR_REGIONAL / "TECHNOLOGY.csv")
    fuels_reg = cargar_set(DIR_REGIONAL / "FUEL.csv")

    iar_nac = cargar_pares(DIR_NACIONAL / "InputActivityRatio.csv")
    oar_nac = cargar_pares(DIR_NACIONAL / "OutputActivityRatio.csv")
    iar_reg = cargar_pares(DIR_REGIONAL / "InputActivityRatio.csv")
    oar_reg = cargar_pares(DIR_REGIONAL / "OutputActivityRatio.csv")

    exist_tech, techs_reg_sin_pref = existencia_por_region(techs_reg)
    exist_fuel, fuels_reg_sin_pref = existencia_por_region(fuels_reg)

    tf_nac = fuels_a_techs(iar_nac, oar_nac)
    tf_reg = fuels_a_techs(iar_reg, oar_reg)

    # 2. Correspondencia de FUEL (identidad + renombramientos por evidencia)
    mapa_fuel, fuente_fuel, score_fuel, candidatos_fuel, sin_corr_fuel = \
        inferir_correspondencia_fuel(fuels_nac, exist_fuel, tf_nac, tf_reg)
    renombre_fuel_inv = {reg: nac for nac, reg in mapa_fuel.items()}   # base_reg -> nacional

    def fuel_regional_a_nacional(cod_regional: str) -> str | None:
        """Código FUEL regional (AN_DSL002 o DSL001) -> FUEL nacional o None."""
        _, base = split_regional_name(cod_regional)
        if base in renombre_fuel_inv:
            return renombre_fuel_inv[base]
        return base if base in fuels_nac else None

    def conexiones_tech(cod_tech_reg: str, cod_tech_nac: str | None):
        """Fuels regionales de entrada/salida de una tecnología y su mapeo nacional.
        Devuelve (iar_reg, iar_nac, oar_reg, oar_nac, pares_sin_equivalente)."""
        sin_equiv = []
        cols = {}
        for etiqueta, pares_r, pares_n in [("iar", iar_reg, iar_nac), ("oar", oar_reg, oar_nac)]:
            f_reg = sorted(pares_r.get(cod_tech_reg, set()))
            f_nac = []
            for fr in f_reg:
                fn = fuel_regional_a_nacional(fr)
                if fn:
                    f_nac.append(fn)
                    # ¿existe el par equivalente en el nacional?
                    if cod_tech_nac and fn not in pares_n.get(cod_tech_nac, set()):
                        sin_equiv.append(f"{etiqueta}:{fr}->{fn}")
                else:
                    sin_equiv.append(f"{etiqueta}:{fr}->(sin fuel nacional)")
            cols[f"{etiqueta}_reg"] = ";".join(f_reg)
            cols[f"{etiqueta}_nac"] = ";".join(sorted(set(f_nac)))
        return cols, sin_equiv

    filas: list[dict] = []
    alertas: list[dict] = []

    def alerta(categoria: str, detalle: str) -> None:
        alertas.append({"categoria": categoria, "detalle": detalle})

    # 3a. TECNOLOGÍAS dirigidas por el nacional (mapeo_directo / nacional_sin_regionalizar)
    for tech in sorted(techs_nac):
        regiones = sorted(exist_tech.get(tech, set()))
        if not regiones:
            if tech in techs_reg_sin_pref:   # existe global (sin prefijo) en el regional
                cols, sin_equiv = conexiones_tech(tech, tech)
                filas.append({"codigo_nacional": tech, "codigo_regional": tech, "region": "",
                              "prefijo": "", "tipo": "technology", "clasificacion": "mapeo_directo",
                              "renombrado": False, "base_regional": tech,
                              "fuente_correspondencia": "identidad", "score_jaccard": "",
                              "nota": "tecnología global (sin prefijo regional)", **cols})
            else:
                filas.append({"codigo_nacional": tech, "codigo_regional": "", "region": "",
                              "prefijo": "", "tipo": "technology",
                              "clasificacion": "nacional_sin_regionalizar",
                              "renombrado": False, "base_regional": "",
                              "fuente_correspondencia": "", "score_jaccard": "",
                              "nota": "tecnología nacional sin presencia regional"})
            continue
        for region in regiones:
            cod_reg = f"{region}_{tech}"
            cols, sin_equiv = conexiones_tech(cod_reg, tech)
            filas.append({"codigo_nacional": tech, "codigo_regional": cod_reg,
                          "region": region, "prefijo": region, "tipo": "technology",
                          "clasificacion": "mapeo_directo", "renombrado": False,
                          "base_regional": tech, "fuente_correspondencia": "identidad",
                          "score_jaccard": "", "nota": "", **cols})
            if sin_equiv:
                alerta("conexion_sin_equivalente_nacional",
                       f"{cod_reg}: {'; '.join(sin_equiv[:8])}")

    # 3b. TECNOLOGÍAS regionales huérfanas (sin_match)
    for base in sorted(b for b in exist_tech if b not in techs_nac):
        for region in sorted(exist_tech[base]):
            cod_reg = f"{region}_{base}"
            cols, _ = conexiones_tech(cod_reg, None)
            filas.append({"codigo_nacional": "", "codigo_regional": cod_reg,
                          "region": region, "prefijo": region, "tipo": "technology",
                          "clasificacion": "sin_match", "renombrado": False,
                          "base_regional": base, "fuente_correspondencia": "",
                          "score_jaccard": "", "nota": _nota_tech_sin_match(base), **cols})
    # Tecnologías regionales sin prefijo y SIN equivalente nacional (TRN* de comercio,
    # etc.). Las que sí son nacionales ya se emitieron como "global" en 3a.
    for cod in sorted(c for c in techs_reg_sin_pref if c not in techs_nac):
        cols, _ = conexiones_tech(cod, None)
        filas.append({"codigo_nacional": "", "codigo_regional": cod,
                      "region": "", "prefijo": "", "tipo": "technology",
                      "clasificacion": "sin_match", "renombrado": False, "base_regional": cod,
                      "fuente_correspondencia": "", "score_jaccard": "",
                      "nota": _nota_tech_sin_match(cod), **cols})
        alerta("codigo_regional_sin_prefijo", f"technology {cod}")

    # 4a. FUELS dirigidos por el nacional (mapeo_directo / nacional_sin_regionalizar)
    for fuel in sorted(fuels_nac):
        base_reg = mapa_fuel.get(fuel)
        regiones = sorted(exist_fuel.get(base_reg, set())) if base_reg else []
        if not regiones:
            if fuel in fuels_reg_sin_pref:   # existe global (sin prefijo) en el regional
                filas.append({"codigo_nacional": fuel, "codigo_regional": fuel, "region": "",
                              "prefijo": "", "tipo": "fuel", "clasificacion": "mapeo_directo",
                              "renombrado": False, "base_regional": fuel,
                              "fuente_correspondencia": "identidad", "score_jaccard": "",
                              "nota": "commodity global (sin prefijo regional)"})
            else:
                filas.append({"codigo_nacional": fuel, "codigo_regional": "", "region": "",
                              "prefijo": "", "tipo": "fuel",
                              "clasificacion": "nacional_sin_regionalizar",
                              "renombrado": False, "base_regional": base_reg or "",
                              "fuente_correspondencia": fuente_fuel.get(fuel, ""),
                              "score_jaccard": "",
                              "nota": "fuel nacional sin presencia regional"})
            continue
        renombrado = base_reg != fuel
        for region in regiones:
            filas.append({"codigo_nacional": fuel, "codigo_regional": f"{region}_{base_reg}",
                          "region": region, "prefijo": region, "tipo": "fuel",
                          "clasificacion": "mapeo_directo", "renombrado": renombrado,
                          "base_regional": base_reg,
                          "fuente_correspondencia": fuente_fuel.get(fuel, ""),
                          "score_jaccard": round(score_fuel.get(fuel, 0.0), 3),
                          "nota": f"renombrado desde {fuel}" if renombrado else ""})

    # 4b. FUELS regionales huérfanos (base no reclamada por ningún nacional)
    reclamadas = set(mapa_fuel.values())
    for base in sorted(b for b in exist_fuel if b not in reclamadas):
        for region in sorted(exist_fuel[base]):
            filas.append({"codigo_nacional": "", "codigo_regional": f"{region}_{base}",
                          "region": region, "prefijo": region, "tipo": "fuel",
                          "clasificacion": "sin_match", "renombrado": False,
                          "base_regional": base, "fuente_correspondencia": "",
                          "score_jaccard": "",
                          "nota": "revisar: fuel regional sin equivalente nacional"})
    # Fuels regionales sin prefijo y SIN equivalente nacional (BDL001, GSL001, ...).
    # Los que sí son nacionales (URN, FOL) ya se emitieron como "global" en 4a.
    for cod in sorted(c for c in fuels_reg_sin_pref if c not in fuels_nac):
        filas.append({"codigo_nacional": "", "codigo_regional": cod,
                      "region": "", "prefijo": "", "tipo": "fuel",
                      "clasificacion": "sin_match", "renombrado": False, "base_regional": cod,
                      "fuente_correspondencia": "", "score_jaccard": "",
                      "nota": "revisar: fuel regional sin prefijo ni equivalente nacional"})
        alerta("codigo_regional_sin_prefijo", f"fuel {cod} (sin equivalente nacional)")

    # --- Alertas de renombramientos y splits ---
    for nac, reg in sorted((n, r) for n, r in mapa_fuel.items() if n != r):
        sc = score_fuel.get(nac, 0.0)
        if fuente_fuel.get(nac) == "jaccard" and sc < UMBRAL_JACCARD_ALERTA:
            otros = sorted(candidatos_fuel.get(nac, set()) - {reg})
            alerta("renombramiento_baja_confianza",
                   f"{nac} -> {reg} (Jaccard {sc:.2f} < {UMBRAL_JACCARD_ALERTA}"
                   + (f"; otros candidatos: {';'.join(otros)}" if otros else "") + ")")

    for nac in sin_corr_fuel:
        cand = sorted(candidatos_fuel.get(nac, set()))
        if cand:
            # Sus candidatos ya los tomó otro fuel (mejor solape): típicamente un
            # commodity global -renovables SOL/WND/GEO...- o una relación N:1 no
            # expresable como renombramiento 1 a 1. Se indica quién tomó cada base.
            detalle = "; ".join(
                f"{c} (lo tomó {renombre_fuel_inv[c]})" if c in renombre_fuel_inv else c
                for c in cand)
            alerta("fuel_candidatos_ya_asignados",
                   f"{nac}: candidatos {detalle}. Revisar si es commodity global o split N:1")
        else:
            alerta("fuel_sin_correspondencia",
                   f"{nac}: sin evidencia de conexión regional (posible split 1->N tipo OIL, "
                   "o commodity solo-nacional); revisar a mano")

    for base_reg, nac in renombre_fuel_inv.items():
        if base_reg in fuels_nac and base_reg not in mapa_fuel:
            alerta("colision_de_cadena",
                   f"{nac} -> {base_reg}: '{base_reg}' es también un fuel nacional que "
                   "quedó sin base regional propia (revisar)")

    # 5. Escritura
    columnas = ["codigo_nacional", "codigo_regional", "region", "prefijo", "tipo",
                "clasificacion", "renombrado", "base_regional", "fuente_correspondencia",
                "score_jaccard", "nota", "iar_reg", "iar_nac", "oar_reg", "oar_nac"]
    maestro = pd.DataFrame(filas).reindex(columns=columnas)
    orden_tipo = {"technology": 0, "fuel": 1}
    orden_clas = {"mapeo_directo": 0, "sin_match": 1, "nacional_sin_regionalizar": 2}
    maestro = maestro.sort_values(
        by=["tipo", "clasificacion", "codigo_nacional", "codigo_regional"],
        key=lambda c: c.map(orden_tipo).fillna(c.map(orden_clas)) if c.name in ("tipo", "clasificacion") else c
    ).reset_index(drop=True)

    pendientes = maestro[maestro["clasificacion"].isin(["sin_match", "nacional_sin_regionalizar"])]
    pendientes = pendientes[["tipo", "clasificacion", "codigo_nacional", "codigo_regional",
                             "region", "base_regional", "nota"]].sort_values(
        by=["tipo", "clasificacion", "base_regional", "codigo_regional"]).reset_index(drop=True)

    df_alertas = pd.DataFrame(alertas, columns=["categoria", "detalle"]).sort_values(
        by=["categoria", "detalle"]).reset_index(drop=True)

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    maestro.to_csv(DIR_SALIDA / "mapeo_maestro.csv", index=False, encoding="utf-8-sig")
    pendientes.to_csv(DIR_SALIDA / "mapeo_pendientes.csv", index=False, encoding="utf-8-sig")
    df_alertas.to_csv(DIR_SALIDA / "mapeo_alertas.csv", index=False, encoding="utf-8-sig")

    # 6. Resumen en consola
    def cuenta(tipo, clas):
        return len(maestro[(maestro["tipo"] == tipo) & (maestro["clasificacion"] == clas)])

    print(f"Raíz: {RAIZ}")
    print(f"Nacional: {len(techs_nac)} tecnologías, {len(fuels_nac)} fuels | "
          f"Regional: {len(techs_reg)} tecnologías, {len(fuels_reg)} fuels")
    n_ren = sum(1 for n, r in mapa_fuel.items() if n != r)
    print(f"FUEL: {len(mapa_fuel)} correspondencias ({n_ren} renombramientos, "
          f"{len(sin_corr_fuel)} nacionales sin base regional)")
    print("Tabla maestra por clasificación:")
    for tipo in ("technology", "fuel"):
        print(f"  {tipo:11} directo={cuenta(tipo, 'mapeo_directo'):5} "
              f"sin_match={cuenta(tipo, 'sin_match'):5} "
              f"nac_sin_regionalizar={cuenta(tipo, 'nacional_sin_regionalizar'):3}")
    print(f"Salidas en {DIR_SALIDA.relative_to(RAIZ)}/:")
    print(f"  mapeo_maestro.csv    ({len(maestro)} filas)")
    print(f"  mapeo_pendientes.csv ({len(pendientes)} filas)")
    print(f"  mapeo_alertas.csv    ({len(df_alertas)} alertas en "
          f"{df_alertas['categoria'].nunique()} categorías)")
    if not df_alertas.empty:
        print("Alertas por categoría:")
        for cat, n in df_alertas["categoria"].value_counts().items():
            print(f"    {cat}: {n}")


if __name__ == "__main__":
    main()

"""Genera config/participaciones_industriales.xlsx (formato nuevo) a partir del
archivo legado Insumos/Participacion_Regional_Industrial.xlsx.

Formato de salida (el que consumen los módulos src/regionalizador.py):
    Parámetro | TECHNOLOGY | FUEL | Región | 2022 | ... | 2055

- AccumulatedAnnualDemand: participación por FUEL industrial (hoja
  Participacion_Fuel del legado, textos LEAP -> códigos SAND vía mapeo).
- TotalTechnologyAnnualActivityLowerLimit: participación por tecnología
  industrial (hoja Participacion_Plano, rutas LEAP 'Uso\\fuel\\eficiencia'),
  agregada por familia tecnológica: se suma la demanda (Valor) de todas las
  eficiencias de 'Uso\\fuel' y con ese total se sacan los porcentajes, de modo
  que las variantes _LOW/_MID de una misma familia reciben idéntica
  distribución regional.

Los años del legado llegan hasta 2054; 2055 hereda la participación de 2054.
Ejecutar desde la raíz del proyecto:  python scripts/generar_participaciones_industriales.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from sand_io import cargar_sand, columnas_anio  # noqa: E402

LEGADO = RAIZ / "Insumos" / "Participacion_Regional_Industrial.xlsx"
SALIDA = RAIZ / "config" / "participaciones_industriales.xlsx"
SAND_NACIONAL = RAIZ / "SAND_Nacional_base" / "01-04-2026 SAND BASE v10.xlsx"

REGIONES = ["Antioquia", "Caribe", "Este", "Insular", "Nordeste", "Oriente", "Suroccidente"]

# Código FUEL nacional -> texto de la hoja Participacion_Fuel (sin tildes,
# tal como viene en el legado). INDOTH_* queda excluido a propósito.
MAPEO_FUEL = {
    "INDCLIM": "Aire acondicionado",
    "INDDHT": "Calor directo",
    "INDIHT": "Calor indirecto",
    "INDMPW": "Fuerza motriz",
    "INDILU": "Iluminacion",
    "INDREF": "Refrigeracion",
}

# Código TECHNOLOGY nacional -> ruta LEAP 'Uso\fuel\eficiencia' (hoja Participacion_Plano)
MAPEO_TECNOLOGIA = {
    "DEMINDBAGBOI_LOW": "Calor indirecto\\Bagazo\\Eficiencia_existente",
    "DEMINDBAGBOI_MID": "Calor indirecto\\Bagazo\\Mejor eficiencia_Colombia",
    "DEMINDBAGFUR_LOW": "Calor directo\\Bagazo\\Eficiencia_existente",
    "DEMINDBAGFUR_MID": "Calor directo\\Bagazo\\Mejor eficiencia_Colombia",
    "DEMINDCOABOI_LOW": "Calor indirecto\\Carbón mineral\\Eficiencia_existente",
    "DEMINDCOABOI_MID": "Calor indirecto\\Carbón mineral\\Mejor eficiencia_Colombia",
    "DEMINDCOAFUR_LOW": "Calor directo\\Carbón mineral\\Eficiencia_existente",
    "DEMINDCOAFUR_MID": "Calor directo\\Carbón mineral\\Mejor eficiencia_Colombia",
    "DEMINDCOAOTH_LOW": "Otros\\Carbón mineral\\Eficiencia_existente",
    "DEMINDDSLBOI_LOW": "Calor indirecto\\Diesel\\Eficiencia_existente",
    "DEMINDDSLBOI_MID": "Calor indirecto\\Diesel\\Mejor eficiencia_Colombia",
    "DEMINDDSLFUR_LOW": "Calor directo\\Diesel\\Eficiencia_existente",
    "DEMINDDSLFUR_MID": "Calor directo\\Diesel\\Mejor eficiencia_Colombia",
    "DEMINDELCAIR_LOW": "Aire acondicionado\\Electricidad\\Eficiencia_existente",
    "DEMINDELCAIR_MID": "Aire acondicionado\\Electricidad\\Mejor eficiencia_Colombia",
    "DEMINDELCBOI_LOW": "Calor indirecto\\Electricidad\\Eficiencia_existente",
    "DEMINDELCBOI_MID": "Calor indirecto\\Electricidad\\Mejor eficiencia_Colombia",
    "DEMINDELCFUR_LOW": "Calor directo\\Electricidad\\Eficiencia_existente",
    "DEMINDELCFUR_MID": "Calor directo\\Electricidad\\Mejor eficiencia_Colombia",
    "DEMINDELCILU_LOW": "Iluminacion\\Electricidad\\Eficiencia_existente",
    "DEMINDELCILU_MID": "Iluminacion\\Electricidad\\Mejor eficiencia_Colombia",
    "DEMINDELCMPW_LOW": "Fuerza motriz\\Electricidad\\Eficiencia_existente",
    "DEMINDELCMPW_MID": "Fuerza motriz\\Electricidad\\Mejor eficiencia_Colombia",
    "DEMINDELCOTH_LOW": "Otros\\Electricidad\\Eficiencia_existente",
    "DEMINDELCOTH_MID": "Otros\\Electricidad\\Mejor eficiencia_Colombia",
    "DEMINDELCREF_LOW": "Refrigeracion\\Electricidad\\Eficiencia_existente",
    "DEMINDELCREF_MID": "Refrigeracion\\Electricidad\\Mejor eficiencia_Colombia",
    "DEMINDLPGBOI_LOW": "Calor indirecto\\GLP\\Eficiencia_existente",
    "DEMINDLPGBOI_MID": "Calor indirecto\\GLP\\Mejor eficiencia_Colombia",
    "DEMINDLPGFUR_LOW": "Calor directo\\GLP\\Eficiencia_existente",
    "DEMINDLPGFUR_MID": "Calor directo\\GLP\\Mejor eficiencia_Colombia",
    "DEMINDNGSBOI_LOW": "Calor indirecto\\Gas Natural\\Eficiencia_existente",
    "DEMINDNGSBOI_MID": "Calor indirecto\\Gas Natural\\Mejor eficiencia_Colombia",
    "DEMINDNGSFUR_LOW": "Calor directo\\Gas Natural\\Eficiencia_existente",
    "DEMINDNGSFUR_MID": "Calor directo\\Gas Natural\\Mejor eficiencia_Colombia",
    "DEMINDWASBOI_LOW": "Calor indirecto\\Resíduos\\Eficiencia_existente",
    "DEMINDWASBOI_MID": "Calor indirecto\\Resíduos\\Mejor eficiencia_Colombia",
    "DEMINDWASFUR_LOW": "Calor directo\\Resíduos\\Eficiencia_existente",
    "DEMINDWASFUR_MID": "Calor directo\\Resíduos\\Mejor eficiencia_Colombia",
}


EFICIENCIAS = ("Eficiencia_existente", "Mejor eficiencia_Colombia", "Mejor eficiencia_internacional")


def _familia(ruta: str) -> str:
    """Ruta LEAP sin el segmento final de eficiencia ('Uso\\fuel')."""
    partes = str(ruta).strip().split("\\")
    return "\\".join(partes[:-1]) if partes[-1] in EFICIENCIAS else "\\".join(partes)


def _completar_anios(df: pd.DataFrame, anios: list[int]) -> pd.DataFrame:
    """Reindexa a todos los años del SAND, heredando el último disponible (ffill)."""
    df = df.reindex(columns=sorted(set(df.columns) | set(anios)))
    return df.ffill(axis=1)[anios]


def main() -> None:
    anios = [int(a) for a in columnas_anio(cargar_sand(SAND_NACIONAL))]
    filas = []

    # --- 1. AccumulatedAnnualDemand por FUEL (hoja pivote Participacion_Fuel) ---
    fuel_raw = pd.read_excel(LEGADO, sheet_name="Participacion_Fuel")
    texto_a_codigo = {v.lower().strip(): k for k, v in MAPEO_FUEL.items()}
    fuel_raw["_codigo"] = fuel_raw["Fuel"].astype(str).str.lower().str.strip().map(texto_a_codigo)
    fuel_raw = fuel_raw.dropna(subset=["_codigo"])
    for codigo, grupo in fuel_raw.groupby("_codigo"):
        serie = grupo.set_index("Anio")
        for region in REGIONES:
            fila = {"Parámetro": "AccumulatedAnnualDemand", "TECHNOLOGY": None,
                    "FUEL": codigo, "Región": region}
            valores = _completar_anios(serie[[region]].T, anios).iloc[0]
            fila.update({a: valores[a] for a in anios})
            filas.append(fila)

    # --- 2. LowerLimit por TECHNOLOGY (hoja plana Participacion_Plano) ---
    # La participación se agrega por familia tecnológica ('Uso\fuel'): se suma
    # la demanda (Valor) de todas las eficiencias de la familia y con ese total
    # se sacan los porcentajes regionales, de modo que las variantes _LOW/_MID
    # de una misma familia reciben idéntica distribución.
    tec_raw = pd.read_excel(LEGADO, sheet_name="Participacion_Plano")
    tec_raw["_familia"] = tec_raw["Tecnologia"].map(_familia).str.lower().str.strip()
    agregado = tec_raw.groupby(["_familia", "Region", "Anio"], as_index=False)["Valor"].sum()
    total = agregado.groupby(["_familia", "Anio"])["Valor"].transform("sum")
    agregado["Participacion"] = (agregado["Valor"] / total).where(total != 0, 0.0)

    codigo_a_familia = {k: _familia(v).lower().strip() for k, v in MAPEO_TECNOLOGIA.items()}
    familias_legado = set(agregado["_familia"])
    sin_mapa = {f for f in codigo_a_familia.values() if f not in familias_legado}
    if sin_mapa:
        print(f"ADVERTENCIA: {len(sin_mapa)} familias del mapeo sin datos en el legado")

    pivote = agregado.pivot_table(index=["_familia", "Region"], columns="Anio",
                                  values="Participacion", aggfunc="first")
    pivote = _completar_anios(pivote, anios)
    for codigo, familia in codigo_a_familia.items():
        if familia not in familias_legado:
            continue
        for region in REGIONES:
            valores = pivote.loc[(familia, region)]
            fila = {"Parámetro": "TotalTechnologyAnnualActivityLowerLimit",
                    "TECHNOLOGY": codigo, "FUEL": None, "Región": region}
            fila.update({a: valores[a] for a in anios})
            filas.append(fila)

    salida = pd.DataFrame(filas, columns=["Parámetro", "TECHNOLOGY", "FUEL", "Región"] + anios)
    SALIDA.parent.mkdir(exist_ok=True)
    salida.to_excel(SALIDA, index=False)
    print(f"Escrito {SALIDA} ({len(salida)} filas, "
          f"{salida['Parámetro'].nunique()} parámetros, años {anios[0]}–{anios[-1]})")


if __name__ == "__main__":
    main()

"""Smoke tests de los módulos src/ contra los archivos reales del proyecto.

Ejecutar desde la raíz:  python tests/test_smoke.py
(no requiere pytest; termina con código != 0 si algo falla)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import comparador
import regionalizador
import reporte
import sand_io
import sincronizador
import utils
import yaml_parser

utils.configurar_logging()
FALLOS: list[str] = []


def check(condicion: bool, mensaje: str) -> None:
    estado = "OK " if condicion else "FALLO"
    print(f"[{estado}] {mensaje}")
    if not condicion:
        FALLOS.append(mensaje)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="smoke_sand_"))

    # --- yaml_parser -----------------------------------------------------------
    cfg = yaml_parser.cargar_params_config(RAIZ / "config" / "params_config.yaml")
    paths = yaml_parser.cargar_paths_config(RAIZ / "config" / "paths_config.yaml", raiz=RAIZ)
    otoole = yaml_parser.cargar_config_otoole(paths["escenario_nacional"]["config_yaml"])
    # El config otoole es un template que sigue el patrón OSeMOSYS: puede traer
    # parámetros/sets de más (no usados) y crece con el tiempo (p. ej. el set
    # STORAGE). En vez de un conteo exacto y frágil, verificamos que cargue y
    # contenga lo que el flujo realmente consume.
    params_esenciales = {"AccumulatedAnnualDemand", "InputActivityRatio",
                         "TotalAnnualMaxCapacityInvestment",
                         "TotalTechnologyAnnualActivityLowerLimit"}
    sets_esenciales = {"TECHNOLOGY", "FUEL", "YEAR"}
    check(params_esenciales <= set(otoole["param"]) and sets_esenciales <= set(otoole["set"]),
          "yaml_parser: config otoole con los params/sets esenciales del flujo")
    check(len(cfg["prefijo_region"]) == 7, "yaml_parser: 7 regiones con prefijo")

    # --- utils ------------------------------------------------------------------
    check(utils.es_centinela(99999.0, cfg["valor_centinela"], True), "utils: 99999 es centinela")
    check(utils.es_centinela(9999, cfg["valor_centinela"], True), "utils: 9999 (patrón 9s) es centinela")
    check(not utils.es_centinela(12.5, cfg["valor_centinela"], True), "utils: 12.5 no es centinela")
    check(utils.split_regional_name("CA_TRABUS") == ("CA", "TRABUS"), "utils: split prefijo regional")
    check(utils.norm_indice(1.0) == "1", "utils: norm_indice 1.0 -> '1'")

    # --- sand_io ------------------------------------------------------------------
    df_nac = sand_io.cargar_sand(paths["escenario_nacional"]["sand"])
    df_reg = sand_io.cargar_sand(paths["escenario_regional"]["sand"])
    anios = sand_io.columnas_anio(df_nac)
    check(len(df_nac) > 60000 and len(df_reg) > 25000, "sand_io: SANDs cargados")
    check(anios[0] == "2022" and anios[-1] == "2055", "sand_io: años 2022–2055")

    # --- comparador: modo lista_parametros (aditivo + intensivo) ------------------
    res = comparador.comparar_escenarios(
        df_nac, df_reg, otoole["param"], cfg,
        modo="lista_parametros",
        parametros_filtro=["AccumulatedAnnualDemand", "CapitalCost"],
        fuels=["TRABUS", "TRAAVI"],
        tecnologias=["DEMINDNGSBOI"],
        modo_filtro="contiene",
    )
    comp = res["comparacion"]
    check(not comp.empty, "comparador: produce comparación")
    check((comp[comp["Parametro"] == "AccumulatedAnnualDemand"]["Region"] == "SUMA_REGIONES").all(),
          "comparador: aditivo compara contra SUMA_REGIONES")
    reg_int = set(comp[comp["Parametro"] == "CapitalCost"]["Region"])
    check(reg_int and reg_int.issubset(set(cfg["prefijo_region"].values()) | {"SIN_PREFIJO"}),
          "comparador: intensivo compara región por región")
    check(comp["Diferencia"].abs().is_monotonic_decreasing, "comparador: ordenado por |Diferencia| desc")
    check("Diferencia_Pct" in comp.columns, "comparador: columna Diferencia %")
    check(set(res["anomalias"].columns) == {"Parametro", "Tecnologia_Fuel", "Tipo_Anomalia", "Detalle"},
          "comparador: hoja de anomalías con columnas esperadas")
    check((res["resumen"].iloc[0]["Parametro"] == "== TOTAL =="), "comparador: resumen con fila total")

    # centinela: ninguna fila aditiva con valor nacional 9999/99999
    aad = comp[comp["Parametro"] == "AccumulatedAnnualDemand"]
    check(not aad["Valor_Nacional"].apply(lambda v: utils.es_centinela(v, cfg["valor_centinela"], True)).any(),
          "comparador: centinelas excluidos de la ruta aditiva")

    # anomalía global: código nacional ausente del regional en TODOS los
    # parámetros consultados (INDOTH_COA no está regionalizado)
    res_glob = comparador.comparar_escenarios(
        df_nac, df_reg, otoole["param"], cfg,
        modo="lista_parametros", parametros_filtro=["AccumulatedAnnualDemand"],
        fuels=["INDOTH_COA", "INDCLIM"], modo_filtro="exacto",
    )
    an_glob = res_glob["anomalias"]
    fila_glob = an_glob[an_glob["Tipo_Anomalia"] == "FUEL_NO_EN_REGIONAL"]
    check(len(fila_glob) == 1 and (fila_glob["Tecnologia_Fuel"] == "INDOTH_COA").all(),
          "comparador: anomalía global FUEL_NO_EN_REGIONAL (INDOTH_COA, no INDCLIM)")
    check((fila_glob["Parametro"] == "TODOS (consultados)").all(),
          "comparador: anomalía global marcada como 'TODOS (consultados)'")

    # las anomalías respetan los filtros de tecnología/fuel (bug corregido:
    # REGIONES_INCOMPLETAS se calculaba del lado regional sin filtrar)
    res_filtro = comparador.comparar_escenarios(
        df_nac, df_reg, otoole["param"], cfg,
        modo="lista_parametros",
        parametros_filtro=["AccumulatedAnnualDemand", "InputActivityRatio"],
        tecnologias=["RES"], fuels=["RES"], modo_filtro="contiene",
    )
    an_filtro = res_filtro["anomalias"]
    fuera = an_filtro[(an_filtro["Tecnologia_Fuel"] != "-")
                      & ~an_filtro["Tecnologia_Fuel"].astype(str).str.contains("RES")]
    check(fuera.empty, "comparador: las anomalías respetan el filtro TECHNOLOGY/FUEL")

    # parámetro sin YEAR (DiscountRate, indexado solo por REGION)
    res_ti = comparador.comparar_escenarios(
        df_nac, df_reg, otoole["param"], cfg,
        modo="parametro", parametros_filtro=["DiscountRate"],
    )
    check(len(res_ti["comparacion"]) >= 1 and res_ti["comparacion"]["Año"].isna().all(),
          "comparador: parámetro tiempo-independiente (DiscountRate) comparado sin Año")

    # --- comparador: inventario, análisis por sector y contextualización ----------
    regiones = list(cfg["prefijo_region"].values())
    mapeos = comparador.cargar_mapeos(RAIZ / "Insumos" / "Mapeo")
    inv = comparador.inventario_tech_fuel(
        df_nac, df_reg, regiones,
        diccionario_tech=mapeos["diccionario_tech"], diccionario_fuel=mapeos["diccionario_fuel"])
    # Los conteos deben cuadrar con los códigos únicos del nacional:
    # Solo Nacional + En ambos == universo TECHNOLOGY del SAND nacional
    tech = inv["tech"]
    n_solo_nac = (tech["Categoria"] == "Solo Nacional").sum()
    n_ambos = (tech["Categoria"] == "En ambos").sum()
    n_nac = df_nac["TECHNOLOGY"].dropna().apply(utils.normalize_text).nunique()
    check(n_solo_nac + n_ambos == n_nac,
          f"comparador: inventario TECHNOLOGY cuadra con el nacional ({n_solo_nac}+{n_ambos}=={n_nac})")
    check({"Estado_Mapeo", "Nombre_Equivalente", "Grupo"} <= set(tech.columns),
          "comparador: inventario enriquecido con el diccionario de mapeo")
    if mapeos["diccionario_fuel"] is not None:
        check(not inv["renombramientos_fuel"].empty
              and (inv["renombramientos_fuel"]["Fuente"] == "diccionario_fuel").all(),
              "comparador: renombramientos de FUEL desde diccionario_fuel.xlsx")

    check(comparador.detectar_sectores(df_nac) == comparador.SECTORES,
          "comparador: los 6 sectores activos en el nacional")
    df_reg_prep = comparador.preparar_regional(df_reg, regiones)
    sec = comparador.analisis_por_sector(
        df_nac, df_reg, "TRA", otoole["param"], cfg,
        diccionario_tech=mapeos["diccionario_tech"], diccionario_fuel=mapeos["diccionario_fuel"],
        df_reg_prep=df_reg_prep)
    rv = sec["resumen_validacion"]
    check(not rv.empty and (rv["Filas_OK"] + rv["Filas_Discrepancia"] == rv["Filas"]).all(),
          "comparador: análisis por sector TRA con conteos OK+discrepancia consistentes")
    check(sec["modos_tra"] is not None and not sec["modos_tra"].empty,
          "comparador: desglose por modo de transporte (MODOS_TRA)")
    desg = sec["desglose"]
    cols_reg = [c for c in regiones if c in desg.columns]
    # Solo filas con dato regional (las solo-nacionales dejan Suma_Regional en NaN)
    con_reg = desg[desg["Suma_Regional"].notna()]
    diff_desg = (con_reg[cols_reg].sum(axis=1) - con_reg["Suma_Regional"]).abs().max()
    check(not con_reg.empty and diff_desg < 1e-9,
          "comparador: desglose por región suma igual a Suma_Regional")
    sec_filtrado = comparador.analisis_por_sector(
        df_nac, df_reg, "TRA", otoole["param"], cfg,
        parametros=["AccumulatedAnnualDemand"], df_reg_prep=df_reg_prep)
    check(set(sec_filtrado["resumen_validacion"]["Parametro"]) == {"AccumulatedAnnualDemand"},
          "comparador: análisis por sector respeta el filtro de parámetros")
    check(not comparador.tabla_por_sector({"TRA": sec}).empty,
          "comparador: tabla pivot sector × parámetro")

    if mapeos["mapeo"] is not None:
        ctx = comparador.contextualizar_anomalias(res_glob["anomalias"], mapeos["mapeo"])
        check({"TIENE_MAPEO", "REGIONES_ESPERADAS", "NOMBRE_REGIONAL", "ACCION_SUGERIDA"}
              <= set(ctx.columns)
              and set(ctx["ACCION_SUGERIDA"]) <= {"cambio_de_nombre", "no_existe_en_region",
                                                  "verificar_creacion", "sin_info"},
              "comparador: anomalías contextualizadas con acción sugerida")

    # --- regionalizador ------------------------------------------------------------
    # Archivo industrial canónico fijo: la config activa (paths_config.yaml) puede
    # apuntar a otro escenario, pero la equivalencia de abajo es contra las
    # salidas industriales de referencia en SANDs_Reducidos/
    df_pct = regionalizador.cargar_participaciones(RAIZ / "config" / "participaciones_industriales.xlsx")
    check({"Parametro", "TECHNOLOGY", "FUEL", "Region", "Año", "Participacion"} <= set(df_pct.columns),
          "regionalizador: participaciones en formato largo estándar")
    malas = regionalizador.validar_participaciones(df_pct, cfg["tolerancia_participacion"])
    check(malas.empty, "regionalizador: participaciones suman 1.0")

    # Filtros explícitos derivados del archivo de participaciones: un filtro
    # amplio ('contiene IND') arrastraría códigos sin participación -> error
    fuels_aad = sorted(df_pct[df_pct["Parametro"] == "AccumulatedAnnualDemand"]["FUEL"].dropna().unique())
    techs_ll = sorted(df_pct[df_pct["Parametro"] == "TotalTechnologyAnnualActivityLowerLimit"]
                      ["TECHNOLOGY"].dropna().unique())
    res_aad = regionalizador.regionalizar(
        df_nac, df_reg, df_pct,
        parametros=["AccumulatedAnnualDemand"],
        params_otoole=otoole["param"], cfg=cfg,
        fuels_filtro=fuels_aad, modo_filtro="exacto",
    )
    res_ll = regionalizador.regionalizar(
        df_nac, df_reg, df_pct,
        parametros=["TotalTechnologyAnnualActivityLowerLimit"],
        params_otoole=otoole["param"], cfg=cfg,
        tecnologias_filtro=techs_ll, modo_filtro="exacto",
    )
    sands = {**res_aad["sands"], **res_ll["sands"]}
    log_total = pd.concat([res_aad["log"], res_ll["log"]], ignore_index=True)
    check(set(sands) == {"AccumulatedAnnualDemand", "TotalTechnologyAnnualActivityLowerLimit"},
          "regionalizador: genera SAND por parámetro")

    # Equivalencia con las salidas de los notebooks originales (SANDs_Reducidos)
    gen = sands["AccumulatedAnnualDemand"].set_index("FUEL")[anios].apply(
        pd.to_numeric, errors="coerce").sort_index()
    ref = sand_io.cargar_sand(RAIZ / "SANDs_Reducidos" / "SAND_AccumulatedAnnualDemand_Regional_Industrial.xlsx",
                              hoja=0, validar=False)
    ref = ref[ref["FUEL"].notna()]  # el archivo curado trae filas vacías
    cols = [c for c in anios if c in ref.columns]
    ref = ref.set_index("FUEL")[cols].astype(float).sort_index()
    comunes = gen.index.intersection(ref.index).unique()
    diff = (gen.loc[comunes, cols] - ref.loc[comunes]).abs().max().max()
    check(set(ref.index) <= set(gen.index) and diff < 1e-9,
          f"regionalizador: demanda reproduce el notebook original (Δmax={diff:.1e})")

    gen_l = sands["TotalTechnologyAnnualActivityLowerLimit"].set_index("TECHNOLOGY")[anios].apply(
        pd.to_numeric, errors="coerce").sort_index()
    ref_l = sand_io.cargar_sand(RAIZ / "SANDs_Reducidos" / "SAND_LowerLimit_Industria.xlsx", hoja=0, validar=False)
    ref_l = ref_l[ref_l["TECHNOLOGY"].notna()]
    cols_l = [c for c in anios if c in ref_l.columns]
    ref_l = ref_l.set_index("TECHNOLOGY")[cols_l].astype(float).sort_index()
    comunes_l = gen_l.index.intersection(ref_l.index).unique()
    diff_l = (gen_l.loc[comunes_l, cols_l] - ref_l.loc[comunes_l]).abs().max().max()
    check(set(ref_l.index) <= set(gen_l.index) and diff_l < 1e-9,
          f"regionalizador: límites reproducen el notebook original (Δmax={diff_l:.1e})")
    check((log_total["Nivel"] == "OK").sum() == 2, "regionalizador: log con 2 parámetros OK")

    # participación constante (columna 'Participacion') + centinela copiado
    pct_const = pd.DataFrame({
        "Parámetro": ["TotalAnnualMaxCapacityInvestment"] * 7,
        "TECHNOLOGY": ["DEMINDNGSBOI_LOW"] * 7,
        "FUEL": [None] * 7,
        "Región": list(cfg["prefijo_region"]),
        "Participacion": [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
    })
    ruta_const = tmp / "pct_const.xlsx"
    pct_const.to_excel(ruta_const, index=False)
    df_pct_const = regionalizador.cargar_participaciones(ruta_const)
    check((df_pct_const["Año"] == regionalizador.ANIO_CONSTANTE).all(),
          "regionalizador: detecta participación constante")
    res_max = regionalizador.regionalizar(
        df_nac, df_reg, df_pct_const,
        parametros=["TotalAnnualMaxCapacityInvestment"],
        params_otoole=otoole["param"], cfg=cfg,
        tecnologias_filtro=["DEMINDNGSBOI_LOW"], modo_filtro="exacto",
    )
    sand_max = res_max["sands"]["TotalAnnualMaxCapacityInvestment"]
    fila_nac = df_nac[(df_nac["Parameter"] == "TotalAnnualMaxCapacityInvestment")
                      & (df_nac["TECHNOLOGY"] == "DEMINDNGSBOI_LOW")].iloc[0]
    anio_chk = next((a for a in anios if utils.es_centinela(fila_nac[a], cfg["valor_centinela"], True)), None)
    if anio_chk is not None:
        vals = pd.to_numeric(sand_max[anio_chk], errors="coerce").dropna().unique()
        check(len(vals) == 1 and utils.es_centinela(vals[0], cfg["valor_centinela"], True),
              "regionalizador: centinela se copia sin repartir")
    else:
        # sin centinelas en esta fila: verificar el reparto multiplicativo
        v = utils.safe_float(fila_nac[anios[0]]) or 0
        suma = pd.to_numeric(sand_max[anios[0]], errors="coerce").sum()
        check(abs(suma - v) < 1e-6, "regionalizador: reparto constante reproduce el nacional")

    # combo sin participación: se omite con log descriptivo y el flujo continúa
    res_sin_pct = regionalizador.regionalizar(
        df_nac, df_reg, df_pct,
        parametros=["ResidualCapacity"], params_otoole=otoole["param"], cfg=cfg,
        tecnologias_filtro=["DEMINDNGSBOI_LOW"], modo_filtro="exacto",
    )
    log_sin_pct = res_sin_pct["log"]
    omitidos = log_sin_pct[(log_sin_pct["Nivel"] == "OMITIDO")
                           & log_sin_pct["Mensaje"].str.contains("Participación no encontrada")]
    check(len(omitidos) == 1 and "ResidualCapacity" not in res_sin_pct["sands"],
          "regionalizador: combo sin participación se omite con log y el flujo continúa")

    # --- regionalizador: formatos anchos estandarizados automáticamente ---------------
    # Regiones anchas + columna Anio, sin Parámetro (se infiere de la lista);
    # LowerLimit no se indexa por FUEL -> advertencia y descarte de sus filas
    ruta_fuel_res = RAIZ / "Insumos" / "Participacion_Fuel_RES.xlsx"
    if ruta_fuel_res.exists():
        pct_fuel = regionalizador.cargar_participaciones(
            ruta_fuel_res,
            parametros=["AccumulatedAnnualDemand", "TotalTechnologyAnnualActivityLowerLimit"],
            params_otoole=otoole["param"])
        check(set(pct_fuel["Parametro"].unique()) == {"AccumulatedAnnualDemand"}
              and pct_fuel["FUEL"].notna().all() and (pct_fuel["Año"] >= 2022).all(),
              "regionalizador: formato ancho Fuel|Anio|regiones (parámetro inferido, "
              "índice no aplicable descartado)")
        fuel_res = sorted(pct_fuel["FUEL"].unique())[0]
        res_fr = regionalizador.regionalizar(
            df_nac, df_reg, pct_fuel, parametros=["AccumulatedAnnualDemand"],
            params_otoole=otoole["param"], cfg=cfg,
            fuels_filtro=[fuel_res], modo_filtro="exacto")
        sand_fr = res_fr["sands"]["AccumulatedAnnualDemand"]
        nac_fr = float(df_nac[(df_nac["Parameter"] == "AccumulatedAnnualDemand")
                              & (df_nac["FUEL"] == fuel_res)].iloc[0]["2030"])
        suma_fr = pd.to_numeric(sand_fr["2030"], errors="coerce").sum()
        check(abs(suma_fr - nac_fr) < 1e-6,
              f"regionalizador: reparto desde formato ancho reproduce el nacional ({fuel_res})")
    else:
        print("       (omitido: Insumos/Participacion_Fuel_RES.xlsx no existe)")

    # Regiones anchas constantes con prefijos AN..SO y alias 'Fuel'/'Tecnologia'
    ruta_ej1 = RAIZ / "Insumos" / "ejemplo1.xlsx"
    if ruta_ej1.exists():
        pct_ej1 = regionalizador.cargar_participaciones(ruta_ej1, params_otoole=otoole["param"])
        check((pct_ej1["Año"] == regionalizador.ANIO_CONSTANTE).all()
              and set(pct_ej1["Region"].unique()) <= set(utils.REGIONES),
              "regionalizador: formato ancho constante con prefijos AN..SO")
    else:
        print("       (omitido: Insumos/ejemplo1.xlsx no existe)")

    # Solo regiones (comodín '*'): aplica a cualquier combo del parámetro
    ruta_ej3 = RAIZ / "Insumos" / "ejemplo3.xlsx"
    if ruta_ej3.exists():
        pct_ej3 = regionalizador.cargar_participaciones(
            ruta_ej3, parametros=["AccumulatedAnnualDemand"], params_otoole=otoole["param"])
        check((pct_ej3["TECHNOLOGY"] == regionalizador.COMODIN).all(),
              "regionalizador: archivo solo-regiones produce participación comodín '*'")
        res_ej3 = regionalizador.regionalizar(
            df_nac, df_reg, pct_ej3, parametros=["AccumulatedAnnualDemand"],
            params_otoole=otoole["param"], cfg=cfg,
            fuels_filtro=["INDDHT"], modo_filtro="exacto")
        sand_ej3 = res_ej3["sands"]["AccumulatedAnnualDemand"]
        nac_dht = float(df_nac[(df_nac["Parameter"] == "AccumulatedAnnualDemand")
                               & (df_nac["FUEL"] == "INDDHT")].iloc[0]["2030"])
        suma_ej3 = pd.to_numeric(sand_ej3["2030"], errors="coerce").sum()
        check(abs(suma_ej3 - nac_dht) < 1e-6,
              "regionalizador: comodín reparte manteniendo el código del nacional")
    else:
        print("       (omitido: Insumos/ejemplo3.xlsx no existe)")

    # Sin columna Parámetro y sin lista -> error claro
    if ruta_fuel_res.exists():
        try:
            regionalizador.cargar_participaciones(ruta_fuel_res)
            check(False, "regionalizador: archivo sin 'Parámetro' debe exigir la lista")
        except ValueError as exc:
            check("PARAMETROS_A_REGIONALIZAR" in str(exc),
                  "regionalizador: error claro si falta 'Parámetro' y no se pasa la lista")

    # --- sincronizador (flujo integrado, notebooks/03) ---------------------------------
    mapeo_reg = regionalizador.cargar_mapeo_regional(RAIZ / "Insumos" / "Mapeo")
    params_sync = ["AccumulatedAnnualDemand", "CapitalCost"]
    res_sync = comparador.comparar_escenarios(
        df_nac, df_reg, otoole["param"], cfg,
        modo="lista_parametros", parametros_filtro=params_sync)

    resumen_dif = sincronizador.resumen_diferencias(res_sync["comparacion"], umbral_pct=1.0)
    check(set(resumen_dif["Parametro"]) == set(params_sync)
          and set(resumen_dif["Estado"]) <= {"OK", "REVISAR"},
          "sincronizador: resumen de diferencias por parámetro")

    ruta_pct_sync = RAIZ / "Insumos" / "Mapeo" / "participaciones.xlsx"
    pct_sync = (regionalizador.cargar_participaciones(
        ruta_pct_sync, parametros=params_sync, params_otoole=otoole["param"])
        if ruta_pct_sync.exists() else None)

    dec = sincronizador.construir_decisiones(
        res_sync["comparacion"], res_sync["anomalias"], df_nac, df_reg,
        mapeo_reg, cfg, umbral_pct=1.0, df_pct=pct_sync)
    check(not dec.empty and list(dec.columns) == sincronizador.COLUMNAS_DECISIONES,
          "sincronizador: tabla de decisiones con las columnas esperadas")
    check(set(dec["ACCION"]) <= sincronizador.ACCIONES_VALIDAS,
          "sincronizador: toda ACCION inferida es válida")
    # Un renombramiento conocido (ELC -> ELC003) no debe proponerse para regionalizar:
    # la diferencia es de nomenclatura. Regresión del bug de None->NaN en apply().
    renombradas = dec[dec["Nombre_Regional"].astype(str).str.strip().ne("")
                      & dec["Nombre_Regional"].notna()]
    check(not renombradas.empty
          and not renombradas["Nombre_Regional"].astype(str).eq("nan").any()
          and (renombradas["ACCION"] == sincronizador.ACCION_IGNORAR).all(),
          "sincronizador: renombramientos detectados con su nombre real y marcados 'ignorar'")

    # Un código que existe en el regional no puede clasificarse como "faltante":
    # SIN_CORRESPONDENCIA es por parámetro, no por código.
    faltantes = dec[dec["TIPO_DIFERENCIA"].isin(
        [sincronizador.TIPO_TECH_FALTANTE, sincronizador.TIPO_FUEL_FALTANTE])]
    check(faltantes["Regiones_Existentes"].astype(str).str.strip().eq("").all(),
          "sincronizador: 'faltante' solo si el código no existe en ninguna región")
    sin_datos = dec[dec["TIPO_DIFERENCIA"] == sincronizador.TIPO_SIN_DATOS_PARAMETRO]
    check(not sin_datos.empty
          and sin_datos["Regiones_Existentes"].astype(str).str.strip().ne("").all(),
          "sincronizador: código existente sin datos del parámetro se reclasifica")

    # Cobertura parcial que coincide con lo que el mapeo espera = falso positivo
    # (solo aplica si el código existe en alguna región; si no existe en ninguna
    # es una ausencia real y se resuelve por la rama de faltantes)
    incompletas = dec[(dec["TIPO_DIFERENCIA"] == sincronizador.TIPO_REGIONES_INCOMPLETAS)
                      & dec["Regiones_Faltantes"].astype(str).str.strip().eq("")
                      & dec["Regiones_Existentes"].astype(str).str.strip().ne("")
                      & dec["Nombre_Regional"].astype(str).str.strip().eq("")]
    check(not incompletas.empty
          and (incompletas["ACCION"] == sincronizador.ACCION_IGNORAR).all(),
          "sincronizador: cobertura parcial esperada por el mapeo se marca 'ignorar'")

    sin_regla = dec["Motivo_Accion"].str.startswith("Sin regla aplicable").sum()
    check(sin_regla == 0,
          f"sincronizador: ninguna diferencia cae al fallback sin regla ({sin_regla})")

    ruta_dec = tmp / "decisiones_pendientes.xlsx"
    sincronizador.exportar_decisiones(dec, ruta_dec)
    dec_leida = sincronizador.leer_decisiones(ruta_dec)
    check(len(dec_leida) == len(dec)
          and dec_leida["ACCION"].tolist() == dec["ACCION"].tolist(),
          "sincronizador: ida y vuelta por Excel conserva las decisiones")

    dec_mala = dec.head(3).copy()
    dec_mala["ACCION"] = "accion_inventada"
    sincronizador.exportar_decisiones(dec_mala, tmp / "dec_mala.xlsx")
    try:
        sincronizador.leer_decisiones(tmp / "dec_mala.xlsx")
        check(False, "sincronizador: ACCION inválida debe abortar en modo estricto")
    except ValueError as exc:
        check("no reconocida" in str(exc), "sincronizador: ACCION inválida aborta con error claro")
    tolerante = sincronizador.leer_decisiones(tmp / "dec_mala.xlsx", estricto=False)
    check((tolerante["ACCION"] == sincronizador.ACCION_MANTENER).all(),
          "sincronizador: modo no estricto degrada la ACCION inválida a mantener_regional")

    if pct_sync is not None:
        aplicado = sincronizador.aplicar_decisiones(
            df_nac, df_reg, pct_sync, dec_leida, otoole["param"], cfg, mapeo_reg,
            years_filtro=list(range(2022, 2055)))
        n_reg = (dec_leida["ACCION"] == sincronizador.ACCION_REGIONALIZAR).sum()
        check(bool(aplicado["sands"]) == (n_reg > 0),
              f"sincronizador: aplica las {n_reg} decisiones 'regionalizar'")
        # Solo se tocan los combos decididos: nada fuera de la lista
        decididos = {(utils.normalize_text(f["TECHNOLOGY"]), utils.normalize_text(f["FUEL"]))
                     for _, f in dec_leida[dec_leida["ACCION"]
                                           == sincronizador.ACCION_REGIONALIZAR].iterrows()}
        bases = set()
        for df_s in aplicado["sands"].values():
            for col in ("TECHNOLOGY", "FUEL"):
                bases |= {utils.split_regional_name(v, regiones)[1]
                          for v in df_s[col].dropna().unique()}
        esperados = {c for combo in decididos for c in combo if c}
        check(bases <= esperados,
              "sincronizador: la regionalización se limita a los combos decididos")

        reg_sync = sincronizador.integrar_sands(df_reg, aplicado["sands"])
        filas_nuevas = sum(len(d) for d in aplicado["sands"].values())
        check(len(reg_sync) >= len(df_reg) and len(reg_sync) <= len(df_reg) + filas_nuevas,
              "sincronizador: el SAND sincronizado sustituye filas en vez de duplicarlas")

        res_desp = comparador.comparar_escenarios(
            df_nac, reg_sync, otoole["param"], cfg,
            modo="lista_parametros", parametros_filtro=params_sync)
        resolucion = sincronizador.comparar_resolucion(
            res_sync["comparacion"], res_desp["comparacion"], dec_leida, umbral_pct=1.0)
        m = resolucion["metricas"]
        check(m["resueltas"] > 0 and m["nuevas"] == 0,
              f"sincronizador: {m['resueltas']} diferencias resueltas, {m['nuevas']} nuevas")
        check(set(resolucion["detalle"]["Estado"]) <= {"RESUELTA", "PERSISTE", "NUEVA"},
              "sincronizador: estados de resolución válidos")
    else:
        print("       (omitido: Insumos/Mapeo/participaciones.xlsx no existe)")

    # --- reporte ---------------------------------------------------------------------
    ruta_rep = reporte.exportar_reporte_comparacion(
        tmp / "reporte_comp.xlsx", comp, res["anomalias"], res["resumen"],
        umbral_alerta_pct=cfg["umbral_alerta_pct"],
    )
    check(ruta_rep.exists(), "reporte: Excel de comparación con 3 hojas")
    ruta_ext = reporte.exportar_reporte_comparacion(
        tmp / "reporte_ext.xlsx", comp, res["anomalias"], res["resumen"],
        umbral_alerta_pct=cfg["umbral_alerta_pct"],
        hojas_extra={"Sector_Detalle": rv, "Vacia": pd.DataFrame()},
    )
    from openpyxl import load_workbook
    hojas_ext = load_workbook(ruta_ext, read_only=True).sheetnames
    check("Sector_Detalle" in hojas_ext and "Vacia" not in hojas_ext,
          "reporte: hojas_extra se agregan y las vacías se omiten")
    top = reporte.top_diferencias(comp, n=5)
    check(len(top) <= 5 and "Diferencia_Abs_Total" in top.columns, "reporte: top-N de diferencias")
    fig = reporte.grafica_top_diferencias(comp, n=5, ruta_png=tmp / "top.png")
    check(fig is not None and (tmp / "top.png").exists(), "reporte: gráfica top-N exportada a PNG")
    rutas_sand = regionalizador.escribir_sands(sands, tmp, "SmokeTest")
    check(len(rutas_sand) == 2 and all(r.exists() for r in rutas_sand), "reporte: SANDs reducidos escritos")
    ruta_log = reporte.escribir_log_regionalizacion(log_total, tmp)
    check(all(r.exists() for r in ruta_log), "reporte: log de regionalización (xlsx + txt)")

    print(f"\n{'='*60}")
    if FALLOS:
        print(f"{len(FALLOS)} FALLOS:")
        for f in FALLOS:
            print(f"  - {f}")
        sys.exit(1)
    print("TODOS LOS SMOKE TESTS OK")


if __name__ == "__main__":
    main()

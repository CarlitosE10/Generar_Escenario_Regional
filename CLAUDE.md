# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Regionalización del modelo energético nacional OSeMOSYS de la UPME (Colombia) en 7 regiones. No es un paquete de software: es un conjunto de notebooks de Jupyter que transforman archivos Excel en formato **SAND** hacia CSVs de **otoole**, generan parámetros regionales a partir del SAND Nacional, y validan que Regional ≡ Nacional. Todo el código, comentarios y salidas están en español. No hay tests ni build; el "run" es ejecutar notebooks de arriba hacia abajo.

Dependencias: `pandas`, `openpyxl`, `pyyaml`, `otoole`, `highspy`, `matplotlib`, Jupyter.

## Módulos reutilizables (`src/`) y orquestadores (`notebooks/`)

La lógica de comparación y regionalización está refactorizada en módulos importables, conducidos por los índices que `config_depurado.yaml` define para cada parámetro:

- `src/utils.py` — convenciones SAND, prefijos de región, `configurar_logging`, `es_centinela` (99999 y patrón "solo 9s"), `norm_indice` (para cruces: `1.0` → `'1'`), `split_regional_name`.
- `src/sand_io.py` — `cargar_sand` / `escribir_sand` / `columnas_anio` / `a_formato_largo` / `construir_sand_salida`.
- `src/yaml_parser.py` — `cargar_config_otoole` (38 params/7 sets por tipo), `cargar_params_config`, `cargar_paths_config` (resuelve rutas anidadas contra la raíz).
- `src/comparador.py` — `comparar_escenarios(modo='general'|'parametro'|'lista_parametros')`: ruta aditiva (nacional = suma de regiones, centinelas excluidos) vs intensiva (región por región), parámetros sin YEAR vía `Time indipendent variables`, y detección de anomalías: por parámetro (SIN_CORRESPONDENCIA, PARAMETRO_NO_EN_REGIONAL, PARAMETRO_SIN_VALORES, FUEL_NO_COINCIDE, REGIONES_INCOMPLETAS) y globales (TECHNOLOGY_NO_EN_REGIONAL / FUEL_NO_EN_REGIONAL: código nacional ausente del regional en TODOS los parámetros consultados que lo usan, reportado una sola vez con `Parametro = "TODOS (consultados)"`). Modo `general` ≈ 20 s / 2.1M filas. También inventario estructural (`comparar_parametros`, `comparar_dimension`).
- `src/regionalizador.py` — `cargar_participaciones(path, parametros=None, params_otoole=None)` estandariza automáticamente los formatos de participación: canónico (años anchos o columna `Participacion` constante), regiones anchas (columnas Antioquia..Suroccidente o AN..SO, con `Anio` largo o constantes), archivos sin columna `Parámetro` (se replican para `parametros` = PARAMETROS_A_REGIONALIZAR; obligatorio pasar la lista) y archivos solo-regiones → participación comodín `*` (aplica a cualquier combo del parámetro vía fallback en `_participacion_combo`, manteniendo los códigos del nacional). Alias de encabezado: Parámetro/Parameter, Tecnología/Tecnologia/Technology, Fuel, Región/Region, Año/Anio/Year. Con `params_otoole` valida la indexación: si un parámetro no se indexa por la dimensión clave del archivo, advierte y descarta esas filas. `validar_participaciones` (suma ≈ 1), `regionalizar` (aditivo = nacional × participación con centinelas copiados; intensivo = copia a regiones existentes; omite y loguea códigos sin correspondencia regional; si falta el combo en participaciones se omite con log OMITIDO descriptivo y el flujo continúa), `escribir_sands` (por parámetro o consolidado).
- `src/reporte.py` — `exportar_reporte_comparacion` (hojas Comparacion/Anomalias/Resumen, resaltado rojo si `|Diferencia %| >` umbral, trunca a 200k filas), `top_diferencias` + `grafica_top_diferencias` (exporta PNG), `escribir_log_regionalizacion` (xlsx + txt con timestamp).

Config declarativa en `config/`:
- `params_config.yaml` — `parametros_intensivos` (todo lo demás es aditivo), `valor_centinela` + `centinela_patron_9s`, `prefijo_region`, `region_osemosys` (RE1), tolerancias/umbral/top-N.
- `paths_config.yaml` — rutas anidadas: `escenario_nacional.sand/config_yaml`, `escenario_regional.sand`, `participaciones.archivo`, `outputs.reportes/sands_reducidos`.
- `participaciones_industriales.xlsx` — formato canónico `Parámetro | TECHNOLOGY | FUEL | Región | 2022..2055`; se regenera desde el legado con `python scripts/generar_participaciones_industriales.py`. `cargar_participaciones` también acepta los formatos anchos (ver `src/regionalizador.py` arriba), p. ej. `Insumos/Participacion_Fuel_RES.xlsx` (`Fuel|Anio|regiones`) o archivos solo-regiones (comodín).

Orquestadores (*Run All* de punta a punta, importan `src/` resolviendo la raíz): `notebooks/01_Comparacion.ipynb` (filtros configurables, reporte a `reportes/` + PNG) y `notebooks/02_Regionalizacion.ipynb` (SANDs reducidos a `SANDs_Reducidos/SAND_{Parametro}_{descripcion}.xlsx` + log a `reportes/`).

Smoke tests: `python tests/test_smoke.py` desde la raíz (verifica además que la regionalización reproduce a ~1e-14 las salidas de los notebooks originales en `SANDs_Reducidos/`). Nota: donde el nacional no tiene valor, los SAND generados dejan la celda vacía (los archivos curados antiguos escribían 0 — mismo significado para otoole).

Los notebooks originales de la raíz siguen siendo la referencia histórica del pipeline (sección siguiente), pero para trabajo nuevo de comparación/regionalización usar los módulos.

## Comando clave

Convertir un directorio CSV (salida de `SAND_a_CSV.ipynb`) al Excel estándar de otoole:

```bash
otoole convert csv excel CSV_Regional salida.xlsx config_depurado.yaml
```

`config_depurado.yaml` es la fuente de verdad de índices y dtypes de los 38 parámetros y 7 sets (formato config de otoole).

## Formato SAND (convención central)

Un SAND es un Excel con **una sola hoja** (`Parameters`, o `Hoja1` en los reducidos) con todos los parámetros apilados. Columnas fijas de dimensión:

```
Parameter, REGION, TECHNOLOGY, EMISSION, MODE_OF_OPERATION, FUEL, TIMESLICE, STORAGE, REGION2, Time indipendent variables
```

- **`Time indipendent variables`** está mal escrito así en los archivos — usar esa grafía exacta. Guarda el valor de parámetros sin índice YEAR.
- Los años van como columnas anchas 2022–2055 (encabezados de texto). Los notebooks de regionalización/comparación excluyen 2055 (`ANIO_MAX = 2054`).
- El set `REGION` de OSeMOSYS es siempre `RE1`: la regionalización NO usa el índice REGION, se codifica como **prefijo** en TECHNOLOGY y FUEL.

### Prefijos de región (canónicos, verificados contra `CSV_Regional/FUEL.csv`)

Antioquia=`AN`, Caribe=`CA`, Este=`SE`, Insular=`IN`, Nordeste=`NE`, Oriente=`OR`, Suroccidente=`SO`. Formato: `<prefijo>_<código nacional>` (ej. `AN_INDCLIM`, `SE_PWRSTD`). Ojo: "Este" → `SE`, no `ES`. (El markdown de `validacion_nacional_vs_regional.ipynb` usa otros nombres de región — Andina, Santanderes, etc.; los canónicos son los de `mapeo_region` en `Demanda_Regionalizada_Industrial.ipynb`.)

### Clasificación de parámetros al regionalizar/validar

- **Aditivos** (demandas, capacidades, límites de actividad): valor nacional = suma de las 7 regiones; regional = nacional × porcentaje de participación.
- **No aditivos / valor fijo** (ratios, factores, costos unitarios, vidas útiles: `InputActivityRatio`, `OutputActivityRatio`, `CapitalCost`, etc.): cada región copia el valor nacional tal cual.
- **Condicionales** (`TotalAnnualMaxCapacityInvestment`, `TotalTechnologyAnnualActivityUpperLimit`, `TotalTechnologyModelPeriodActivityUpperLimit`): si la fila nacional es toda ceros o patrón "solo 9s" (9999, 99999…) se trata como no aditivo (copiar); si tiene valores reales, es aditivo y requiere porcentaje.

Tolerancias usadas: comparación a 3 decimales en regionalización; `1e-4` absoluta en validaciones.

## Pipeline (orden de ejecución)

1. **`Participacion_Regional_Industrial.ipynb`** — lee `Insumos/Datos_Industrial_Regionalizado.xlsx` (exporte LEAP, una hoja por región, encabezado en fila 6) y calcula participaciones regionales por tecnología y por FUEL → `Insumos/Participacion_Regional_Industrial.xlsx`. Incluye reparación de rutas `Branch` truncadas a 100 caracteres por el exporte de origen (se restauran por prefijo contra los segmentos canónicos `Eficiencia_existente`, `Mejor eficiencia_Colombia`, `Mejor eficiencia_internacional`).
2. **`Demanda_Regionalizada_Industrial.ipynb`** — `AccumulatedAnnualDemand` industrial regional = nacional × participación (hoja `Participacion_Fuel`) → `Insumos/AccumulatedAnnualDemand_Regional_Industrial.xlsx` en formato SAND.
3. **`Asignacion_Limites_Regionales_Industria.ipynb`** — reparte `TotalTechnologyAnnualActivityLowerLimit` industrial usando la hoja `Participacion_Plano` y el diccionario `MAPEO_TECNOLOGIA` (código OSeMOSYS → ruta LEAP `Uso\fuel\eficiencia`) → SAND en `SAND_Regional/`.
4. **`regionalizacion_nacional_regional.ipynb`** — regionaliza el sector TRA "regional-first": itera las filas del Regional existente, busca el equivalente nacional, clasifica aditivo/no aditivo/condicional y recalcula solo lo que no cuadra a 3 decimales. Espera `Nacional/`, `Regional/`, `Configuracion.xlsx` y `output/` — rutas de otro entorno; ajustar a `SAND_Nacional_base/` y `SAND_Regional/` antes de ejecutar aquí.
5. **`SAND_a_CSV.ipynb`** — convierte cualquier SAND al directorio CSV de otoole (melt de años, sets derivados de valores únicos, CSVs vacíos para parámetros ausentes). Cambiar `SAND_FILE` / `OUTPUT_DIR` en la celda de configuración para convertir otro SAND (incluidos los de `SANDs_Reducidos/`, que solo traen un subconjunto de parámetros).
6. **`Modelo_Regional.ipynb`** — otoole convert + resolución del LP con GLPK o HiGHS (`highspy`).

Validaciones (comparan Nacional vs Regional y exportan reportes Excel):
- `Validacion_Demanda_Nacional_vs_Regional.ipynb` — solo `AccumulatedAnnualDemand` TRA*.
- `Validacion_General_Nacional_vs_Regional.ipynb` — multi-parámetro, configurable por listas de filtro; distingue aditivos vs `PARAMETROS_VALOR_FIJO` → `Reporte_Validacion_Nacional_vs_Regional.xlsx`.
- `validacion_nacional_vs_regional.ipynb` — comparación exhaustiva con gráficos (también usa rutas `Nacional/`/`Regional/` de otro entorno).

## Directorios de datos

- `SAND_Nacional_base/` — SAND nacional de referencia (`01-04-2026 SAND BASE v10.xlsx`).
- `SAND_Regional/` — escenarios SAND regionales (entrada de `SAND_a_CSV.ipynb`).
- `SANDs_Reducidos/` — SANDs parciales (un parámetro o pocos); los archivos `~$*.xlsx` son locks de Excel, ignorarlos.
- `CSV_Regional/`, `CSV_Nacional/`, `CSV_desde_SAND/`, `CSV/` — salidas otoole-CSV de distintas conversiones.
- `Insumos/` — datos de entrada (exporte LEAP industrial) y participaciones calculadas.
- `Excel/` — plantilla otoole original.

Los notebooks usan rutas relativas al directorio del proyecto: ejecutar Jupyter desde la raíz.

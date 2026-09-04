# Volcado de memoria — Antecedentes proyecto PEN Regionalizado / SAND Regionalizado (UPME)

**Fecha de extracción:** 2026-09-02
**Fuentes usadas:** `CLAUDE.md` del repositorio (`Generar_Escenario_Regional`), historial de commits de git, memoria persistente de sesiones anteriores con Claude. **No tengo acceso al texto literal de conversaciones de chat anteriores a esta sesión** (solo a lo que quedó documentado en estos archivos), así que es probable que falten detalles que solo se discutieron verbalmente y no se registraron en el repo. Los marco con ⚠️ donde soy consciente de un vacío.

Contexto general: el proyecto regionaliza el modelo energético nacional OSeMOSYS de la UPME (Colombia) en 7 regiones (Antioquia=AN, Caribe=CA, Sureste=SE, Insular=IN, Nordeste=NE, Oriente=OR, Suroccidente=SO), transformando archivos Excel formato **SAND** hacia CSVs de **otoole**, generando parámetros regionales a partir del SAND Nacional, y validando Regional ≡ Nacional. Se enmarca en la Sección 6 del documento "Estructura metodológica: Regionalización del Plan Energético Nacional (PEN) 2025-2055".

---

## 1. Errores generales identificados

1. **Columna `Año` presente pero vacía interpretada como "participación por año"** (`src/regionalizador.py::cargar_participaciones`). El archivo `participaciones.xlsx` de transporte traía la columna `Año` presente pero completamente vacía; la rama "por año" filtraba con `Año.notna()`, lo que descartaba **todas** las filas → 0 participaciones cargadas → todos los combos caían en `SIN_PARTICIPACION`. Bug detectado en 2026-07.
   **Solución:** una columna `Año` presente pero toda vacía se trata ahora como ausente = participación constante.

2. **Filas duplicadas en la lista de parámetros a regionalizar.** Si `PARAMETROS_A_REGIONALIZAR` traía un parámetro repetido, cada réplica generaba la misma fila de participación; un combo `(Region, Año)` repetido hacía que `pct_combo.get()` devolviera una `Series` en vez de un escalar, lanzando `"truth value of a Series is ambiguous"` al regionalizar.
   **Solución:** `cargar_participaciones` descarta filas duplicadas exactas, y `detectar_omisiones`/`regionalizar_con_mapeo` deduplican la lista de parámetros antes de procesar.

3. **`_decisiones_dict` no casaba las decisiones tomadas en modo interactivo con los combos a regionalizar** (bug corregido 2026-07-22). `iterrows` convierte `None` → `NaN`, y sin normalizar `TECHNOLOGY`/`FUEL` con `normalize_text` ninguna decisión (`crear_todas`, `crear_regiones:...`, `crear_existentes`, etc.) casaba con su combo. **Consecuencia real:** corridas previas con decisiones `crear_*` las ignoraron en silencio, quedando `Combos_Creados=0` sin ningún error visible.
   **Solución:** normalizar `TECHNOLOGY`/`FUEL` con `normalize_text` antes de construir el diccionario de decisiones.

4. **Cuellos de botella de rendimiento en la generación del LP multi-región** (modelo de 7 regiones). Se exploraron `linopy` y una interfaz persistente de Gurobi como alternativas más rápidas a la generación estándar vía Pyomo.
   **Solución/mitigación:** una app del equipo de datos para simular jobs redujo el tiempo de simulación de ~1h40min a 10-20 min, facilitando la iteración.

5. **Generación de reportes Excel con estilo condicional celda por celda tardaba horas** con cientos de miles de filas (reportes de comparación Nacional vs Regional).
   **Solución:** `src/reporte.py::exportar_reporte_comparacion` usa **una sola regla de formato condicional** (resaltado rojo si `|Diferencia %| > umbral`) en vez de aplicar estilos por celda, y trunca la hoja a 50k filas ordenadas por `|Diferencia|` descendente.

6. **Conflictos de entorno conda** entre versiones de NumPy/pandas/GLPK en el stack Python de UPME — resueltos (mencionado en memoria, sin más detalle documentado en el repo). ⚠️

7. **Errores de MathProg/GLPK** depurados al resolver el LP regional — mencionado en memoria pero sin bitácora específica en `CLAUDE.md` de cuál era el error exacto. ⚠️

---

## 2. Errores en la estructura del modelo

1. **Discrepancia entre el PDF de referencia del modelo y el código real (`model_definition.py`).** El PDF describía una función objetivo de **maximización** con beneficios sociales/ambientales y restricciones de empleo/economía circular. Se confirmó revisando el código que ese planteamiento **nunca se implementó**: el modelo real únicamente **minimiza costos**.
   **Ajuste:** se corrigió el borrador de la Sección 6 del documento metodológico para no mencionar la alternativa de maximización (6.1.2, solo se describe la función objetivo vigente); también se eliminaron de 6.1.4 las extensiones "restricciones definidas por el usuario" y "límites de actividad por modo" (nunca se usaron), y se eliminó la subsección 6.1.5 completa.

2. **Reclasificación de la anomalía `SIN_CORRESPONDENCIA`.** El comparador la reportaba como si la tecnología/fuel faltara en el regional, pero es una anomalía **por parámetro**: si el código sí existe en el regional (en otros parámetros), no es una tecnología faltante sino que ese parámetro específico no está poblado para ese código (`sin_datos_para_el_codigo`) — y sí se puede regionalizar.

3. **Falso positivo en `REGIONES_INCOMPLETAS`.** El comparador comparaba la cobertura regional contra las 7 regiones totales, pero cuando el mapeo esperaba menos regiones (p. ej. una tecnología que solo existe en 3) y la cobertura coincidía con lo esperado, se marcaba igual como incompleta.
   **Ajuste:** se reclasifica como `ignorar` cuando la cobertura coincide con las regiones esperadas del mapeo. **Impacto reportado:** sin estas dos reglas (esta y la anterior de `SIN_CORRESPONDENCIA`), ~75% de los casos caía en un fallback inútil (falsos positivos sin acción real).

4. **Manejo de `PARTICIPACION_REGION_INEXISTENTE`** (bug detectado en el sector residencial, 2026-07). Un código existe solo en algunas regiones, pero el archivo de participaciones le asignaba % > 0 a una región donde el código **no existe**. Sin control, la rama aditiva de `regionalizar_con_mapeo` emitía valores según las regiones del archivo de participaciones y terminaba creando un código inexistente en el modelo regional.
   **Ajuste de lógica:** se estableció el orden "existencia primero, participación después"; la decisión por defecto es `crear_existentes` (emitir solo donde el código sí existe, soltando la fracción de participación de las regiones inexistentes — nunca `omitir`, que descartaría el código entero). La rama aditiva ahora acota la emisión a `(regiones del % ∩ existentes) ∪ creadas`, y registra en el log de ADVERTENCIA las regiones descartadas.

5. **Refactorización arquitectónica del pipeline**: de notebooks monolíticos originales (`regionalizacion_nacional_regional.ipynb`, etc., con rutas de otro entorno tipo `Nacional/`/`Regional/`/`Configuracion.xlsx`) a **módulos reutilizables** en `src/` (`utils`, `sand_io`, `yaml_parser`, `comparador`, `regionalizador`, `reporte`, `sincronizador`, `residencial_in_se`) orquestados por notebooks numerados `00_Generar_Mapeo` → `01_Comparacion` → `02_Regionalizacion` → `03_Flujo_Integrado` → `04_Residencial_IN_SE`.

6. **Eliminación de la configuración de rutas centralizada en YAML** (`config/paths_config.yaml` + `yaml_parser.cargar_paths_config`, ambos eliminados). Decisión de diseño: cada notebook (01/02/03/04) y `tests/test_smoke.py` define sus propias rutas inline con `pathlib` en una celda "Configuración de rutas" (`SAND_NACIONAL`, `SAND_REGIONAL`, `CONFIG_OTOOLE`, `DIR_REPORTES`, `DIR_SANDS_REDUCIDOS`, `RUTA_PARTICIPACIONES`), por preferencia explícita de mantenerlas visibles en el notebook en vez de un archivo de configuración separado.

7. **Corrección al principio de regionalización descrito en el documento metodológico (sección 6.6.1).** No es correcto describir la desagregación espacial como una réplica uniforme en las 7 regiones: la presencia de cada tecnología/energético depende de su existencia real o potencial por región (puede estar presente en 1 hasta 7 regiones). Además existen tecnologías/energéticos **sin prefijo regional** que se tratan como elementos globales/nacionales del modelo (distintos de las tecnologías `TRN*` de intercambio).

---

## 3. Errores de datos

1. **Rutas `Branch` truncadas a 100 caracteres en el exporte LEAP industrial** (`Insumos/Datos_Industrial_Regionalizado.xlsx`, insumo de `Participacion_Regional_Industrial.ipynb`). El exporte de origen (LEAP) truncaba las rutas jerárquicas.
   **Corrección:** reparación por prefijo contra los segmentos canónicos conocidos (`Eficiencia_existente`, `Mejor eficiencia_Colombia`, `Mejor eficiencia_internacional`).

2. **Correspondencia FUEL nacional → base regional no es 1 a 1 en `diccionario.xlsx`.** El archivo guía propone varias bases posibles para un mismo código nacional (p. ej. `ELC` → `ELC001` y `ELC003`; `OIL` → tres variantes `OIL001_*`). Leerlo directamente y recorrerlo sin más lógica dependía del orden de las filas y podía hacer que dos FUELs nacionales distintos aterrizaran en el mismo código regional (colisión: ambos escribirían p. ej. `AN_ELC003` y se pisarían entre sí).
   **Solución:** `RENOMBRES_FUEL_MANUALES` con prioridad absoluta para casos conocidos, y para el resto evidencia por similitud de Jaccard entre las tecnologías conectadas en IAR/OAR reales, priorizando bases que ya existan con prefijo regional (evitar mandar un FUEL a una base solo-global como `URN`/`FOL`/`DSL001`, lo que lo dejaría sin regionalizar), asignación greedy 1 a 1 y desempate alfabético. Sin evidencia de conexión, no se inventa el renombramiento (p. ej. `OIL` queda sin correspondencia y se genera alerta, porque un reparto 1→3 no es expresable como un simple rename).
   **Caso especial documentado:** la familia `ELC` se fijó a mano (`ELC`→`ELC001`, `ELC002`→`ELC003`, `ELC003`→`ELCEV001`, `ELC004`→`ELCEV002`) por **decisión UPME del 2026-07-21**, porque `diccionario.xlsx` la contradecía; el criterio automático por Jaccard coincide con esa decisión manual.

3. **Aristas de relleno con ratio 0 en Input/OutputActivityRatio.** El SAND nacional trae ~10.500 aristas OAR "de relleno" con `output_ratio = 0` (cada tecnología `BACKSTOP` conectada a todos los fuels por defecto), replicadas igual en los CSV. Sin filtrarlas, tanto la topología del RES como el cálculo de `EQUIVALENCIA` (tech/fuel) salían incorrectos.
   **Solución:** se descartan las aristas con ratio 0 antes de construir la topología (validado con `assert` que aborta si se viola este criterio en `00_Generar_Mapeo.ipynb`).

4. **Sector residencial en las regiones aisladas Insular (IN) y Sureste (SE).** El modelo simplificó el residencial en estas dos regiones: sumó las variantes urbana/rural (`_URB`/`_RUR`) y quitó el sufijo, y algunas tecnologías/fuels ni siquiera existen ahí respecto a las 5 regiones interconectadas.
   **Solución (reglas de fallback, `src/residencial_in_se.py`):** si el código sin sufijo existe, colapsa a ese código; si no, para tecnologías eléctricas cae a `DEMRESELCOTH` conservando el nivel de eficiencia (`_HIG`/`_LOW`, o `_MID` para el resto); para fuels `RESILU`/`RESTV`/`RESWHT`/`RESWSH` cae a `RESOTH`; cualquier otro caso se marca `sin_regla` y se registra para revisión manual. La recomposición se hace sobre el **valor absoluto** del nacional (no sumando fracciones/porcentajes), porque URB y RUR tienen niveles de demanda distintos y sumar participaciones sería incorrecto.

5. **Año con participación sumando 0 entre las 7 regiones ("fila-plantilla sin dato").** `00_Generar_Mapeo.ipynb` genera filas-plantilla vacías para parámetros aditivos sin datos, donde el año suma 0% entre regiones.
   **Riesgo detectado (caso "BGS", 2026-07-22):** si se repartía por ese 0%, se escribían ceros donde en realidad no había reparto definido; en un parámetro tipo *upper limit* eso convertía silenciosamente un "sin dato" en un **límite duro de 0** para esa tecnología/región.
   **Solución:** `_participacion_combo` descarta esos años en vez de repartir; si todos los años de un combo quedan descartados, el combo se marca como "sin participación" (en un condicional, sus años quedan en 0 fijado / centinela, y los años reales sin dato quedan vacíos con ADVERTENCIA en el log, en vez de fallar silenciosamente).

6. **Formato heterogéneo de los archivos de participaciones regionales.** `cargar_participaciones` tuvo que estandarizar múltiples formatos que convivían: formato canónico (años anchos o columna `Participacion` constante), formato de regiones anchas (columnas `Antioquia`…`Suroccidente` o `AN`…`SO`), archivos sin columna `Parámetro` o con la columna presente pero completamente vacía (el caso que genera hoy `00_Generar_Mapeo.ipynb`), y archivos "solo-regiones" que se tratan como comodín `*` aplicable a cualquier combo del parámetro. También se manejan alias de encabezado inconsistentes entre archivos (`Parámetro`/`Parameter`, `Tecnología`/`Tecnologia`/`Technology`, `Región`/`Region`, `Año`/`Anio`/`Year`).

7. **Traducción de rutas LEAP a códigos OSeMOSYS (industrial), obsoleta desde 2026-07.** Antes de la v2, las participaciones industriales traducían rutas LEAP con un diccionario `MAPEO_TECNOLOGIA` extraído por *parsing* de `Asignacion_Limites_Regionales_Industria.ipynb` — una fuente de error frágil basada en texto libre.
   **Solución:** desde `Participacion_Regional_Industrial_v2.ipynb` (2026-07), las participaciones industriales traen **códigos OSeMOSYS directos** (hojas `Participacion_Fuel` / `Participacion_Technology`), agrupando eficiencias LOW/MID/HIG por familia `Uso\fuel` para que compartan la misma distribución regional, eliminando el paso de traducción.

8. **Celdas vacías vs. ceros explícitos con el mismo significado.** Donde el nacional no tiene valor, los SAND generados por el nuevo flujo dejan la celda vacía; los archivos curados antiguos escribían `0` explícito para el mismo caso (mismo significado semántico para otoole, pero formato distinto) — señalado como nota de compatibilidad en el smoke test.

---

## 4. Hallazgos clave

1. **Confirmación formal de la función objetivo real del modelo**: solo minimiza costos; no existe la función de maximización de beneficios sociales/ambientales que describía el PDF de referencia (ver también sección 2, punto 1).

2. **Decisiones metodológicas deliberadas (no errores) documentadas explícitamente**, para que no se reabran como hallazgos durante la revisión:
   - Se omite la anomalía de transporte de carbón de la Tabla 3.
   - La ausencia de margen de reserva en el escenario regional es una decisión deliberada.
   - La reducción de los límites de emisión en el escenario regional es una decisión deliberada.

3. **Las 7 regiones del modelo están correctamente nombradas y verificadas**: Insular (IN), Sureste (SE), Antioquia (AN), Caribe (CA), Nordeste (NE), Oriente (OR), Suroccidente (SO). Nota de nomenclatura: la región "Sureste" es `SE`, **no** `ES`; corresponde a la columna "Este" de los insumos LEAP originales.
   ⚠️ Nota de coherencia: `CLAUDE.md` señala que el notebook `validacion_nacional_vs_regional.ipynb` usa **otros nombres de región** (Andina, Santanderes, etc.) que no son los canónicos — vale la pena verificar que ningún entregable final arrastre esa nomenclatura antigua.

4. **`REGION` en OSeMOSYS es siempre `RE1`**: la regionalización no usa el índice `REGION` del modelo; se codifica como **prefijo** de dos letras en `TECHNOLOGY` y `FUEL` (ej. `AN_INDCLIM`, `SE_PWRSTD`), formato `<prefijo>_<código nacional>`.

5. **Clasificación de parámetros al regionalizar/validar** (marco conceptual central del proyecto):
   - **Aditivos** (demandas, capacidades, límites de actividad): nacional = suma de las 7 regiones; regional = nacional × % de participación.
   - **No aditivos / valor fijo** (ratios, factores, costos unitarios, vidas útiles — `InputActivityRatio`, `OutputActivityRatio`, `CapitalCost`, etc.): cada región copia el valor nacional tal cual.
   - **Condicionales** (`TotalAnnualMaxCapacityInvestment`, `TotalTechnologyAnnualActivityUpperLimit`, `TotalTechnologyModelPeriodActivityUpperLimit`): si la fila nacional es toda ceros o sigue el patrón "solo 9s" (valor centinela, p. ej. 9999, 99999) se trata como no aditivo (se copia); si tiene valores reales, es aditivo y requiere porcentaje de participación. Se resuelven año por año: los años en 0/centinela se copian tal cual a las regiones existentes, solo los años con valor real se reparten.

6. **Impacto cuantificado de las dos reglas de reclasificación de anomalías** (`SIN_CORRESPONDENCIA` por parámetro y `REGIONES_INCOMPLETAS` vs. cobertura esperada del mapeo): sin ellas, **~75% de los casos** de discrepancia caían en un fallback inútil, es decir, generaban ruido sin indicar una acción real a tomar.

7. **Herramientas y flujo consolidado**:
   - `otoole` es la herramienta clave para regionalizar, validar y mapear parámetros (convierte SAND ↔ CSV/Excel y genera plantillas del modelo).
   - `scripts/mapeo_maestro.py` genera una tabla maestra de mapeo nacional↔regional **autocontenida**, derivada únicamente de los CSV de los escenarios (sin depender de `graphml` ni de `diccionario.xlsx`), verificada con cobertura total (cada código nacional/regional aparece exactamente una vez, 0 duplicados) — pensada para que el equipo la pueda reproducir al actualizar los insumos.
   - El flujo de sincronización (`03_Flujo_Integrado.ipynb` + `src/sincronizador.py`) resuelve el caso de sincronizar dos escenarios que **ya existen** (nacional nuevo vs. regional nuevo) sin regionalizar desde cero: clasifica cada diferencia en 4 acciones (`regionalizar`, `mantener_regional`, `crear_en_regional`, `ignorar`) para revisión y edición manual del equipo antes de aplicar.
   - La creación de códigos nuevos en el regional **nunca es automática** (`aplicar_decisiones` devuelve `pendientes_creacion` para revisión).

8. **Decisión de comunicación para audiencia no técnica**: para la presentación de resultados del modelo regionalizado ante el Ministerio, se evaluó **no nombrar "OSeMOSYS" directamente** y presentarlo institucionalmente como "el modelo/plataforma de optimización del PEN Regionalizado". ⚠️ Confirmar si esta decisión quedó cerrada o sigue en evaluación.

9. **Estructura final acordada para la Sección 6 del documento metodológico** ("Planteamiento del modelo OSEMOSYS/App"): fusiona el desglose técnico de OSeMOSYS (6.1, elaborado con Claude Code) con las subsecciones de evolución analítica y gobernanza colaborativa propuestas por un compañero de equipo (pasan a ser 6.3 y 6.4), **omitiendo** una subsección independiente de visualización/entregables (para evitar que el documento tome tono de tutorial).

10. **Trazabilidad de cambios**: los "SANDs Reducidos" (`SANDs_Reducidos/SAND_{Parametro}_{descripcion}.xlsx`) son archivos que contienen únicamente los cambios propuestos por parámetro, generados en cada corrida de regionalización — sirven tanto para trazabilidad como para que el equipo tenga insumos de prueba independientes del SAND completo.

11. **Validación de reproducibilidad**: el smoke test (`tests/test_smoke.py`) verifica que la regionalización actual reproduce, a una tolerancia de ~1e-14, las salidas de los notebooks originales guardadas en `SANDs_Reducidos/` — es decir, la refactorización a módulos no cambió los resultados numéricos del pipeline legado.

---

## ⚠️ Posibles vacíos a completar por ti

Para que el reporte y la presentación queden completos, te pediría confirmar o añadir lo siguiente, que no encontré documentado con suficiente detalle en el repo ni en mi memoria de sesiones anteriores:

- Detalle concreto de los errores de MathProg/GLPK depurados (punto 1.7) — ¿qué mensaje de error exacto y en qué notebook/parámetro?
- Detalle de los conflictos de conda NumPy/pandas (punto 1.6) — ¿versión específica en conflicto y cómo se fijó (pin de versión, entorno nuevo, etc.)?
- Estado final de la decisión de no nombrar "OSeMOSYS" ante el Ministerio (punto 4.8) — ¿ya se presentó así o sigue pendiente?
- Si hubo hallazgos o correcciones adicionales discutidos *solo en conversación* (no en el repo) sobre el sector transporte (TRA), industrial, o sobre la validación general (`Validacion_General_Nacional_vs_Regional.ipynb`) que quieras que incluya.
- Cifras o resultados concretos (no solo metodología) que quieras que el reporte/presentación resalte: por ejemplo diferencias porcentuales relevantes encontradas en la validación Nacional vs. Regional, o números de casos de omisión resueltos en la corrida final.
- Público objetivo y extensión esperada tanto del reporte como de la presentación (para cuando pasemos a redactarlos).

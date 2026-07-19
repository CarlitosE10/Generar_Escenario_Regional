"""Parseo de los archivos YAML de configuración del flujo.

- config_depurado.yaml: config de otoole (índices y dtypes de parámetros/sets).
- config/params_config.yaml: clasificación aditivos/intensivos/condicionales,
  prefijos de región y diccionarios de mapeo para la regionalización.
- config/paths_config.yaml: rutas de los archivos SAND y de participaciones.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def cargar_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_config_otoole(path: str | Path) -> dict[str, dict]:
    """Config de otoole separada por tipo: {'param': {...}, 'set': {...}, 'result': {...}}."""
    config = cargar_yaml(path)
    por_tipo: dict[str, dict] = {"param": {}, "set": {}, "result": {}}
    for nombre, spec in config.items():
        tipo = spec.get("type")
        if tipo not in por_tipo:
            raise ValueError(f"Tipo desconocido '{tipo}' para '{nombre}' en {path}")
        por_tipo[tipo][nombre] = spec
    return por_tipo


def cargar_params_config(path: str | Path) -> dict:
    """Clasificación de parámetros y convenciones. Valida las claves obligatorias.

    Estructura esperada (ver config/params_config.yaml):
      parametros_intensivos: lista de Parameter que se copian por región;
        todo parámetro que NO esté aquí se trata como ADITIVO.
      valor_centinela: valor "sin límite" excluido de la validación aditiva.
      centinela_patron_9s: además trata cualquier "solo 9s" (9999...) como centinela.
      prefijo_region: {NombreRegion: prefijo} (7 regiones)
      region_osemosys: código del set REGION (ej. RE1)
      tolerancia_comparacion / tolerancia_participacion / umbral_alerta_pct /
      anio_maximo_comparacion / top_n_grafica: parámetros numéricos del flujo.
    """
    config = cargar_yaml(path)
    obligatorias = ["parametros_intensivos", "valor_centinela", "prefijo_region", "region_osemosys"]
    faltantes = [k for k in obligatorias if k not in config]
    if faltantes:
        raise ValueError(f"Claves faltantes en {path}: {faltantes}")

    prefijos = list(config["prefijo_region"].values())
    if len(prefijos) != len(set(prefijos)):
        raise ValueError("Prefijos de región duplicados en params_config.yaml")

    # Defaults de los parámetros numéricos opcionales
    config.setdefault("centinela_patron_9s", True)
    config.setdefault("tolerancia_comparacion", 1e-4)
    config.setdefault("tolerancia_participacion", 1e-3)
    config.setdefault("umbral_alerta_pct", 0.01)
    config.setdefault("anio_maximo_comparacion", None)
    config.setdefault("top_n_grafica", 20)
    return config


def cargar_paths_config(path: str | Path, raiz: str | Path | None = None) -> dict[str, Path]:
    """Rutas del flujo resueltas contra la raíz del proyecto.

    raiz: directorio base para rutas relativas; por defecto, la carpeta que
    contiene el archivo de configuración (config/) sube un nivel.
    """
    path = Path(path)
    config = cargar_yaml(path)
    raiz = Path(raiz) if raiz is not None else path.parent.parent

    def resolver(valor: str) -> Path:
        p = Path(valor)
        return p if p.is_absolute() else (raiz / p)

    def resolver_nivel(valor):
        if isinstance(valor, dict):
            return {k: resolver_nivel(v) for k, v in valor.items()}
        return resolver(valor)

    return {clave: resolver_nivel(valor) for clave, valor in config.items()}

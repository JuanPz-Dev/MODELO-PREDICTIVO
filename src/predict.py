from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
from src.decision import predecir_con_decision

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "modelo_demanda.pkl"


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el modelo una sola vez por proceso."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {MODEL_PATH}. "
            "Ejecuta: python -m src.train"
        )

    artefacto = joblib.load(MODEL_PATH)
    return artefacto["model"], artefacto["features"]


def validar_experiencia(experiencia: dict):
    categorias_validas = {
        "CULTURAL",
        "GASTRONOMIC",
        "HISTORICAL",
        "ISLANDS_AND_BEACHES",
        "NIGHTLIFE",
    }

    categoria = experiencia["categoria"]
    precio = experiencia["precio"]
    duracion = experiencia["duracion_minutos"]
    minimo = experiencia["participantes_minimos"]
    maximo = experiencia["participantes_maximos"]
    horarios = experiencia["cantidad_horarios"]
    dias = experiencia["dias_disponibles"]
    hora = experiencia["hora_promedio"]

    if categoria not in categorias_validas:
        raise ValueError("La categoría no corresponde a Localy.")

    if precio <= 0 or duracion <= 0:
        raise ValueError("El precio y la duración deben ser mayores que cero.")

    if minimo < 1:
        raise ValueError("Los participantes mínimos deben ser al menos 1.")

    if maximo < minimo:
        raise ValueError(
            "Los participantes máximos no pueden ser menores que los mínimos."
        )

    if horarios < 0 or dias < 0:
        raise ValueError(
            "Los horarios y días disponibles no pueden ser negativos."
        )

    if not 0 <= hora <= 23:
        raise ValueError("La hora promedio debe estar entre 0 y 23.")

    historico = [
        "reservas_confirmadas",
        "reservas_pendientes",
        "reservas_canceladas",
        "confirmadas_semana_anterior",
        "confirmadas_hace_dos_semanas",
        "promedio_confirmadas_ultimas_3_semanas",
        "canceladas_semana_anterior",
    ]
    if any(experiencia[campo] < 0 for campo in historico):
        raise ValueError(
            "Las cantidades del historial no pueden ser negativas."
        )


def construir_resultado(nivel, probabilidades, clases):
    resultado = {
        "nivel_demanda": str(nivel),
        "experimental": True,
        "advertencia": (
            "Modelo experimental entrenado con datos de localy. "
        ),
    }

    if probabilidades is not None:
        resultado["probabilidades_modelo"] = {
            str(clase): round(float(probabilidad), 4)
            for clase, probabilidad in zip(clases, probabilidades)
        }
    return resultado


def predecir_demanda_lote(experiencias: list[dict]):
    """Predice la demanda de varias experiencias en una sola ejecución."""

    if not experiencias:
        return []

    for experiencia in experiencias:
        validar_experiencia(experiencia)

    modelo, features = cargar_modelo()
    datos = pd.DataFrame(experiencias)

    faltantes = [
        columna for columna in features
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan características requeridas por el modelo: {faltantes}"
        )

    datos = datos[features]

    # Una sola ejecución para todas las experiencias.
    niveles = predecir_con_decision(modelo, datos)

    probabilidades_lote = None

    if hasattr(modelo, "predict_proba"):
        probabilidades_lote = modelo.predict_proba(datos)

    resultados = []

    for indice, nivel in enumerate(niveles):
        probabilidades = (
            probabilidades_lote[indice]
            if probabilidades_lote is not None
            else None
        )

        resultados.append(
            construir_resultado(
                nivel,
                probabilidades,
                modelo.classes_ if probabilidades is not None else [],
            )
        )

    return resultados


def predecir_demanda(
    categoria: str,
    precio: float,
    duracion_minutos: int,
    participantes_minimos: int,
    participantes_maximos: int,
    cantidad_horarios: int,
    dias_disponibles: int,
    hora_promedio: float,
    reservas_confirmadas: int,
    reservas_pendientes: int,
    reservas_canceladas: int,
    confirmadas_semana_anterior: int,
    confirmadas_hace_dos_semanas: int,
    promedio_confirmadas_ultimas_3_semanas: float,
    canceladas_semana_anterior: int,
):
    """Mantiene disponible la predicción individual."""

    experiencia = {
        "categoria": categoria,
        "precio": precio,
        "duracion_minutos": duracion_minutos,
        "participantes_minimos": participantes_minimos,
        "participantes_maximos": participantes_maximos,
        "cantidad_horarios": cantidad_horarios,
        "dias_disponibles": dias_disponibles,
        "hora_promedio": hora_promedio,
        "reservas_confirmadas": reservas_confirmadas,
        "reservas_pendientes": reservas_pendientes,
        "reservas_canceladas": reservas_canceladas,
        "confirmadas_semana_anterior": confirmadas_semana_anterior,
        "confirmadas_hace_dos_semanas": confirmadas_hace_dos_semanas,
        "promedio_confirmadas_ultimas_3_semanas":
            promedio_confirmadas_ultimas_3_semanas,
        "canceladas_semana_anterior": canceladas_semana_anterior,
    }

    return predecir_demanda_lote([experiencia])[0]


if __name__ == "__main__":
    resultado = predecir_demanda(
        categoria="CULTURAL",
        precio=50000,
        duracion_minutos=120,
        participantes_minimos=2,
        participantes_maximos=15,
        cantidad_horarios=4,
        dias_disponibles=3,
        hora_promedio=10.0,
        reservas_confirmadas=3,
        reservas_pendientes=1,
        reservas_canceladas=0,
        confirmadas_semana_anterior=2,
        confirmadas_hace_dos_semanas=4,
        promedio_confirmadas_ultimas_3_semanas=3.0,
        canceladas_semana_anterior=1,
    )

    print("\n=== PREDICCIÓN EXPERIMENTAL DE DEMANDA ===")
    print(f"Nivel estimado: {resultado['nivel_demanda']}")

    print("\nProbabilidades del modelo:")
    for clase, probabilidad in resultado.get(
        "probabilidades_modelo", {}
    ).items():
        print(f"  {clase}: {probabilidad:.1%}")

    print(f"\nAdvertencia: {resultado['advertencia']}")

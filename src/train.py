from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "dataset_demanda_completo.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "modelo_demanda.pkl"

TARGET = "nivel_demanda"
TIME_COLUMN = "semana_objetivo"
NUM_TEST_WEEKS = 3

NUMERIC_FEATURES = [
    "precio",
    "duracion_minutos",
    "participantes_minimos",
    "participantes_maximos",
    "cantidad_horarios",
    "dias_disponibles",
    "hora_promedio",
    "reservas_confirmadas",
    "reservas_pendientes",
    "reservas_canceladas",
    "confirmadas_hace_dos_semanas",
    "promedio_confirmadas_ultimas_3_semanas",
]

CATEGORICAL_FEATURES = ["categoria"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABELS = ["BAJA", "MEDIA", "ALTA"]


def crear_pipeline(clasificador):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("sampler", RandomOverSampler(random_state=42)),
        ("classifier", clasificador),
    ])


def crear_modelo():
    return crear_pipeline(
        RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            class_weight=None, 
            random_state=42,
            n_jobs=-1,
        )
    )


def mostrar_metricas(nombre, y_real, predicciones):
    print(f"\n=== {nombre} ===")
    print(f"Accuracy: {accuracy_score(y_real, predicciones):.3f}")
    print(
        "F1 macro: "
        f"{f1_score(y_real, predicciones, average='macro', zero_division=0):.3f}"
    )

    print("\nReporte por clase:")
    print(classification_report(
        y_real,
        predicciones,
        labels=LABELS,
        zero_division=0,
    ))

    print("Matriz de confusión (filas=reales, columnas=predichas):")
    print(confusion_matrix(y_real, predicciones, labels=LABELS))
    print(f"Orden de clases: {LABELS}")


def validar_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No existe {DATASET_PATH}. Ejecuta primero "
            "'python -m src.extract_dataset'."
        )

    df = pd.read_csv(DATASET_PATH)

    required = FEATURES + [TARGET, TIME_COLUMN]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors="raise")
    df = df.dropna(subset=[TARGET, TIME_COLUMN]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
    df = df[df[TARGET].isin(LABELS)].copy()

    if df.empty:
        raise ValueError("No hay filas válidas para entrenar.")

    return df


def evaluar_semanas(df, semanas_prueba):
    resultados_reales = []
    resultados_baseline = []
    resultados_modelo = []

    for semana in semanas_prueba:
        train_df = df[df[TIME_COLUMN] < semana].copy()
        test_df = df[df[TIME_COLUMN] == semana].copy()

        if train_df.empty or test_df.empty:
            print(f"\nSe omite la semana {semana}: partición vacía.")
            continue

        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]

        print(f"\n{'=' * 55}")
        print(f"EVALUACIÓN DE LA SEMANA {int(semana)}")
        print(f"{'=' * 55}")
        print(f"Filas de entrenamiento: {len(train_df)}")
        print(f"Filas de prueba: {len(test_df)}")

        print("\nClases en entrenamiento:")
        print(y_train.value_counts().reindex(LABELS, fill_value=0))

        print("\nClases reales en prueba:")
        print(y_test.value_counts().reindex(LABELS, fill_value=0))

        clases_ausentes = sorted(set(LABELS) - set(y_train.unique()))
        if clases_ausentes:
            print(
                "\nADVERTENCIA: no hay ejemplos de estas clases en el "
                f"entrenamiento: {clases_ausentes}"
            )

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        pred_baseline = baseline.predict(X_test)

        modelo = crear_modelo()
        modelo.fit(X_train, y_train)
        
        # Corrección: Evaluar usando nuestra función calibrada de umbrales
        predicciones = modelo.predict(X_test)

        mostrar_metricas(
            f"MODELO BASE - SEMANA {int(semana)}",
            y_test,
            pred_baseline,
        )

        mostrar_metricas(
            f"RANDOM FOREST CON BALANCEO - SEMANA {int(semana)}",
            y_test,
            predicciones,
        )

        resultados_reales.extend(y_test.tolist())
        resultados_baseline.extend(pred_baseline.tolist())
        resultados_modelo.extend(predicciones.tolist())

    if not resultados_reales:
        raise ValueError("No fue posible evaluar ninguna semana.")

    print("\n" + "=" * 55)
    print("RESULTADOS COMBINADOS DE LAS SEMANAS DE PRUEBA")
    print("=" * 55)

    mostrar_metricas(
        "MODELO BASE - RESULTADO COMBINADO",
        resultados_reales,
        resultados_baseline,
    )

    mostrar_metricas(
        "RANDOM FOREST CON BALANCEO - RESULTADO COMBINADO",
        resultados_reales,
        resultados_modelo,
    )


def entrenar():
    df = validar_dataset()
    semanas = sorted(df[TIME_COLUMN].unique())

    print("\n=== RESUMEN DEL DATASET ===")
    print(f"Filas disponibles: {len(df)}")
    print(f"Semanas objetivo: {semanas}")
    print("\nDistribución general:")
    print(df[TARGET].value_counts().reindex(LABELS, fill_value=0))

    if len(semanas) < 4:
        raise ValueError(
            "Se necesitan al menos cuatro semanas objetivo para evaluar "
            "hasta tres semanas con datos históricos anteriores."
        )
    semanas_prueba = semanas[-NUM_TEST_WEEKS:]

    print("\n=== PLAN DE EVALUACIÓN TEMPORAL ===")
    print(f"Semanas que se evaluarán: {semanas_prueba}")
    print("Cada semana se evalúa con un modelo entrenado exclusivamente con semanas anteriores.")
    evaluar_semanas(df, semanas_prueba)

    print("\n" + "=" * 55)
    print("ENTRENAMIENTO DEL MODELO FINAL")
    print("=" * 55)

    modelo_final = crear_modelo()
    modelo_final.fit(df[FEATURES], df[TARGET])

    clases_finales = sorted(df[TARGET].unique())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": modelo_final,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classes": clases_finales,
        "target": TARGET,
        "prediction_horizon": "next_week",
        "dataset_rows": len(df),
        "target_weeks": [int(week) for week in semanas],
        "evaluation_week": int(semanas_prueba[-1]),
        "data_type": "synthetic_initial",
    }, MODEL_PATH)

    print(f"\nModelo experimental guardado en: {MODEL_PATH}")
    print(f"Filas utilizadas en el modelo final: {len(df)}")
    print(f"Clases aprendidas: {clases_finales}")
    print("Horizonte de predicción: semana siguiente.")


if __name__ == "__main__":
    entrenar()
import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOTE_MASIVO = "2026-10-24 16:04:58.273421"

OUTPUT_COMPLETO = DATA_DIR / "dataset_demanda_completo.csv"
OUTPUT_SIN_LOTE = DATA_DIR / "dataset_demanda_sin_lote.csv"


def conectar_mysql():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "localy"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def construir_dataset(servicios, reservas, horarios, calendario, output_file):
    df = servicios.merge(calendario, how="cross")

    df["fecha_creacion"] = pd.to_datetime(df["fecha_creacion"])
    
    df["semana_creacion"] = (df["fecha_creacion"].dt.strftime("%G%V").astype(int))
    df = df[df["semana_id"] >= df["semana_creacion"]].copy()

    df = df.merge(
        reservas,
        on=["servicio_id", "semana_id"],
        how="left",
    )

    df = df.merge(
        horarios,
        on="servicio_id",
        how="left",
    )

    columnas_cero = [
        "reservas_confirmadas",
        "reservas_pendientes",
        "reservas_canceladas",
        "cantidad_horarios",
        "dias_disponibles",
    ]

    df[columnas_cero] = df[columnas_cero].fillna(0)
    df["hora_promedio"] = df["hora_promedio"].fillna(0)

    df = df.sort_values(
        ["servicio_id", "fecha_inicio_semana"]
    ).reset_index(drop=True)

    agrupado = df.groupby("servicio_id", sort=False)

    # Información disponible al finalizar la semana de referencia.
    df["confirmadas_semana_anterior"] = (
        agrupado["reservas_confirmadas"].shift(0)
    )

    df["confirmadas_hace_dos_semanas"] = (
        agrupado["reservas_confirmadas"].shift(1)
    )

    df["promedio_confirmadas_ultimas_3_semanas"] = (
        agrupado["reservas_confirmadas"]
        .transform(
            lambda serie: serie.rolling(3, min_periods=1).mean()
        )
    )

    df["canceladas_semana_anterior"] = (
        agrupado["reservas_canceladas"].shift(0)
    )

    # Objetivo: demanda de la semana siguiente.
    df["reservas_objetivo"] = (
        agrupado["reservas_confirmadas"].shift(-1)
    )

    df["fecha_objetivo"] = (
        df.groupby("servicio_id", sort=False)["fecha_inicio_semana"]
        .shift(-1)
    )

    df["semana_objetivo"] = (
        df["fecha_objetivo"].dt.strftime("%G%V")
    )

    hoy = pd.Timestamp.today().normalize()
    inicio_semana_actual = hoy - pd.Timedelta(days=hoy.weekday())

    # Solo semanas objetivo que ya finalizaron.
    df = df[
        df["fecha_objetivo"] < inicio_semana_actual
    ].copy()

    objetivo = df["reservas_objetivo"]

    df["nivel_demanda"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
    )

    validos = objetivo.notna()

    # Umbrales experimentales actuales.
    df.loc[validos & (objetivo <= 1), "nivel_demanda"] = "BAJA"

    df.loc[
        validos & objetivo.between(2, 3),
        "nivel_demanda",
    ] = "MEDIA"

    df.loc[
        validos & (objetivo >= 4),
        "nivel_demanda",
    ] = "ALTA"

    columnas_historial = [
        "confirmadas_semana_anterior",
        "confirmadas_hace_dos_semanas",
        "promedio_confirmadas_ultimas_3_semanas",
        "canceladas_semana_anterior",
        "reservas_objetivo",
        "nivel_demanda",
        "semana_objetivo",
    ]

    df = df.dropna(subset=columnas_historial).copy()

    df["semana_objetivo"] = df["semana_objetivo"].astype(int)
    df["reservas_objetivo"] = df["reservas_objetivo"].astype(int)

    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\nDataset guardado: {output_file}")
    print(f"Filas utilizables: {len(df)}")

    print("\nDistribución de clases:")
    print(
        df["nivel_demanda"]
        .value_counts()
        .reindex(["BAJA", "MEDIA", "ALTA"], fill_value=0)
    )

    print("\nDistribución por semana objetivo:")
    print(
        pd.crosstab(
            df["semana_objetivo"],
            df["nivel_demanda"],
        ).reindex(columns=["BAJA", "MEDIA", "ALTA"], fill_value=0)
    )

    return df


def extraer_dataset():
    conexion = conectar_mysql()

    try:
        servicios = pd.read_sql(
            """
            SELECT
                ts.id AS servicio_id,
                ts.created_at AS fecha_creacion,
                ts.category AS categoria,
                ts.price AS precio,
                ts.duration_minutes AS duracion_minutos,
                ts.min_participants AS participantes_minimos,
                ts.max_participants AS participantes_maximos
            FROM tour_services ts
            WHERE ts.service_status = 'APPROVED'
            """,
            conexion,
        )
        
        reservas = pd.read_sql(
                """
                SELECT
                    datos.servicio_id,
                    datos.semana_id,
                    datos.es_lote_masivo,
                    datos.es_experimental,
                    
                    SUM(
                        CASE
                            WHEN datos.reservation_status IN ('PAID', 'COMPLETED')
                            THEN 1 ELSE 0
                            END
                            ) AS reservas_confirmadas,
                            
                    SUM(
                        CASE
                            WHEN datos.reservation_status = 'PENDING'
                            THEN 1 ELSE 0
                            END
                            ) AS reservas_pendientes,

                    SUM(
                        CASE
                            WHEN datos.reservation_status = 'CANCELLED'
                            THEN 1 ELSE 0
                            END
                            ) AS reservas_canceladas
            FROM (
                SELECT
                    r.tour_service_id AS servicio_id,
                    YEARWEEK(r.reservation_date, 3) AS semana_id,
                CASE 
                    WHEN r.created_at = %s THEN 1 
                    ELSE 0
                END AS es_lote_masivo,
                0 AS es_experimental,
                CONVERT(r.reservation_status USING utf8mb4)
                COLLATE utf8mb4_unicode_ci AS reservation_status
        FROM reservation r

        UNION ALL

        SELECT
            re.tour_service_id AS servicio_id,
            YEARWEEK(re.reservation_date, 3) AS semana_id,
            0 AS es_lote_masivo,
            1 AS es_experimental,
            CONVERT(re.reservation_status USING utf8mb4)
            COLLATE utf8mb4_unicode_ci AS reservation_status
        FROM reservation_experimental re

    ) AS datos

    GROUP BY
        datos.servicio_id,
        datos.semana_id,
        datos.es_lote_masivo,
        datos.es_experimental
    """,
        conexion,
        params=(LOTE_MASIVO,),
    )


        horarios = pd.read_sql(
            """
            SELECT
                sch.tour_service_id AS servicio_id,
                COUNT(*) AS cantidad_horarios,
                COUNT(DISTINCT sch.day_of_week) AS dias_disponibles,
                AVG(HOUR(sch.start_time)) AS hora_promedio
            FROM tour_schedules sch
            INNER JOIN tour_services ts
                ON ts.id = sch.tour_service_id
            WHERE ts.service_status = 'APPROVED'
            GROUP BY sch.tour_service_id
            """,
            conexion,
        )

    finally:
        conexion.close()

    if servicios.empty:
        raise ValueError("No hay servicios aprobados.")

    if reservas.empty:
        raise ValueError("No hay reservas para construir el historial.")

    servicios.to_csv(
        DATA_DIR / "servicios_aprobados.csv",
        index=False,
        encoding="utf-8",
    )

    servicios["servicio_id"] = servicios["servicio_id"].astype(int)
    reservas["servicio_id"] = reservas["servicio_id"].astype(int)
    reservas["semana_id"] = reservas["semana_id"].astype(int)

    # Calendario continuo común para ambos escenarios.
    semanas_observadas = sorted(reservas["semana_id"].unique())

    fechas = pd.to_datetime(
        pd.Series(semanas_observadas).astype(str) + "1",
        format="%G%V%u",
        errors="raise",
    )

    calendario = pd.DataFrame({
        "fecha_inicio_semana": pd.date_range(
            start=fechas.min(),
            end=fechas.max(),
            freq="7D",
        )
    })

    calendario["semana_id"] = (
        calendario["fecha_inicio_semana"]
        .dt.strftime("%G%V")
        .astype(int)
    )

    columnas_reservas = [
        "reservas_confirmadas",
        "reservas_pendientes",
        "reservas_canceladas",
    ]

    # Escenario A: todas las reservas.
    reservas_completas = (
        reservas.groupby(
            ["servicio_id", "semana_id"],
            as_index=False,
        ).agg({
        "reservas_confirmadas": "sum",
        "reservas_pendientes": "sum",
        "reservas_canceladas": "sum",
        "es_experimental": "max",})
    )

    # Escenario B: excluir únicamente el lote identificado.
    reservas_sin_lote = (
        reservas[reservas["es_lote_masivo"] == 0]
        .groupby(
            ["servicio_id", "semana_id"],
            as_index=False,
        )[columnas_reservas]
        .sum()     
    )

    print("\n========== ESCENARIO A: COMPLETO ==========")

    dataset_completo = construir_dataset(
        servicios,
        reservas_completas,
        horarios,
        calendario,
        OUTPUT_COMPLETO,
    )

    print("\n========== ESCENARIO B: SIN LOTE MASIVO ==========")

    dataset_sin_lote = construir_dataset(
        servicios,
        reservas_sin_lote,
        horarios,
        calendario,
        OUTPUT_SIN_LOTE,
    )

    print("\n========== COMPARACIÓN ==========")
    print(f"Filas completas: {len(dataset_completo)}")
    print(f"Filas sin lote: {len(dataset_sin_lote)}")
    print(f"Semanas del calendario: {len(calendario)}")

    print(
        "\nIMPORTANTE: los datos de origen son sintéticos. "
        "Los umbrales son experimentales y no representan "
        "demanda real validada."
    )


if __name__ == "__main__":
    extraer_dataset()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import predecir_demanda, predecir_demanda_lote

app = FastAPI(
    title="Localy - API de predicción de demanda",
    description=("API experimental para estimar el nivel de demanda "
                 "de experiencias turísticas durante la semana siguiente."),
    version="1.0.0",
)

class ExperienciaRequest(BaseModel):
    categoria: str
    precio: float = Field(gt=0)
    duracion_minutos: int = Field(gt=0)
    participantes_minimos: int = Field(ge=1)
    participantes_maximos: int = Field(ge=1)
    cantidad_horarios: int = Field(ge=0)
    dias_disponibles: int = Field(ge=0)
    hora_promedio: float = Field(ge=0, le=23)

    # Características históricas requeridas por el modelo.
    # En esta etapa son datos de entrada experimentales.
    reservas_confirmadas: int = Field(ge=0)
    reservas_pendientes: int = Field(ge=0)
    reservas_canceladas: int = Field(ge=0)
    confirmadas_semana_anterior: int = Field(ge=0)
    confirmadas_hace_dos_semanas: int = Field(ge=0)
    promedio_confirmadas_ultimas_3_semanas: float = Field(ge=0)
    canceladas_semana_anterior: int = Field(ge=0)


@app.get("/")
def inicio():
    return {
        "servicio": "Localy ML",
        "estado": "activo",
        "documentacion": "/docs",
        "tipo_modelo": "experimental",
    }


@app.get("/health")
def salud():
    return {"estado": "ok"}


@app.post("/predict")
def predecir(request: ExperienciaRequest):
    try:
        return predecir_demanda(**request.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.post("/predict/batch")
def predecir_lote(requests: list[ExperienciaRequest]):
    try:
        if not requests:
            raise ValueError("Debes enviar al menos una experiencia.")

        experiencias = [
            request.model_dump()
            for request in requests
        ]

        return predecir_demanda_lote(experiencias)

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

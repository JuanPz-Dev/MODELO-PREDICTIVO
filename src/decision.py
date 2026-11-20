import numpy as np


def predecir_con_decision(modelo, datos):
    """
    Utiliza directamente la clasificación del modelo Random Forest.
    No aplica factores manuales a las probabilidades.
    """
    return modelo.predict(datos)

import numpy as np
from .kinetics import integrated_model
from .parameters import calculate_characteristic_times

def predict_stability(order, k, c0, times):
    t = np.asarray(times, dtype=float)
    if np.any(t < 0) or k <= 0 or c0 <= 0:
        raise ValueError("El tiempo no puede ser negativo; k y C0 deben ser positivos.")
    concentration = integrated_model(order, t, c0, k)
    return {"time": t, "concentration": concentration,
            "remaining_percent": 100*concentration/c0,
            "characteristic_times": calculate_characteristic_times(order, k, c0)}

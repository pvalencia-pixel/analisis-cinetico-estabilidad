import numpy as np

def time_to_fraction(order, k, c0, fraction):
    """Tiempo para que C/C0 alcance una fracción dada (0 < fraction < 1)."""
    if not (0 < fraction < 1):
        raise ValueError("La fracción debe estar entre 0 y 1.")
    if k <= 0 or c0 <= 0:
        raise ValueError("k y C0 deben ser positivos.")
    if order == 0:
        return c0*(1-fraction)/k
    if order == 1:
        return -np.log(fraction)/k
    if order == 2:
        return (1/fraction-1)/(k*c0)
    raise ValueError("Orden no válido.")

def calculate_characteristic_times(order, k, c0):
    fractions = {"t50": 0.50, "t98": 0.98, "t95": 0.95, "t90": 0.90}
    return {name: float(time_to_fraction(order, k, c0, f)) for name, f in fractions.items()}

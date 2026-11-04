import numpy as np
from scipy.stats import linregress

R_GAS = 8.314462618  # J mol-1 K-1

def analyze_arrhenius(temperature_c, k_values):
    tc = np.asarray(temperature_c, dtype=float)
    k = np.asarray(k_values, dtype=float)
    if tc.ndim != 1 or k.ndim != 1 or len(tc) != len(k) or len(tc) < 3:
        raise ValueError("Ingrese al menos tres pares de temperatura y k.")
    if not np.all(np.isfinite(tc)) or not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError("Temperaturas y k deben ser numéricas; k debe ser positiva.")
    tk = tc + 273.15
    if np.any(tk <= 0) or len(np.unique(tk)) < 3:
        raise ValueError("Se requieren al menos tres temperaturas Kelvin distintas.")
    inv_t = 1.0/tk
    ln_k = np.log(k)
    reg = linregress(inv_t, ln_k)
    ea_j = -reg.slope*R_GAS
    A = float(np.exp(reg.intercept))
    return {
        "temperature_c": tc, "temperature_k": tk, "inv_temperature": inv_t,
        "k": k, "ln_k": ln_k, "slope": float(reg.slope),
        "intercept": float(reg.intercept), "r2": float(reg.rvalue**2),
        "Ea_J_mol": float(ea_j), "Ea_kJ_mol": float(ea_j/1000),
        "A": A, "R": R_GAS
    }

def predict_k(arrhenius_result, temperature_c):
    tk = np.asarray(temperature_c, dtype=float) + 273.15
    if np.any(tk <= 0):
        raise ValueError("Temperatura Kelvin no válida.")
    return arrhenius_result["A"] * np.exp(-arrhenius_result["Ea_J_mol"]/(R_GAS*tk))

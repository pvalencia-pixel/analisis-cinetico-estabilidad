import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

R_ORDERS = {
    "Orden cero": {"transform": lambda c: c, "k_unit": "concentración/tiempo"},
    "Orden uno": {"transform": np.log, "k_unit": "tiempo⁻¹"},
    "Orden dos": {"transform": lambda c: 1.0 / c, "k_unit": "concentración⁻¹·tiempo⁻¹"},
}

def _validate_time_concentration(time, concentration):
    t = np.asarray(time, dtype=float)
    c = np.asarray(concentration, dtype=float)
    if t.ndim != 1 or c.ndim != 1 or len(t) != len(c):
        raise ValueError("Tiempo y concentración deben ser vectores de igual longitud.")
    if len(t) < 3:
        raise ValueError("Se requieren al menos tres observaciones.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)):
        raise ValueError("Los datos contienen valores no numéricos o infinitos.")
    if np.any(t < 0):
        raise ValueError("El tiempo no puede ser negativo.")
    if np.any(c <= 0):
        raise ValueError("Las concentraciones deben ser mayores que cero.")
    if len(np.unique(t)) < 3:
        raise ValueError("Se requieren al menos tres tiempos diferentes.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Los tiempos deben estar ordenados de menor a mayor y no repetirse.")
    return t, c

def integrated_model(order, t, c0, k):
    t = np.asarray(t, dtype=float)
    if order == 0:
        return c0 - k*t
    if order == 1:
        return c0*np.exp(-k*t)
    if order == 2:
        return 1.0/(1.0/c0 + k*t)
    raise ValueError("El orden debe ser 0, 1 o 2.")

def analyze_kinetics(time, concentration):
    """Ajusta modelos linealizados y calcula errores en la escala original."""
    t, c = _validate_time_concentration(time, concentration)
    results = {}
    for order, label in [(0, "Orden cero"), (1, "Orden uno"), (2, "Orden dos")]:
        y = c if order == 0 else (np.log(c) if order == 1 else 1.0/c)
        reg = linregress(t, y)
        k = (-reg.slope if order in (0, 1) else reg.slope)
        valid_k = bool(k > 0)
        c0_fit = float(reg.intercept if order == 0 else
                       np.exp(reg.intercept) if order == 1 else
                       1.0/reg.intercept if reg.intercept > 0 else np.nan)
        pred_c = integrated_model(order, t, c0_fit, k) if valid_k and np.isfinite(c0_fit) and c0_fit > 0 else np.full_like(c, np.nan)
        rmse = float(np.sqrt(np.mean((c-pred_c)**2))) if np.all(np.isfinite(pred_c)) else np.nan
        results[label] = {
            "order": order, "slope": float(reg.slope), "intercept": float(reg.intercept),
            "r2": float(reg.rvalue**2), "k": float(k),
            "k_unit": R_ORDERS[label]["k_unit"], "valid_k": valid_k, "c0_fit": c0_fit,
            "rmse_original": rmse, "time": t, "observed": c, "transformed": y,
            "fitted_transformed": reg.intercept + reg.slope*t,
            "predicted_concentration": pred_c,
        }
    admissible = {name: r for name, r in results.items() if r["valid_k"] and np.isfinite(r["rmse_original"])}
    if not admissible:
        raise ValueError("Ningún modelo produjo una constante k positiva. Revise los datos.")
    results["_summary"] = {
        "best_by_r2": max(admissible, key=lambda name: admissible[name]["r2"]),
        "best_by_rmse_linearized": min(admissible, key=lambda name: admissible[name]["rmse_original"]),
        "warning": "Los R² de variables transformadas no son estrictamente comparables. Para seleccionar el modelo, compare también los ajustes no lineales en concentración original."
    }
    return results

def nonlinear_refit(time, concentration, order):
    """Ajuste no lineal en concentración original; devuelve RMSE, R², AIC, BIC y residuos."""
    t, c = _validate_time_concentration(time, concentration)
    c0_guess = float(c[0])
    duration = max(t[-1]-t[0], 1e-12)
    k_guess = max(1e-12, (c[0]-c[-1]) / duration)
    if order == 1:
        k_guess = max(1e-12, -np.log(c[-1]/c[0])/duration)
    elif order == 2:
        k_guess = max(1e-12, (1/c[-1]-1/c[0])/duration)

    def f(x, c0, k):
        return integrated_model(order, x, c0, k)

    popt, pcov = curve_fit(
        f, t, c, p0=[c0_guess, k_guess],
        bounds=([1e-12, 0], [np.inf, np.inf]), maxfev=20000
    )
    pred = f(t, *popt)
    residuals = c - pred
    rss = float(np.sum(residuals**2))
    n = len(c)
    p = 2  # C0 y k
    rss_safe = max(rss, np.finfo(float).tiny)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_tot = float(np.sum((c-np.mean(c))**2))
    r2_original = 1-rss/ss_tot if ss_tot > 0 else np.nan
    aic = float(n*np.log(rss_safe/n) + 2*p)
    bic = float(n*np.log(rss_safe/n) + p*np.log(n))
    return {
        "c0": float(popt[0]), "k": float(popt[1]), "rmse": rmse,
        "r2_original": float(r2_original), "aic": aic, "bic": bic,
        "rss": rss, "residuals": residuals, "covariance": pcov,
        "predicted": pred
    }

def compare_nonlinear_models(time, concentration):
    """Compara los tres órdenes en la escala original de concentración."""
    labels = {0: "Orden cero", 1: "Orden uno", 2: "Orden dos"}
    comparison = {}
    for order, label in labels.items():
        try:
            comparison[label] = nonlinear_refit(time, concentration, order)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            comparison[label] = {"error": str(exc)}
    valid = {name: result for name, result in comparison.items() if "aic" in result}
    if not valid:
        raise ValueError("No fue posible ajustar ningún modelo no lineal.")
    best = min(valid, key=lambda name: valid[name]["aic"])
    return comparison, best

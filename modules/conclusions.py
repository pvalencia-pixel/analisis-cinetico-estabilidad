def generate_conclusions(kinetics, arrhenius=None, prediction_temperature=None):
    """Motor de conclusiones basado en reglas; no es una IA generativa."""
    summary = kinetics["_summary"]
    selected = summary["best_by_r2"]
    model = kinetics[selected]
    lines = [
        f"El modelo candidato según el mayor R² de las regresiones integradas es {selected.lower()} (R² = {model['r2']:.5f}).",
        f"La constante cinética estimada por regresión linealizada es k = {model['k']:.6g} {model['k_unit']}.",
        f"El RMSE en concentración original para este modelo es {model['rmse_original']:.6g}.",
        "La selección debe confirmarse inspeccionando residuos, ajuste no lineal y plausibilidad fisicoquímica; los R² transformados no son directamente comparables."
    ]
    if arrhenius:
        lines.append(f"El ajuste de Arrhenius presenta R² = {arrhenius['r2']:.5f} y una energía de activación aparente de {arrhenius['Ea_kJ_mol']:.3f} kJ/mol.")
        if prediction_temperature is not None:
            lines.append(f"Para {prediction_temperature:.2f} °C, el modelo de Arrhenius estima k = {arrhenius['predicted_k']:.6g} en las mismas unidades de k experimentales.")
    lines.append("Estas estimaciones son matemáticas y dependen de la calidad de los datos, del mecanismo de degradación y del intervalo de temperaturas. No constituyen por sí solas una asignación regulatoria de vida útil.")
    return " ".join(lines)

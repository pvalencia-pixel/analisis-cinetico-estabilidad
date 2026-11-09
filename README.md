# Análisis cinético y predicción de estabilidad — V2.0

Aplicación de fisicoquímica farmacéutica para comparar modelos cinéticos integrados de orden cero, uno y dos; estimar parámetros característicos; ajustar la ecuación de Arrhenius y proyectar perfiles de concentración.

## Novedades de la versión 2.0
- Comparación de modelos no lineales en la escala original de concentración.
- Cálculo de R², RMSE, AIC y BIC para los ajustes no lineales.
- Selección orientativa del modelo con menor AIC.
- Gráfica de residuos para evaluar patrones sistemáticos.
- Validación adicional: el tiempo experimental no puede ser negativo.
- Identificación de la versión en la interfaz.

## Ejecutar localmente
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Ejecutar pruebas
```bash
pytest -q
```

## Modelos
- Orden cero: C = C0 - kt
- Orden uno: ln(C) = ln(C0) - kt
- Orden dos: 1/C = 1/C0 + kt
- Arrhenius: ln(k) = ln(A) - Ea/(RT)

## Interpretación estadística
Los R² de las regresiones linealizadas corresponden a escalas transformadas diferentes y no deben compararse como si fueran equivalentes. La V2.0 incorpora ajustes no lineales directamente sobre la concentración original. En estos ajustes se calculan RMSE, AIC y BIC; un AIC o BIC menor favorece el modelo dentro del conjunto comparado, pero no demuestra por sí solo que el mecanismo químico sea correcto. La gráfica de residuos permite detectar desviaciones sistemáticas.

AIC y BIC se comparan únicamente entre modelos ajustados a los mismos datos y a la misma variable respuesta. Si el error residual es prácticamente cero, los criterios pueden tomar valores muy negativos; esto es esperable en datos sintéticos perfectamente descritos.

## Consideraciones científicas
La predicción de Arrhenius requiere constantes cinéticas obtenidas para el mismo orden de reacción, con unidades compatibles y un mecanismo de degradación comparable en el intervalo de temperatura. La extrapolación fuera del intervalo experimental debe identificarse como estimación y no como evidencia experimental.

La herramienta es educativa/investigativa. No asigna por sí sola una vida útil regulatoria ni sustituye estudios formales de estabilidad, evaluación de métodos indicativos de estabilidad o los criterios regulatorios aplicables.

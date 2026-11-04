# Análisis cinético y predicción de estabilidad

Aplicación de fisicoquímica farmacéutica para comparar modelos cinéticos integrados de orden cero, uno y dos; calcular parámetros característicos; ajustar Arrhenius y proyectar estabilidad.

## Ejecutar localmente
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Ejecutar pruebas
```bash
pytest -q
```

## Google Colab
1. Suba el ZIP del proyecto a Colab.
2. Descomprímalo.
3. Ejecute `!pip install -r requirements.txt`.
4. Para pruebas: `!pytest -q`.
5. Para uso estable se recomienda desplegar desde GitHub en Streamlit Community Cloud.

## Modelos
- Orden cero: C = C0 - kt
- Orden uno: ln(C) = ln(C0) - kt
- Orden dos: 1/C = 1/C0 + kt
- Arrhenius: ln(k) = ln(A) - Ea/(RT)

## Consideraciones científicas
El mayor R² de las transformaciones linealizadas se presenta como criterio de selección solicitado, pero no debe interpretarse como comparación estadística perfecta entre escalas transformadas. Se informa RMSE en concentración original y se incorpora un ajuste no lineal de contraste. Verifique residuos, incertidumbre, mecanismo, unidades y validez del intervalo de temperatura.

La herramienta es educativa/investigativa y no asigna por sí sola una vida útil regulatoria ni sustituye estudios formales de estabilidad.

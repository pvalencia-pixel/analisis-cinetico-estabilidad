import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.kinetics import analyze_kinetics, nonlinear_refit, compare_nonlinear_models
from modules.parameters import calculate_characteristic_times
from modules.arrhenius import analyze_arrhenius, predict_k
from modules.stability import predict_stability
from modules.conclusions import generate_conclusions
from utils.validation import load_experimental_file, clean_time_concentration

st.set_page_config(page_title="Análisis cinético y estabilidad", page_icon="🧪", layout="wide")
st.title("🧪 Análisis cinético y predicción de estabilidad")
st.caption("Herramienta de fisicoquímica farmacéutica | Versión 2.0")

with st.sidebar:
    st.header("Navegación")
    page = st.radio("Módulo", ["Análisis cinético", "Arrhenius y predicción", "Acerca del modelo"])
    st.divider()
    st.caption("Los resultados son estimaciones matemáticas; requieren evaluación científica.")

if page == "Análisis cinético":
    st.subheader("1. Datos experimentales")
    mode = st.radio("Método de ingreso", ["Manual", "Cargar archivo"], horizontal=True)
    if mode == "Manual":
        default = pd.DataFrame({"Tiempo": [0,10,20,30,40,50], "Concentración": [100,95,90,85,80,75]})
        edited = st.data_editor(default, num_rows="dynamic", use_container_width=True)
        df = edited.rename(columns={"Tiempo":"time", "Concentración":"concentration"})
    else:
        uploaded = st.file_uploader("Cargue un CSV o Excel", type=["csv","xlsx","xls"])
        if uploaded:
            raw = load_experimental_file(uploaded)
            st.write("Vista previa", raw.head())
            c1, c2 = st.columns(2)
            time_col = c1.selectbox("Columna de tiempo", raw.columns)
            conc_col = c2.selectbox("Columna de concentración", [c for c in raw.columns if c != time_col])
            df = clean_time_concentration(raw, time_col, conc_col)
        else:
            df = None
    if df is not None and st.button("Analizar cinética", type="primary"):
        try:
            df = df[["time","concentration"]].apply(pd.to_numeric, errors="raise").sort_values("time")
            result = analyze_kinetics(df.time, df.concentration)
            st.session_state["kinetics"] = result
            st.session_state["data"] = df
            comparison, best_aic = compare_nonlinear_models(df.time, df.concentration)
            st.session_state["nonlinear_comparison"] = comparison
            st.session_state["best_by_aic"] = best_aic
            st.success("Análisis completado.")
        except Exception as e:
            st.error(str(e))

    if "kinetics" in st.session_state:
        result = st.session_state["kinetics"]
        data = st.session_state["data"]
        st.subheader("2. Comparación de modelos")
        rows = []
        for name, r in result.items():
            if name.startswith("_"): continue
            rows.append({"Modelo":name, "Pendiente":r["slope"], "Intercepto":r["intercept"],
                         "R² transformado":r["r2"], "k":r["k"], "Unidad de k":r["k_unit"],
                         "RMSE concentración":r["rmse_original"], "k positiva":r["valid_k"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info("Candidato por R²: " + result["_summary"]["best_by_r2"] + ". Contraste por menor RMSE: " + result["_summary"]["best_by_rmse"] + ".")
        st.warning(result["_summary"]["warning"])
        st.subheader("Comparación no lineal en concentración original")
        comparison = st.session_state.get("nonlinear_comparison", {})
        comparison_rows = []
        for name, fit in comparison.items():
            if "error" in fit:
                comparison_rows.append({"Modelo": name, "Estado": fit["error"]})
            else:
                comparison_rows.append({
                    "Modelo": name, "R² original": fit["r2_original"],
                    "RMSE": fit["rmse"], "AIC": fit["aic"], "BIC": fit["bic"]
                })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
        if "best_by_aic" in st.session_state:
            st.success("Mejor ajuste no lineal según AIC: " + st.session_state["best_by_aic"] + ". Un AIC menor indica mejor equilibrio entre ajuste y complejidad dentro de estos modelos.")
        st.caption("AIC y BIC solo deben compararse entre modelos ajustados a los mismos datos y en la misma escala de respuesta.")
        st.subheader("3. Gráficas integradas")
        tabs = st.tabs(["[A] vs tiempo", "ln[A] vs tiempo", "1/[A] vs tiempo"])
        transforms = [lambda x:x, np.log, lambda x:1/x]
        labels = ["Orden cero", "Orden uno", "Orden dos"]
        for i, tab in enumerate(tabs):
            with tab:
                r = result[labels[i]]
                yname = ["Concentración","ln(Concentración)","1/Concentración"][i]
                fig = go.Figure()
                fig.add_scatter(x=data.time, y=transforms[i](data.concentration), mode="markers", name="Experimental")
                fig.add_scatter(x=r["time"], y=r["fitted_transformed"], mode="lines", name="Regresión")
                fig.update_layout(xaxis_title="Tiempo", yaxis_title=yname, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Pendiente = {r['slope']:.6g} | Intercepto = {r['intercept']:.6g} | R² = {r['r2']:.6f}")
                st.latex(f"y = {r['slope']:.6g}t + ({r['intercept']:.6g})")
        st.subheader("4. Parámetros y ajuste no lineal")
        default_model = st.session_state.get("best_by_aic", result["_summary"]["best_by_r2"])
        chosen = st.selectbox("Modelo para calcular parámetros", ["Orden cero","Orden uno","Orden dos"],
                              index=["Orden cero","Orden uno","Orden dos"].index(default_model))
        rr = result[chosen]
        times = calculate_characteristic_times(rr["order"], rr["k"], rr["c0_fit"])
        c1,c2,c3,c4 = st.columns(4)
        for col,(key,val) in zip([c1,c2,c3,c4],times.items()):
            col.metric(key, f"{val:.4g}")
        try:
            nl = nonlinear_refit(data.time, data.concentration, rr["order"])
            st.write(f"Ajuste no lineal en concentración original: C₀ = {nl['c0']:.6g}; k = {nl['k']:.6g}; R² original = {nl['r2_original']:.6f}; RMSE = {nl['rmse']:.6g}; AIC = {nl['aic']:.4f}; BIC = {nl['bic']:.4f}.")
            residual_df = pd.DataFrame({"Tiempo": data.time, "Residuo": nl["residuals"]})
            residual_fig = px.scatter(residual_df, x="Tiempo", y="Residuo", title="Residuos del ajuste: " + chosen)
            residual_fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(residual_fig, use_container_width=True)
            st.caption("Los residuos deben distribuirse alrededor de cero sin patrones sistemáticos evidentes.")
        except Exception as e:
            st.warning(f"No fue posible completar el ajuste no lineal: {e}")
        st.subheader("5. Interpretación automática")
        st.write(generate_conclusions(result))

elif page == "Arrhenius y predicción":
    st.subheader("1. Datos de temperatura y constantes cinéticas")
    st.write("Introduzca k obtenidas experimentalmente para el mismo orden cinético y con unidades coherentes.")
    default_a = pd.DataFrame({"Temperatura (°C)":[40,50,60,70], "k":[0.002,0.004,0.008,0.015]})
    arr_df = st.data_editor(default_a, num_rows="dynamic", use_container_width=True)
    if st.button("Ajustar Arrhenius", type="primary"):
        try:
            ar = analyze_arrhenius(arr_df["Temperatura (°C)"], arr_df["k"])
            st.session_state["arrhenius"] = ar
            st.success("Regresión de Arrhenius completada.")
        except Exception as e:
            st.error(str(e))
    if "arrhenius" in st.session_state:
        ar = st.session_state["arrhenius"]
        st.subheader("2. Resultados")
        c1,c2,c3 = st.columns(3)
        c1.metric("Ea", f"{ar['Ea_kJ_mol']:.3f} kJ/mol")
        c2.metric("Factor A", f"{ar['A']:.5g}")
        c3.metric("R²", f"{ar['r2']:.6f}")
        arr_table = pd.DataFrame({"T (°C)":ar["temperature_c"],"T (K)":ar["temperature_k"],"1/T (K⁻¹)":ar["inv_temperature"],"k":ar["k"],"ln(k)":ar["ln_k"]})
        st.dataframe(arr_table, use_container_width=True)
        fig = px.scatter(arr_table, x="1/T (K⁻¹)", y="ln(k)", trendline="ols", title="Gráfica de Arrhenius")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Pendiente = {ar['slope']:.6g}; intercepto = {ar['intercept']:.6g}")
        target = st.number_input("Temperatura objetivo (°C)", value=25.0, step=1.0)
        kp = float(predict_k(ar, [target])[0])
        st.metric(f"k estimada a {target:.2f} °C", f"{kp:.6g}")
        st.session_state["arrhenius"]["predicted_k"] = kp
        st.session_state["arrhenius"]["target_temperature"] = target

        if "kinetics" in st.session_state:
            kin = st.session_state["kinetics"]
            default_order = kin["_summary"]["best_by_r2"]
            order_name = st.selectbox("Orden cinético para proyectar", ["Orden cero","Orden uno","Orden dos"],
                                      index=["Orden cero","Orden uno","Orden dos"].index(default_order))
            kr = kin[order_name]
            c0 = st.number_input("Concentración inicial C₀", min_value=0.000001, value=float(kr["c0_fit"]))
            horizon = st.number_input("Horizonte de predicción", min_value=1.0, value=365.0)
            unit = st.text_input("Unidad de tiempo (debe coincidir con k)", value="días")
            if st.button("Generar predicción"):
                try:
                    pred = predict_stability(kr["order"], kp, c0, np.linspace(0,horizon,150))
                    out = pd.DataFrame({"Tiempo":pred["time"],"Concentración":pred["concentration"],"% restante":pred["remaining_percent"]})
                    st.line_chart(out.set_index("Tiempo")["% restante"])
                    st.dataframe(pd.DataFrame([{"Parámetro":k,"Tiempo":v,"Unidad":unit} for k,v in pred["characteristic_times"].items()]), use_container_width=True)
                    st.download_button("Descargar predicción CSV", out.to_csv(index=False).encode("utf-8"), "prediccion_estabilidad.csv", "text/csv")
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("Para proyectar tiempos de estabilidad, primero realice un análisis cinético en la pestaña correspondiente.")

elif page == "Acerca del modelo":
    st.subheader("Fundamento y limitaciones")
    st.markdown("""
    - Orden cero: C = C₀ − kt.
    - Orden uno: ln(C) = ln(C₀) − kt.
    - Orden dos: 1/C = 1/C₀ + kt.
    - Arrhenius: ln(k) = ln(A) − Ea/(RT).

    **Criterio de selección:** el mayor R² de las regresiones linealizadas se muestra como candidato solicitado. Debido a que cada transformación cambia la escala de los errores, también se informa RMSE en concentración original y se ofrece ajuste no lineal.

    **Limitaciones:** el modelo supone un mecanismo cinético constante, condiciones experimentales comparables y una relación de Arrhenius válida en el intervalo considerado. No sustituye estudios de estabilidad formales, evaluación de especificidad analítica ni criterios regulatorios.
    """)

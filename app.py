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
st.caption("Herramienta de fisicoquímica farmacéutica | Versión 2.1")

TIME_UNITS = ["segundos (s)", "minutos (min)", "horas (h)", "días (d)", "semanas (sem)", "meses (mes)", "años (a)"]
CONCENTRATION_UNITS = ["mg/L", "µg/L", "g/L", "mg/mL", "µg/mL", "g/mL", "mol/L", "mmol/L", "µmol/L", "% (m/v)", "% (m/m)", "ppm"]
TEMPERATURE_UNITS = ["°C", "K", "°F"]

def temperature_to_celsius(values, unit):
    values = np.asarray(values, dtype=float)
    if unit == "K":
        return values - 273.15
    if unit == "°F":
        return (values - 32.0) * 5.0 / 9.0
    return values

def celsius_to_unit(value, unit):
    if unit == "K":
        return float(value) + 273.15
    if unit == "°F":
        return float(value) * 9.0 / 5.0 + 32.0
    return float(value)

def kinetic_constant_unit(order, concentration_unit, time_unit):
    if order == "Orden cero":
        return f"{concentration_unit}/{time_unit}"
    if order == "Orden uno":
        return f"1/{time_unit}"
    return f"1/({concentration_unit}·{time_unit})"

with st.sidebar:
    st.header("Navegación")
    page = st.radio("Módulo", ["Análisis cinético", "Arrhenius y predicción", "Acerca del modelo"])
    st.divider()
    st.caption("Los resultados son estimaciones matemáticas; requieren evaluación científica.")

if page == "Análisis cinético":
    st.subheader("1. Datos experimentales")
    unit_col1, unit_col2 = st.columns(2)
    time_unit = unit_col1.selectbox("Unidad de tiempo", TIME_UNITS, index=3, key="kin_time_unit")
    concentration_unit = unit_col2.selectbox("Unidad de concentración", CONCENTRATION_UNITS, index=0, key="kin_conc_unit")
    st.caption("Ingrese los valores numéricos usando exactamente las unidades seleccionadas. La aplicación conserva esas unidades en los resultados.")
    mode = st.radio("Método de ingreso", ["Manual", "Cargar archivo"], horizontal=True)
    if mode == "Manual":
        default = pd.DataFrame({"Tiempo": [0,10,20,30,40,50], "Concentración": [100,95,90,85,80,75]})
        edited = st.data_editor(
            default, num_rows="dynamic", use_container_width=True,
            column_config={
                "Tiempo": st.column_config.NumberColumn("Tiempo", format="%.6f", step=0.01),
                "Concentración": st.column_config.NumberColumn("Concentración", format="%.6f", step=0.01)
            }
        )
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
            st.session_state["time_unit"] = time_unit
            st.session_state["concentration_unit"] = concentration_unit
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
                         "R² transformado":r["r2"], "k":r["k"],
                         "Unidad de k":kinetic_constant_unit(name, st.session_state["concentration_unit"], st.session_state["time_unit"]),
                         "RMSE concentración":r["rmse_original"], "k positiva":r["valid_k"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info("Candidato por R²: " + result["_summary"]["best_by_r2"] + ". Contraste por menor RMSE: " + result["_summary"]["best_by_rmse_linearized"] + ".")
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
                if i == 0:
                    yname = f"Concentración ({st.session_state['concentration_unit']})"
                elif i == 1:
                    yname = f"ln(Concentración en {st.session_state['concentration_unit']})"
                else:
                    yname = f"1/Concentración (1/{st.session_state['concentration_unit']})"
                fig.update_layout(xaxis_title=f"Tiempo ({st.session_state['time_unit']})", yaxis_title=yname, template="plotly_white")
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
            col.metric(key, f"{val:.4g} {st.session_state['time_unit']}")
        try:
            nl = nonlinear_refit(data.time, data.concentration, rr["order"])
            nl_k_unit = kinetic_constant_unit(chosen, st.session_state["concentration_unit"], st.session_state["time_unit"])
            st.write(f"Ajuste no lineal en concentración original: C₀ = {nl['c0']:.6g} {st.session_state['concentration_unit']}; k = {nl['k']:.6g} {nl_k_unit}; R² original = {nl['r2_original']:.6f}; RMSE = {nl['rmse']:.6g} {st.session_state['concentration_unit']}; AIC = {nl['aic']:.4f}; BIC = {nl['bic']:.4f}.")
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
    temp_unit = st.selectbox("Unidad de temperatura", TEMPERATURE_UNITS, index=0, key="arr_temp_unit")
    arr_order = st.selectbox("Orden cinético al que corresponden las constantes k", ["Orden cero", "Orden uno", "Orden dos"])
    arr_time_unit = st.selectbox("Unidad de tiempo utilizada para k", TIME_UNITS, index=3, key="arr_time_unit")
    arr_conc_unit = st.selectbox("Unidad de concentración utilizada para k", CONCENTRATION_UNITS, index=0, key="arr_conc_unit")
    arr_k_unit = kinetic_constant_unit(arr_order, arr_conc_unit, arr_time_unit)
    st.caption(f"Las constantes k deben corresponder al mismo orden cinético y expresarse en {arr_k_unit} para todas las temperaturas.")
    default_a = pd.DataFrame({"Temperatura":[40,50,60,70], "k":[0.002,0.004,0.008,0.015]})
    arr_df = st.data_editor(
        default_a, num_rows="dynamic", use_container_width=True,
        column_config={
            "Temperatura": st.column_config.NumberColumn(f"Temperatura ({temp_unit})", format="%.4f", step=0.1),
            "k": st.column_config.NumberColumn("k", format="%.8f", step=0.0001)
        }
    )
    if st.button("Ajustar Arrhenius", type="primary"):
        try:
            temp_c = temperature_to_celsius(arr_df["Temperatura"], temp_unit)
            ar = analyze_arrhenius(temp_c, arr_df["k"])
            ar["temperature_input"] = np.asarray(arr_df["Temperatura"], dtype=float)
            ar["temperature_unit"] = temp_unit
            ar["k_unit_label"] = arr_k_unit
            ar["kinetic_order"] = arr_order
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
        arr_table = pd.DataFrame({f"T ({ar['temperature_unit']})":ar["temperature_input"], "T (°C)":ar["temperature_c"], "T (K)":ar["temperature_k"], "1/T (K⁻¹)":ar["inv_temperature"], f"k ({ar['k_unit_label']})":ar["k"], "ln(k)":ar["ln_k"]})
        st.dataframe(arr_table, use_container_width=True)
        fig = px.scatter(arr_table, x="1/T (K⁻¹)", y="ln(k)", trendline="ols", title="Gráfica de Arrhenius")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Pendiente = {ar['slope']:.6g}; intercepto = {ar['intercept']:.6g}")
        target_input = st.number_input(f"Temperatura objetivo ({ar['temperature_unit']})", value=celsius_to_unit(25.0, ar["temperature_unit"]), step=0.1, format="%.4f")
        target_c = float(temperature_to_celsius([target_input], ar["temperature_unit"])[0])
        kp = float(predict_k(ar, [target_c])[0])
        st.metric(f"k estimada a {target_input:.2f} {ar['temperature_unit']}", f"{kp:.6g} {ar['k_unit_label']}")
        st.session_state["arrhenius"]["predicted_k"] = kp
        st.session_state["arrhenius"]["target_temperature"] = target

        if "kinetics" in st.session_state:
            kin = st.session_state["kinetics"]
            default_order = kin["_summary"]["best_by_r2"]
            order_name = st.selectbox("Orden cinético para proyectar", ["Orden cero","Orden uno","Orden dos"],
                                      index=["Orden cero","Orden uno","Orden dos"].index(default_order))
            kr = kin[order_name]
            c0 = st.number_input("Concentración inicial C₀", min_value=0.000001, value=float(kr["c0_fit"]), step=0.01, format="%.6f")
            horizon = st.number_input("Horizonte de predicción", min_value=1.0, value=365.0, step=0.1, format="%.4f")
            unit = st.session_state.get("time_unit", arr_time_unit)
            if unit != arr_time_unit:
                st.warning("La unidad de tiempo de la predicción debe coincidir con la utilizada para calcular k. Se utilizará la unidad declarada para k: " + arr_time_unit + ".")
                unit = arr_time_unit
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
    - Unidades admitidas: tiempo en segundos, minutos, horas, días, semanas, meses o años; concentración en mg/L, µg/L, g/L, mg/mL, µg/mL, g/mL, mol/L, mmol/L, µmol/L, % m/v, % m/m o ppm; temperatura en °C, K o °F.
    - Las unidades de tiempo y concentración seleccionadas se conservan durante los cálculos. Las temperaturas se convierten internamente a kelvin para el análisis de Arrhenius.

    **Criterio de selección:** el mayor R² de las regresiones linealizadas se muestra como candidato solicitado. Debido a que cada transformación cambia la escala de los errores, también se informa RMSE en concentración original y se ofrece ajuste no lineal.

    **Limitaciones:** el modelo supone un mecanismo cinético constante, condiciones experimentales comparables y una relación de Arrhenius válida en el intervalo considerado. No sustituye estudios de estabilidad formales, evaluación de especificidad analítica ni criterios regulatorios.
    """)

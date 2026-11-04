import pandas as pd

def load_experimental_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Formato no admitido. Use CSV o Excel.")
    return df

def clean_time_concentration(df, time_col, concentration_col):
    out = df[[time_col, concentration_col]].copy()
    out.columns = ["time", "concentration"]
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["concentration"] = pd.to_numeric(out["concentration"], errors="coerce")
    out = out.dropna().sort_values("time")
    if out.empty:
        raise ValueError("No hay filas numéricas válidas.")
    return out.reset_index(drop=True)

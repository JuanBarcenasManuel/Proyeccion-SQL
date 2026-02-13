import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import os

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Zudalpro | Proyección de Demanda", layout="wide")

# Parámetros de SQL
SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020"
DB_PRINCIPAL = "EnterpriseAdminDb"
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'image_ce2f2b.png')

# --- 2. FUNCIONES DE CARGA ---
@st.cache_resource
def load_model():
    ruta = os.path.join(ruta_base, 'modelo_ventas.json')
    if os.path.exists(ruta):
        model = xgb.XGBRegressor()
        model.load_model(ruta)
        return model
    return None

def create_features(df):
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes'] = df.index.day
    df['mes'] = df.index.month
    df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)
    for lag in [1, 2, 7, 14]:
        df[f'lag_{lag}'] = df['MontoNeto'].shift(lag)
    df['rolling_mean_7'] = df['MontoNeto'].shift(1).rolling(window=7).mean()
    return df

# --- 3. INTERFAZ Y LOGO ---
col_logo, col_tit = st.columns([1, 4])
with col_logo:
    if os.path.exists(ruta_logo): st.image(ruta_logo, width=180)
with col_tit:
    st.title("Sistema de Proyección Suministros")
    st.write("Planificación de demanda avanzada con XGBoost")

# --- 4. SELECCIÓN DE ORIGEN DE DATOS ---
st.sidebar.header("⚙️ Origen de Datos")
origen = st.sidebar.radio("Seleccionar Fuente:", ["SQL Server (Local/VPN)", "Subir Excel/CSV (Nube)"])

pw_clean = None

if origen == "SQL Server (Local/VPN)":
    try:
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DB_PRINCIPAL};UID={UID};PWD={PWD};Encrypt=no;TrustServerCertificate=yes;Connection Timeout=5;"
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")
        SQL = text("SELECT FechaE, TRY_CONVERT(float, Cantidad) * TRY_CONVERT(float, Precio) AS MontoNeto FROM SAITEMFAC WHERE TipoFac = 'A' AND FechaE >= '2025-01-01'")
        with engine.connect() as conn:
            df_sql = pd.read_sql(SQL, conn, parse_dates=["FechaE"])
        df_sql["FechaE"] = pd.to_datetime(df_sql["FechaE"]).dt.normalize()
        pw_clean = df_sql.groupby("FechaE")["MontoNeto"].sum().resample('D').asfreq().fillna(0)
        st.sidebar.success("✅ Conectado a SQL Server")
    except:
        st.sidebar.error("❌ No se pudo conectar a la IP 192.168.150.6. Use la opción de subir archivo.")

else:
    file = st.sidebar.file_uploader("Suba el historial de ventas", type=['xlsx', 'csv'])
    if file:
        df_up = pd.read_excel(file) if file.name.endswith('xlsx') else pd.read_csv(file)
        # Asegúrate de que las columnas se llamen FechaE y MontoNeto
        df_up.columns = ['FechaE', 'MontoNeto']
        df_up["FechaE"] = pd.to_datetime(df_up["FechaE"]).dt.normalize()
        pw_clean = df_up.set_index("FechaE").resample('D').asfreq().fillna(0)
        st.sidebar.success("✅ Archivo cargado")

# --- 5. CÁLCULO DE PROYECCIÓN ---
fecha_inicio = st.sidebar.date_input("Fecha Inicio Proyección:", datetime.now())
model = load_model()

if st.sidebar.button("🚀 Calcular Proyección") and pw_clean is not None and model is not None:
    df_loop = pw_clean.copy()
    results = []
    current_date = pd.Timestamp(fecha_inicio)
    
    for _ in range(30):
        df_loop.loc[current_date, 'MontoNeto'] = 0
        feats = create_features(df_loop)
        X = feats.loc[[current_date], ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']]
        pred = max(0, float(model.predict(X)[0]))
        df_loop.loc[current_date, 'MontoNeto'] = pred
        results.append({'Fecha': current_date, 'Venta': pred})
        current_date += timedelta(days=1)
    
    df_res = pd.DataFrame(results)
    st.metric("💰 PROYECCIÓN PRÓXIMOS 30 DÍAS", f"${df_res['Venta'].sum():,.2f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res['Fecha'], y=df_res['Venta'], mode='lines+markers', name='Proyectado', line=dict(color='orange')))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_res.style.format({'Venta': '${:,.2f}'}), use_container_width=True)

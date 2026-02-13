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
st.set_page_config(page_title="Zudalpro | Proyección", layout="wide", page_icon="📈")

SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020"
DB_PRINCIPAL = "EnterpriseAdminDb"
ruta_base = os.path.dirname(__file__)

# --- 2. FUNCIONES ---
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

# --- 3. LOGO Y TÍTULO ---
st.title("🚀 Sistema de Proyección Suministros 1979 C.A.")
st.markdown("---")

# --- 4. LÓGICA DE DATOS (AUTOMÁTICA O MANUAL) ---
pw_clean = None

# Intento de conexión automática (Solo funcionará en la red local)
try:
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DB_PRINCIPAL};UID={UID};PWD={PWD};Encrypt=no;TrustServerCertificate=yes;Connection Timeout=2;"
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")
    SQL = text("SELECT FechaE, TRY_CONVERT(float, Cantidad) * TRY_CONVERT(float, Precio) AS MontoNeto FROM SAITEMFAC WHERE TipoFac = 'A' AND FechaE >= '2025-01-01'")
    with engine.connect() as conn:
        df_sql = pd.read_sql(SQL, conn, parse_dates=["FechaE"])
    df_sql["FechaE"] = pd.to_datetime(df_sql["FechaE"]).dt.normalize()
    pw_clean = df_sql.groupby("FechaE")["MontoNeto"].sum().resample('D').asfreq().fillna(0)
    st.sidebar.success("✅ Conectado automáticamente al SQL Server")
except:
    st.sidebar.warning("⚠️ No se detectó la red local. Cargue un archivo para continuar.")
    archivo = st.sidebar.file_uploader("Subir historial (Excel/CSV):", type=['xlsx', 'csv'])
    if archivo:
        df_up = pd.read_excel(archivo) if archivo.name.endswith('xlsx') else pd.read_csv(archivo)
        df_up.columns = ['FechaE', 'MontoNeto']
        df_up["FechaE"] = pd.to_datetime(df_up["FechaE"]).dt.normalize()
        pw_clean = df_up.set_index("FechaE").resample('D').asfreq().fillna(0)

# --- 5. EJECUCIÓN DE LA PROYECCIÓN ---
if pw_clean is not None:
    fecha_ini = st.sidebar.date_input("Fecha Inicio Proyección:", datetime.now())
    if st.sidebar.button("Calcular Proyección de 30 Días"):
        model = load_model()
        if model:
            df_loop = pw_clean.copy()
            results = []
            curr = pd.Timestamp(fecha_ini)
            
            with st.spinner("Generando pronóstico..."):
                for _ in range(30):
                    df_loop.loc[curr, 'MontoNeto'] = 0
                    feats = create_features(df_loop)
                    X = feats.loc[[curr], ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']]
                    pred = max(0, float(model.predict(X)[0]))
                    df_loop.loc[curr, 'MontoNeto'] = pred
                    results.append({'Fecha': curr, 'Venta Proyectada': pred})
                    curr += timedelta(days=1)
            
            df_res = pd.DataFrame(results)
            
            # Métricas
            c1, c2 = st.columns(2)
            c1.metric("💰 TOTAL PROYECTADO", f"${df_res['Venta Proyectada'].sum():,.2f}")
            c2.metric("📊 PROMEDIO DIARIO", f"${df_res['Venta Proyectada'].mean():,.2f}")
            
            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_res['Fecha'], y=df_res['Venta Proyectada'], mode='lines+markers', name='Proyección', line=dict(color='#FF4B4B')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla
            st.dataframe(df_res.style.format({'Venta Proyectada': '${:,.2f}'}), use_container_width=True)
        else:
            st.error("No se encontró el archivo del modelo (.json)")
else:
    st.info("Esperando datos... Si estás en la oficina, la conexión es automática. Si estás en la nube, sube un Excel.")

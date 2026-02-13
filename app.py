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
st.set_page_config(page_title="Predicción Ventas | Suministros 1979 C.A", layout="wide", page_icon="🛒")

# Estilo para tarjetas
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONEXIÓN Y DATOS ---
SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020"
DB_PRINCIPAL = "EnterpriseAdminDb"

@st.cache_data
def get_historical_data():
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DB_PRINCIPAL};UID={UID};PWD={PWD};Encrypt=no;TrustServerCertificate=yes;"
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")
    SQL = text(f"SELECT FechaE, TRY_CONVERT(float, Cantidad) * TRY_CONVERT(float, Precio) AS MontoNeto FROM [SAITEMFAC] WHERE TipoFac = 'A' AND FechaE >= '2025-01-01'")
    with engine.connect() as conn:
        df = pd.read_sql(SQL, conn, parse_dates=["FechaE"])
    df["FechaE"] = pd.to_datetime(df["FechaE"]).dt.normalize()
    return df.groupby("FechaE")["MontoNeto"].sum().resample('D').asfreq().fillna(0)

@st.cache_resource
def load_model():
    ruta = os.path.join(os.path.dirname(__file__), 'modelo_ventas.json')
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

# --- 3. CABECERA ---
ruta_logo = os.path.join(os.path.dirname(__file__), 'Suministros.jpg')
c_logo, c_title = st.columns([1, 4])
with c_logo:
    if os.path.exists(ruta_logo): st.image(ruta_logo, width=150)
with c_title:
    st.title("Sistema de Proyección de Demanda")
    st.write("Análisis dinámico basado en historial real de SQL Server")

# --- 4. BARRA LATERAL (DINÁMICA) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_inicio = st.date_input("Proyectar 30 días desde:", datetime.now())
    st.divider()
    if st.button("🔄 Sincronizar y Calcular", use_container_width=True):
        st.session_state.run_proy = True

# --- 5. LÓGICA PRINCIPAL ---
pw_clean = get_historical_data()
model = load_model()

if 'run_proy' in st.session_state and model is not None:
    # Preparar datos base para el bucle
    df_loop = pw_clean[pw_clean.index < pd.Timestamp(fecha_inicio)].copy()
    
    # Si la fecha elegida es hoy, necesitamos asegurar que tenemos datos hasta ayer
    results = []
    features_cols = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']
    
    current_date = pd.Timestamp(fecha_inicio)
    
    for _ in range(30):
        # Crear fila temporal para este día
        df_loop.loc[current_date, 'MontoNeto'] = 0
        feats = create_features(df_loop)
        
        # Predicción
        pred = model.predict(feats.loc[[current_date], features_cols])[0]
        pred = max(0, pred)
        
        # Guardar y actualizar para el siguiente paso del bucle
        df_loop.loc[current_date, 'MontoNeto'] = pred
        results.append({'Fecha': current_date.strftime('%Y-%m-%d'), 'Venta Proyectada': pred})
        current_date += timedelta(days=1)

    df_res = pd.DataFrame(results)
    total_30d = df_res['Venta Proyectada'].sum()

    # --- 6. VISUALIZACIÓN ---
    m1, m2 = st.columns(2)
    with m1:
        st.metric(" TOTAL PROYECTADO (30 DÍAS)", f"${total_30d:,.2f}")
    with m2:
        st.metric("📅 FECHA DE INICIO", fecha_inicio.strftime('%d/%m/%Y'))

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res['Fecha'], y=df_res['Venta Proyectada'], mode='lines+markers', name='Proyección', line=dict(color='red', width=3)))
    fig.update_layout(template="plotly_white", title="Tendencia Diaria Proyectada")
    st.plotly_chart(fig, use_container_width=True)

    # TABLA DINÁMICA DE VENTAS DIARIAS
    st.subheader("📋 Detalle de Ventas Diarias Proyectadas")
    st.dataframe(
        df_res.style.format({'Venta Proyectada': '${:,.2f}'}),
        use_container_width=True,
        height=400
    )

elif model is None:
    st.error("No se encontró 'modelo_ventas.json'.")
else:
    st.info("Seleccione una fecha en el calendario y presione 'Sincronizar y Calcular'.")

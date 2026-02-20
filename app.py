import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Predicción Ventas | Suministros 1979 C.A.",
    layout="wide", 
    page_icon="🛒"
)

# Estilo visual para las tarjetas de métricas
st.markdown("""
    <style>
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. RUTAS DE ARCHIVOS ---
# Esto permite que la app encuentre los archivos tanto en tu PC como en la nube
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg') # Ajustado al nombre en tu carpeta
ruta_datos = os.path.join(ruta_base, 'ventas_historicas.csv')
ruta_modelo = os.path.join(ruta_base, 'modelo_ventas.json')

@st.cache_data
def get_historical_data():
    """Carga los datos desde el CSV generado por el script extractor"""
    if os.path.exists(ruta_datos):
        df = pd.read_csv(ruta_datos)
        # Convertimos la columna de fecha y la ponemos como índice
        df["FechaE"] = pd.to_datetime(df["FechaE"])
        df = df.set_index("FechaE")
        # Aseguramos frecuencia diaria y llenamos huecos con 0
        df = df.resample('D').asfreq().fillna(0)
        return df
    return None

@st.cache_resource
def load_model():
    """Carga el modelo XGBoost entrenado"""
    if os.path.exists(ruta_modelo):
        model = xgb.XGBRegressor()
        model.load_model(ruta_modelo)
        return model
    return None

def create_features(df):
    """Genera las mismas variables usadas durante el entrenamiento del modelo"""
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['es_finde']   = df['dia_semana'].isin([5, 6]).astype(int)
    
    # Lags (valores pasados)
    for lag in [1, 2, 7, 14]:
        df[f'lag_{lag}'] = df['MontoNeto'].shift(lag)
    
    # Media móvil de la última semana
    df['rolling_mean_7'] = df['MontoNeto'].shift(1).rolling(window=7).mean()
    return df

# --- 3. ENCABEZADO ---
col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, width=150)
    else:
        st.subheader("📦 Suministros 1979")

with col_titulo:
    st.title("Sistema de Proyección de Demanda")
    st.write("Predicción Ventas | Departamento de Cadenas de Suministros")

st.markdown("---")

# --- 4. CARGA DE DATOS Y MODELO ---
pw_clean = get_historical_data()
model = load_model()

# --- 5. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_inicio = st.date_input("Proyectar 30 días desde:", datetime.now())
    
    st.divider()
    
    if pw_clean is not None:
        ultima_fecha = pw_clean.index.max().strftime('%d/%m/%Y')
        st.success(f"✅ Datos cargados hasta: {ultima_fecha}")
    else:
        st.error("❌ No se encontró 'ventas_historicas.csv'")
    
    btn_calcular = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- 6. LÓGICA DE PROYECCIÓN (RECURSIVA) ---
if btn_calcular:
    if model is not None and pw_clean is not None:
        with st.spinner("Generando proyección para los próximos 30 días..."):

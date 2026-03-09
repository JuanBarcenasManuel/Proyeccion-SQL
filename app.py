import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image

# --- 1. CONFIGURACIÓN DE PÁGINA ---
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg')

try:
    logo = Image.open(ruta_logo)
    p_icon = logo
except:
    p_icon = "🛒"

st.set_page_config(
    page_title="Predicción Ventas | Suministros 1979 C.A.",
    layout="wide", 
    page_icon=p_icon
)

# Estilo visual
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

# --- 2. RUTAS Y CARGA DE DATOS ---
ruta_datos = os.path.join(ruta_base, 'ventas_historicas.csv')

@st.cache_data
def get_historical_data():
    if os.path.exists(ruta_datos):
        df = pd.read_csv(ruta_datos)
        df["FechaE"] = pd.to_datetime(df["FechaE"])
        df = df.groupby("FechaE")["MontoNeto"].sum().to_frame()
        df = df.resample('D').asfreq().fillna(0)
        return df
    return None

def create_features(df):
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['es_finde']   = df['dia_semana'].isin([5, 6]).astype(int)
    # Variable de tendencia para captar crecimiento
    df['tendencia']  = np.arange(len(df))
    
    # Lags expandidos para captar ciclos mensuales
    for lag in [1, 2, 7, 14, 30]:
        df[f'lag_{lag}'] = df['MontoNeto'].shift(lag)
    
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
    st.write("Predicción Ventas | Inteligencia de Datos")

st.markdown("---")

# --- 4. BARRA LATERAL ---
pw_clean = get_historical_data()

with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_inicio_proy = st.date_input("Proyectar 30 días desde:", datetime.now())
    
    if pw_clean is not None:
        ultima_fecha_data = pw_clean.index.max()
        st.success(f"✅ Datos hasta: {ultima_fecha_data.strftime('%d/%m/%Y')}")
    else:
        st.error("❌ No se encontró 'ventas_historicas.csv'")
    
    btn_calcular = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- 5. LÓGICA DE PROYECCIÓN ---
if btn_calcular and pw_clean is not None:
    with st.spinner("Entrenando modelo con datos recientes..."):
        
        # Preparar datos y entrenar (esto asegura que Marzo aprenda de Febrero)
        df_train = create_features(pw_clean).dropna()
        features_cols = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'tendencia', 
                         'lag_1', 'lag_2', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7']
        
        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
        model.fit(df_train[features_cols], df_train['MontoNeto'])

        # Generar Proyección Iterativa
        df_loop = pw_clean.copy()
        results = []
        current_date = pd.Timestamp(fecha_inicio_proy)
        
        for _ in range(30):
            # Asegurar que la fecha actual esté en el índice para create_features
            df_loop.loc[current_date, 'MontoNeto'] = 0 
            df_with_features = create_features(df_loop)
            X_input = df_with_features.loc[[current_date], features_cols]
            
            pred = model.predict(X_input)[0]
            pred = max(0, float(pred))
            
            df_loop.loc[current_date, 'MontoNeto'] = pred
            results.append({'Fecha': current_date, 'Venta Proyectada': pred})
            current_date += timedelta(days=1)
            
        df_res = pd.DataFrame(results).set_index('Fecha')

        # --- MÉTRICAS ---
        total_proy = df_res['Venta Proyectada'].sum()
        promedio_diario = df_res['Venta Proyectada'].mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📈 TOTAL PROYECTADO", f"${total_proy:,.2f}")
        m2.metric("📊 PROM. DIARIO", f"${promedio_diario:,.2f}")
        m3.metric("📅 PERIODO", "30 Días")
        m4.metric("🏢 ESTADO", "Tiendas OK")

        # Gráfico
        fig = go.Figure()
        # Mostrar últimos 30 días reales para comparar
        df_real_last = pw_clean.tail(30)
        fig.add_trace(go.Scatter(x=df_real_last.index, y=df_real_last['MontoNeto'], name="Real (Febrero)", line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Venta Proyectada'], name="Proyección (Marzo)", line=dict(color='#FF4B4B', width=3), fill='tozeroy'))
        
        fig.update_layout(template="plotly_white", title="Comparativa Ventas Reales vs Proyección")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Detalle Diario")
        st.dataframe(df_res.style.format('${:,.2f}'), use_container_width=True)

st.divider()
st.caption(f"© {datetime.now().year} | Suministros 1979 C.A. | Usuario: JBARCENAS")

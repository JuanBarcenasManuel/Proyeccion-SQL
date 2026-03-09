import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image

# --- 1. CONFIGURACIÓN INICIAL (DEBE SER LO PRIMERO) ---
st.set_page_config(
    page_title="Predicción Ventas | Suministros 1979",
    layout="wide"
)

# Definición de rutas absoluta
base_path = os.path.dirname(__file__)
logo_path = os.path.join(base_path, 'Suministros.jpg')
data_path = os.path.join(base_path, 'ventas_historicas.csv')

# --- 2. FUNCIONES DE CARGA Y PROCESAMIENTO ---
@st.cache_data
def load_data():
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df["FechaE"] = pd.to_datetime(df["FechaE"])
        # Agrupamos por día y sumamos el monto neto
        df = df.groupby("FechaE")["MontoNeto"].sum().reset_index()
        df = df.set_index("FechaE").resample('D').asfreq().fillna(0)
        return df
    return None

def create_features(df):
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes'] = df.index.day
    df['mes'] = df.index.month
    df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)
    # Lags clave para evitar líneas planas
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df['MontoNeto'].shift(lag)
    df['rolling_mean_7'] = df['MontoNeto'].shift(1).rolling(window=7).mean()
    return df

# --- 3. INTERFAZ DE USUARIO ---
# Manejo seguro del logo
c1, c2 = st.columns([1, 4])
with c1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    else:
        st.write("📦 **Suministros 1979**")

with c2:
    st.title("Sistema de Proyección de Demanda")
    st.write("Inteligencia de Datos | Suministros 1979 C.A.")

st.markdown("---")

df_hist = load_data()

with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_proy = st.date_input("Proyectar desde:", datetime.now())
    st.divider()
    if df_hist is not None:
        st.success(f"Datos hasta: {df_hist.index.max().strftime('%d/%m/%Y')}")
    else:
        st.error("Archivo ventas_historicas.csv no detectado.")
    btn = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- 4. LÓGICA DE CÁLCULO ---
if btn and df_hist is not None:
    with st.spinner("Entrenando modelo XGBoost..."):
        # Entrenamiento sin sesgo de días cerrados
        df_train = create_features(df_hist[df_hist['MontoNeto'] > 0]).dropna()
        features = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7']
        
        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=7, random_state=42)
        model.fit(df_train[features], df_train['MontoNeto'])

        # Bucle de proyección a 30 días
        df_work = df_hist.copy()
        results = []
        curr_date = pd.Timestamp(fecha_proy)
        
        # Piso operativo basado en tu requerimiento de $8M/mes
        min_diario = 266666 

        for _ in range(30):
            df_work.loc[curr_date, 'MontoNeto'] = 0
            feats = create_features(df_work).loc[[curr_date], features]
            pred = float(model.predict(feats)[0])
            
            # Ajuste dinámico para no caer en líneas planas
            pred = max(min_diario, pred)
            
            df_work.loc[curr_date, 'MontoNeto'] = pred
            results.append({'Fecha': curr_date, 'Venta Proyectada': pred})
            curr_date += timedelta(days=1)

        df_res = pd.DataFrame(results).set_index('Fecha')

        # Métricas
        m1, m2, m3 = st.columns(3)
        m1.metric("💰 TOTAL PROYECTADO", f"${df_res['Venta Proyectada'].sum():,.2f}")
        m2.metric("📊 PROM. DIARIO", f"${df_res['Venta Proyectada'].mean():,.2f}")
        m3.metric("📅 PERIODO", "30 Días")

        # Gráfico Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist.tail(15).index, y=df_hist.tail(15)['MontoNeto'], name="Real", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Venta Proyectada'], name="Proyección", line=dict(color="#FF4B4B", width=3), fill='tozeroy'))
        fig.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Detalle de Venta")
        st.dataframe(df_res.style.format('${:,.2f}'), use_container_width=True)

st.divider()
st.caption(f"© {datetime.now().year} | Suministros 1979 C.A. | Juan Barcenas")

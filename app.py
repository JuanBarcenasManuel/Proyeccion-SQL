import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image

# --- 1. CONFIGURACIÓN Y RUTAS ---
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg')
ruta_datos = os.path.join(ruta_base, 'ventas_historicas.csv')

try:
    logo_img = Image.open(ruta_logo)
except:
    logo_img = "🛒"

st.set_page_config(page_title="Predicción Ventas | Suministros 1979", layout="wide", page_icon=logo_img)

# Estilo para tarjetas
st.markdown("""<style>.stMetric {background-color: #fff; padding: 20px; border-radius: 12px; border: 1px solid #eee;}</style>""", unsafe_allow_html=True)

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
    # Lags clave
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df['MontoNeto'].shift(lag)
    df['rolling_mean_7'] = df['MontoNeto'].shift(1).rolling(window=7).mean()
    return df

# --- 2. ENCABEZADO ---
c1, c2 = st.columns([1, 4])
with c1: st.image(ruta_logo, width=150) if os.path.exists(ruta_logo) else st.write("📦")
with c2:
    st.title("Sistema de Proyección de Demanda")
    st.write("Suministros 1979 C.A. | Inteligencia de Ventas")

st.markdown("---")

# --- 3. PROCESAMIENTO ---
pw_clean = get_historical_data()

with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_inicio_proy = st.date_input("Proyectar desde:", datetime.now())
    st.divider()
    if pw_clean is not None:
        st.success(f"✅ Datos hasta: {pw_clean.index.max().strftime('%d/%m/%Y')}")
    btn_calcular = st.button("🚀 Calcular Proyección", use_container_width=True)

if btn_calcular and pw_clean is not None:
    with st.spinner("Analizando ciclos de venta..."):
        # Entrenar con días que tienen venta real para no "aprender" de los cierres
        df_train = create_features(pw_clean[pw_clean['MontoNeto'] > 0]).dropna()
        
        features = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7']
        
        model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, subsample=0.8, random_state=42)
        model.fit(df_train[features], df_train['MontoNeto'])

        # Proyección
        df_loop = pw_clean.copy()
        results = []
        curr_date = pd.Timestamp(fecha_inicio_proy)
        
        # Objetivo: $8M mensual -> ~$267k promedio diario
        for _ in range(30):
            df_loop.loc[curr_date, 'MontoNeto'] = 0
            feats_today = create_features(df_loop).loc[[curr_date], features]
            pred = max(0, float(model.predict(feats_today)[0]))
            
            # Si la predicción cae mucho por ruido estadístico, aplicamos un suavizado
            if pred < 150000 and curr_date.dayofweek < 5: # Si es día de semana y predice muy poco
                hist_mean = df_train['MontoNeto'].tail(14).mean()
                pred = hist_mean * 0.8 # Usamos el 80% del histórico reciente como piso
                
            df_loop.loc[curr_date, 'MontoNeto'] = pred
            results.append({'Fecha': curr_date, 'Venta Proyectada': pred})
            curr_date += timedelta(days=1)

        df_res = pd.DataFrame(results).set_index('Fecha')
        
        # Métricas
        t_proy = df_res['Venta Proyectada'].sum()
        p_real = pw_clean[pw_clean.index >= pd.to_datetime(fecha_inicio_proy)]['MontoNeto'].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📈 VENTA ACUM.", f"${p_real:,.2f}")
        m2.metric("💰 TOTAL PROY.", f"${t_proy:,.2f}")
        m3.metric("📅 PERIODO", "30 Días")
        m4.metric("📊 PROM. DIARIO", f"${df_res['Venta Proyectada'].mean():,.2f}")

        # Gráfica
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pw_clean.tail(20).index, y=pw_clean.tail(20)['MontoNeto'], name="Real", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Venta Proyectada'], name="Proyección", line=dict(color="#FF4B4B", width=3), fill='tozeroy'))
        fig.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Detalle Diario")
        st.dataframe(df_res.style.format('${:,.2f}'), use_container_width=True)

st.divider()
st.caption(f"© {datetime.now().year} | Suministros 1979 C.A. | Usuario: JBARCENAS")

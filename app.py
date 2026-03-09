import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image

# --- 1. CONFIGURACIÓN DE PÁGINA (DEBE SER LO PRIMERO) ---
st.set_page_config(
    page_title="Predicción Ventas | Suministros 1979",
    layout="wide"
)

# Definición de rutas relativas para GitHub/Streamlit Cloud
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg')
ruta_datos = os.path.join(ruta_base, 'ventas_historicas.csv')

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

# --- 2. FUNCIONES DE PROCESAMIENTO ---
@st.cache_data
def get_historical_data():
    if os.path.exists(ruta_datos):
        df = pd.read_csv(ruta_datos)
        df["FechaE"] = pd.to_datetime(df["FechaE"])
        # Agrupamos por día para consolidar la venta total de la empresa
        df = df.groupby("FechaE")["MontoNeto"].sum().reset_index()
        df = df.set_index("FechaE").sort_index()
        # Rellenamos días sin venta con 0 para mantener la cronología
        df = df.resample('D').asfreq().fillna(0)
        return df
    return None

def create_features(df):
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['es_finde']   = df['dia_semana'].isin([5, 6]).astype(int)
    # Lags clave para detectar ciclos semanales y mensuales
    for lag in [1, 7, 30]:
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
    st.write("Suministros 1979 C.A. | Planificación y Presupuesto")

st.markdown("---")

# --- 4. CARGA DE DATOS Y SIDEBAR ---
pw_clean = get_historical_data()

with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_inicio_proy = st.date_input("Proyectar desde:", datetime.now())
    
    st.divider()
    st.write("**Ajuste de Conservadurismo**")
    # Este slider permite bajar ese total de 9.4M a algo más realista
    ajuste_sensibilidad = st.slider("Sensibilidad del Modelo (%)", 60, 100, 85) / 100
    
    st.divider()
    if pw_clean is not None:
        ultima_fecha = pw_clean.index.max()
        st.success(f"✅ Datos cargados hasta: {ultima_fecha.strftime('%d/%m/%Y')}")
    
    btn_calcular = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- 5. LÓGICA DE PROYECCIÓN ---
if btn_calcular and pw_clean is not None:
    with st.spinner("Entrenando modelo y suavizando picos..."):
        
        # Entrenamos con días de venta real para evitar sesgo de cierre
        df_train = create_features(pw_clean[pw_clean['MontoNeto'] > 0]).dropna()
        features = ['dia_semana', 'dia_mes', 'es_finde', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7']
        
        # XGBoost con profundidad limitada para evitar sobreajuste (overfitting)
        model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=5, subsample=0.8)
        model.fit(df_train[features], df_train['MontoNeto'])

        # Techo de venta diario (Percentil 90) para que un pico no infle el mes
        techo_diario = pw_clean['MontoNeto'][pw_clean['MontoNeto'] > 0].quantile(0.90)

        df_loop = pw_clean.copy()
        results = []
        current_date = pd.Timestamp(fecha_inicio_proy)
        
        for _ in range(30):
            df_loop.loc[current_date, 'MontoNeto'] = 0
            df_with_features = create_features(df_loop)
            X_input = df_with_features.loc[[current_date], features]
            
            pred = float(model.predict(X_input)[0])
            
            # Aplicación de frenos y ajustes
            pred = pred * ajuste_sensibilidad
            pred = min(pred, techo_diario)
            pred = max(0, pred)
            
            df_loop.loc[current_date, 'MontoNeto'] = pred
            results.append({
                'Fecha': current_date.strftime('%Y-%m-%d'),
                'Venta Proyectada': pred
            })
            current_date += timedelta(days=1)
        
        df_res = pd.DataFrame(results)
        total_proyectado = df_res['Venta Proyectada'].sum()

        # --- 6. VISUALIZACIÓN ---
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("💰 TOTAL PROYECTADO", f"${total_proyectado:,.2f}")
        with m2: st.metric("📊 PROM. DIARIO", f"${df_res['Venta Proyectada'].mean():,.2f}")
        with m3: st.metric("📅 PERIODO", "30 Días")
        with m4: st.metric("🛑 TOPE DIARIO", f"${techo_diario:,.2f}")

        # Gráfico de tendencia
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_res['Fecha'], y=df_res['Venta Proyectada'],
            mode='lines+markers', line=dict(color='#FF4B4B', width=3),
            fill='tozeroy', fillcolor='rgba(255, 75, 75, 0.1)', name="Proyección"
        ))
        fig.update_layout(
            template="plotly_white", 
            xaxis_title="Marzo 2026", 
            yaxis_title="Monto Neto ($)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Detalle Diario")
        st.dataframe(df_res.style.format({'Venta Proyectada': '${:,.2f}'}), use_container_width=True)

elif btn_calcular:
    st.error("No se pudo cargar el archivo ventas_historicas.csv")

st.divider()
st.caption(f"© {datetime.now().year} | Suministros 1979 C.A. | Usuario: JBARCENAS")

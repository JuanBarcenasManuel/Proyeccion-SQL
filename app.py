import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from PIL import Image

# --- 1. CONFIGURACIÓN Y LOGO ---
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg')

try:
    logo = Image.open(ruta_logo)
except:
    logo = "📦"

st.set_page_config(
    page_title="Predicción Ventas | Suministros 1979 C.A.",
    layout="wide", 
    page_icon=logo
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
ruta_datos = os.path.join(ruta_base, 'ventas_historicas.csv')

@st.cache_data
def get_historical_data():
    if os.path.exists(ruta_datos):
        df = pd.read_csv(ruta_datos)
        df["FechaE"] = pd.to_datetime(df["FechaE"])
        # Agrupamos por día para asegurar venta total diaria
        df = df.groupby("FechaE")["MontoNeto"].sum().reset_index()
        df = df.set_index("FechaE").sort_index()
        df = df.resample('D').asfreq().fillna(0)
        return df
    return None

def create_features(df):
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['es_finde']   = df['dia_semana'].isin([5, 6]).astype(int)
    # Lags para captar tendencia reciente
    for lag in [1, 7, 14, 30]:
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
    st.write("Predicción Ventas Reales | Suministros 1979 C.A.")

st.markdown("---")

# --- 4. CARGA DE RECURSOS ---
pw_clean = get_historical_data()

# --- 5. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_inicio_proy = st.date_input("Proyectar 30 días desde:", datetime.now())
    
    st.divider()
    
    if pw_clean is not None:
        ultima_fecha_data = pw_clean.index.max()
        st.success(f"✅ Datos cargados hasta: {ultima_fecha_data.strftime('%d/%m/%Y')}")
    else:
        st.error("❌ No se encontró 'ventas_historicas.csv'")
    
    btn_calcular = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- 6. LÓGICA DE PROYECCIÓN Y MÉTRICAS ---
if btn_calcular:
    if pw_clean is not None:
        with st.spinner("Entrenando modelo con ventas reales..."):
            
            # Entrenamiento rápido: filtramos ceros para que la proyección sea realista
            df_train = create_features(pw_clean[pw_clean['MontoNeto'] > 0]).dropna()
            features_cols = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7']
            
            model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
            model.fit(df_train[features_cols], df_train['MontoNeto'])

            # --- CÁLCULO DE VENTA ACUMULADA ---
            fecha_inicio_dt = pd.to_datetime(fecha_inicio_proy)
            mask_acumulado = (pw_clean.index >= fecha_inicio_dt) & (pw_clean.index <= ultima_fecha_data)
            venta_acumulada_real = pw_clean.loc[mask_acumulado, 'MontoNeto'].sum()

            # --- GENERACIÓN DE PROYECCIÓN ---
            df_loop = pw_clean.copy()
            results = []
            current_date = pd.Timestamp(fecha_inicio_proy)
            
            for _ in range(30):
                df_loop.loc[current_date, 'MontoNeto'] = 0
                df_with_features = create_features(df_loop)
                X_input = df_with_features.loc[[current_date], features_cols]
                
                pred = model.predict(X_input)[0]
                pred = max(0, float(pred)) 
                
                df_loop.loc[current_date, 'MontoNeto'] = pred
                results.append({
                    'Fecha': current_date.strftime('%Y-%m-%d'),
                    'Venta Proyectada': pred
                })
                current_date += timedelta(days=1)
            
            df_res = pd.DataFrame(results)
            total_30d_proy = df_res['Venta Proyectada'].sum()

            # Lógica Promedio Q1
            ventas_positivas = df_res[df_res['Venta Proyectada'] > 0]['Venta Proyectada']
            q1 = np.percentile(ventas_positivas, 25) if not ventas_positivas.empty else 0
            promedio_ajustado = ventas_positivas[ventas_positivas > q1].mean() if not ventas_positivas.empty else 0

            # --- VISUALIZACIÓN DE MÉTRICAS ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📈 VENTA ACUMULADA", f"${venta_acumulada_real:,.2f}")
            with m2:
                st.metric("💰 TOTAL PROYECTADO", f"${total_30d_proy:,.2f}")
            with m3:
                st.metric("📅 PERIODO", "30 Días")
            with m4:
                st.metric("📊 PROM. DIARIO (Q1)", f"${promedio_ajustado:,.2f}")

            # Gráfico de tendencia
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Fecha'], y=df_res['Venta Proyectada'],
                mode='lines+markers', line=dict(color='#FF4B4B', width=3),
                fill='tozeroy', fillcolor='rgba(255, 75, 75, 0.1)', name="Proyección"
            ))
            fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Monto ($)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Detalle Diario Proyectado")
            st.dataframe(df_res.style.format({'Venta Proyectada': '${:,.2f}'}), use_container_width=True)
            
    else:
        st.error("Error: Sube el archivo 'ventas_historicas.csv' a la misma carpeta de este script.")
else:
    st.info("💡 Haz clic en 'Calcular Proyección' para procesar los datos actuales.")

st.divider()
st.caption(f"© {datetime.now().year} | Suministros 1979 C.A. | Usuario: JBARCENAS")

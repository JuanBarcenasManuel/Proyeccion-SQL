import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- 1. CONFIGURACIÓN DE PÁGINA --
st.set_page_config(
    page_title="Predicción Ventas | Suministros 1979",
    layout="wide"
)

ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg')
ruta_datos = os.path.join(ruta_base, 'ventas_historicas.csv')

# Estilo para tarjetas de métricas
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

# --- 2. FUNCIONES ---
@st.cache_data
def get_historical_data():
    if os.path.exists(ruta_datos):
        df = pd.read_csv(ruta_datos)
        df["FechaE"] = pd.to_datetime(df["FechaE"])
        # Agrupamos por fecha y sumamos MontoNeto
        df = df.groupby("FechaE")["MontoNeto"].sum().reset_index()
        df = df.set_index("FechaE").sort_index()
        # Aseguramos continuidad diaria (rellena con 0 si falta un día)
        df = df.resample('D').asfreq().fillna(0)
        return df
    return None

def create_features(df):
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['es_finde']   = df['dia_semana'].isin([5, 6]).astype(int)
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
    st.write("Planificación de Ventas | Suministros 1979 C.A.")

st.markdown("---")

# --- 4. SIDEBAR (LOGICA AUTOMATIZADA) ---
pw_clean = get_historical_data()

with st.sidebar:
    st.header("⚙️ Configuración")
    
    # DETECCIÓN AUTOMÁTICA DE ÚLTIMA FECHA
    if pw_clean is not None:
        ultima_fecha_real = pw_clean.index.max()
        fecha_sugerida = ultima_fecha_real + timedelta(days=1)
        st.success(f"✅ Datos reales hasta: {ultima_fecha_real.strftime('%d/%m/%Y')}")
    else:
        fecha_sugerida = datetime.now()

    # El calendario ahora apunta al día siguiente del último dato del CSV
    fecha_inicio_proy = st.date_input("Proyectar desde:", fecha_sugerida)
    
    st.divider()
    st.write("**Ajuste de Conservadurismo**")
    ajuste_sensibilidad = st.slider("Sensibilidad del Modelo (%)", 60, 100, 85) / 100
    
    btn_calcular = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- 5. CÁLCULOS ---
if btn_calcular and pw_clean is not None:
    with st.spinner("Generando proyección con XGBoost..."):
        
        # Entrenamiento del modelo
        df_train = create_features(pw_clean[pw_clean['MontoNeto'] > 0]).dropna()
        features = ['dia_semana', 'dia_mes', 'es_finde', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7']
        model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=5, subsample=0.8)
        model.fit(df_train[features], df_train['MontoNeto'])

        # VENTA ACUMULADA REAL (Mes actual hasta la fecha de corte)
        fecha_dt = pd.to_datetime(fecha_inicio_proy)
        mask_mes_actual = (pw_clean.index.month == fecha_dt.month) & (pw_clean.index.year == fecha_dt.year)
        venta_acumulada_real = pw_clean.loc[mask_mes_actual, 'MontoNeto'].sum()

        # Proyección a 30 días
        df_loop = pw_clean.copy()
        results = []
        curr_date = pd.Timestamp(fecha_inicio_proy)
        # Techo diario para evitar proyecciones irreales (Percentil 90)
        techo_diario = pw_clean['MontoNeto'][pw_clean['MontoNeto'] > 0].quantile(0.90)
        
        for _ in range(30):
            df_loop.loc[curr_date, 'MontoNeto'] = 0
            feats = create_features(df_loop).loc[[curr_date], features]
            pred = float(model.predict(feats)[0])
            pred = min(pred * ajuste_sensibilidad, techo_diario)
            pred = max(0, pred)
            
            df_loop.loc[curr_date, 'MontoNeto'] = pred
            results.append({
                'Fecha': curr_date,
                'Venta Proyectada': pred,
                'DiaSemana': curr_date.dayofweek
            })
            curr_date += timedelta(days=1)
        
        df_res = pd.DataFrame(results)
        total_proy = df_res['Venta Proyectada'].sum()

        # Promedio Diario (Solo Días Hábiles: Lunes a Viernes)
        dias_habiles = df_res[df_res['DiaSemana'] < 5]
        promedio_habil = dias_habiles['Venta Proyectada'].mean()

        # --- 6. VISUALIZACIÓN ---
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("📈 ACUM. REAL (MES)", f"${venta_acumulada_real:,.2f}")
        with m2: st.metric("💰 TOTAL PROYECTADO", f"${total_proy:,.2f}")
        with m3: st.metric("📅 PERIODO", "30 Días")
        with m4: st.metric("📊 PROM. DÍA HÁBIL", f"${promedio_habil:,.2f}")

        # Gráfico de Proyección
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_res['Fecha'], 
            y=df_res['Venta Proyectada'], 
            mode='lines+markers', 
            name="Proyección", 
            fill='tozeroy',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.update_layout(
            title="Tendencia de Venta Proyectada (Próximos 30 días)",
            template="plotly_white", 
            hovermode="x unified",
            xaxis_title="Fecha",
            yaxis_title="Monto ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detalle Diario en Tabla
        st.subheader("📋 Detalle Diario de Proyección")
        df_final = df_res[['Fecha', 'Venta Proyectada']].copy()
        df_final['Fecha'] = df_final['Fecha'].dt.strftime('%d/%m/%Y')
        st.dataframe(df_final.set_index('Fecha').style.format('${:,.2f}'), use_container_width=True)

st.divider()
st.caption(f"© {datetime.now().year} | Suministros 1979 C.A. | Usuario: JBARCENAS")

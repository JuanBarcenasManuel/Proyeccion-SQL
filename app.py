import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Predicción Ventas | Zudalpro", 
    layout="wide", 
    page_icon="🛒"
)

# Estilo visual personalizado
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MANEJO DE RUTAS Y LOGO ---
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg')

col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, width=150)
    else:
        st.subheader("📦 Suministros")

with col_titulo:
    st.title("Sistema de Proyección de Demanda")
    st.write("Análisis predictivo de ventas para Suministros")

st.markdown("---")

# --- 3. CARGA DEL MODELO ---
@st.cache_resource
def load_xgboost_model():
    ruta_modelo = os.path.join(ruta_base, 'modelo_ventas.json')
    if os.path.exists(ruta_modelo):
        model = xgb.XGBRegressor()
        model.load_model(ruta_modelo)
        return model
    return None

model = load_xgboost_model()

# --- 4. FUNCIÓN DE PROCESAMIENTO ---
def create_features_row(date, l1, l2, l7, l14, r7):
    date_pd = pd.Timestamp(date)
    return pd.DataFrame({
        'dia_semana': [date_pd.dayofweek],
        'dia_mes': [date_pd.day],
        'mes': [date_pd.month],
        'es_finde': [1 if date_pd.dayofweek in [5, 6] else 0],
        'lag_1': [l1],
        'lag_2': [l2],
        'lag_7': [l7],
        'lag_14': [l14],
        'rolling_mean_7': [r7]
    })

# --- 5. INTERFAZ LATERAL (INPUTS) ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    fecha_sel = st.date_input("Fecha a Proyectar", datetime(2026, 2, 13))
    
    st.divider()
    st.subheader("Valores Históricos")
    lag1 = st.number_input("Ventas Ayer ($)", value=15000.0, step=100.0)
    lag2 = st.number_input("Ventas hace 2 días ($)", value=14800.0, step=100.0)
    lag7 = st.number_input("Ventas hace 7 días ($)", value=14500.0, step=100.0)
    lag14 = st.number_input("Ventas hace 14 días ($)", value=14000.0, step=100.0)
    roll7 = st.number_input("Promedio Móvil (7 días) ($)", value=14700.0, step=100.0)
    
    predict_btn = st.button("🚀 Calcular Proyección", use_container_width=True)

# --- 6. CÁLCULO DE PREDICCIÓN ---
if 'pred' not in st.session_state:
    st.session_state.pred = None

if predict_btn:
    if model:
        # Generar las columnas en el orden exacto que requiere el modelo
        features_df = create_features_row(fecha_sel, lag1, lag2, lag7, lag14, roll7)
        order = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']
        
        # Ejecutar predicción
        resultado = model.predict(features_df[order])[0]
        st.session_state.pred = max(0, resultado) # No permitir ventas negativas
    else:
        st.error("⚠️ Error: No se encontró el archivo 'modelo_ventas.json' en la carpeta.")

# --- 7. PANEL PRINCIPAL (RESULTADOS) ---
col_stats, col_chart = st.columns([1, 2], gap="large")
pred = st.session_state.pred

with col_stats:
    st.subheader("📊 Resultado")
    if pred is not None:
        delta_val = ((pred / lag1) - 1) * 100 if lag1 != 0 else 0
        st.metric(
            label=f"Venta Estimada ({fecha_sel})", 
            value=f"${pred:,.2f}", 
            delta=f"{delta_val:.2f}% vs ayer"
        )
        
        with st.expander("Ver variables enviadas"):
            st.write(create_features_row(fecha_sel, lag1, lag2, lag7, lag14, roll7).T)
    else:
        st.info("Ajuste los valores en el panel izquierdo y presione 'Calcular Proyección'.")

with col_chart:
    st.subheader("📈 Tendencia Estimada (7 días)")
    if pred is not None:
        # Generar una curva visual simple para los próximos días
        fechas_futuras = pd.date_range(start=fecha_sel, periods=7)
        # El primer valor es la predicción real, los demás son tendencia simulada para el gráfico
        ventas_sim = [pred] + [pred * (1 + np.random.uniform(-0.04, 0.04)) for _ in range(6)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fechas_futuras, y=ventas_sim,
            mode='lines+markers',
            line=dict(color='#FF4B4B', width=4),
            fill='tozeroy',
            fillcolor='rgba(255, 75, 75, 0.1)',
            name="Proyección"
        ))
        
        fig.update_layout(
            height=400, 
            template="plotly_white", 
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Fecha",
            yaxis_title="Ventas ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- 8. PIE DE PÁGINA ---
st.divider()
st.caption(f"© {datetime.now().year} Zudalpro Suministros | Modelo Predictivo XGBoost v1.0")
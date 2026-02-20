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
ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'Suministros.jpg') 
ruta_datos = os.path.join(ruta_base, 'ventas_historicas.csv')
ruta_modelo = os.path.join(ruta_base, 'modelo_ventas.json')

@st.cache_data
def get_historical_data():
    """Carga los datos desde el CSV generado por el extractor"""
    if os.path.exists(ruta_datos):
        df = pd.read_csv(ruta_datos)
        df["FechaE"] = pd.to_datetime(df["FechaE"])
        df = df.set_index("FechaE")
        # Aseguramos frecuencia diaria y llenamos huecos con 0
        df = df.resample('D').asfreq().fillna(0)
        return df
    return None

@st.cache_resource
def load_model():
    """Carga el modelo XGBoost guardado"""
    if os.path.exists(ruta_modelo):
        model = xgb.XGBRegressor()
        model.load_model(ruta_modelo)
        return model
    return None

def create_features(df):
    """Ingeniería de variables para el modelo"""
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['es_finde']   = df['dia_semana'].isin([5, 6]).astype(int)
    
    for lag in [1, 2, 7, 14]:
        df[f'lag_{lag}'] = df['MontoNeto'].shift(lag)
    
    df['rolling_mean_7'] = df['MontoNeto'].shift(1).rolling(window=7).mean()
    return df

# --- 3. ENCABEZADO Y LOGO ---
col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    if os.path.exists(ruta_logo):
        st.image(ruta_logo, width=150)
    else:
        st.subheader("📦 Suministros 1979")

with col_titulo:
    st.title("Sistema de Proyección de Demanda")
    st.write("Predicción Ventas | Suministros 1979 C.A.")

st.markdown("---")

# --- 4. CARGA DE RECURSOS ---
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

# --- 6. LÓGICA DE PROYECCIÓN ---
if btn_calcular:
    if model is not None and pw_clean is not None:
        with st.spinner("Generando proyección para los próximos 30 días..."):
            df_loop = pw_clean.copy()
            features_cols = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']
            results = []
            
            current_date = pd.Timestamp(fecha_inicio)
            
            for _ in range(30):
                # 1. Crear fila temporal
                df_loop.loc[current_date, 'MontoNeto'] = 0
                
                # 2. Generar variables
                df_with_features = create_features(df_loop)
                
                # 3. Preparar input (última fila)
                X_input = df_with_features.loc[[current_date], features_cols]
                
                # 4. Predicción
                pred = model.predict(X_input)[0]
                pred = max(0, float(pred)) 
                
                # 5. Actualizar historial para el siguiente ciclo
                df_loop.loc[current_date, 'MontoNeto'] = pred
                
                results.append({
                    'Fecha': current_date.strftime('%Y-%m-%d'),
                    'Día': current_date.day_name(),
                    'Venta Proyectada': pred
                })
                current_date += timedelta(days=1)
            
            df_res = pd.DataFrame(results)
            total_30d = df_res['Venta Proyectada'].sum()

            # --- VISUALIZACIÓN ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("💰 TOTAL PROYECTADO", f"${total_30d:,.2f}")
            with m2:
                st.metric("📅 PERIODO", "30 Días")
            with m3:
                promedio = total_30d / 30
                st.metric("📊 PROM. DIARIO", f"${promedio:,.2f}")

            # Gráfico Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Fecha'], 
                y=df_res['Venta Proyectada'],
                mode='lines+markers',
                line=dict(color='#FF4B4B', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 75, 75, 0.1)',
                name="Proyección"
            ))
            fig.update_layout(
                title="Tendencia de Demanda Proyectada",
                template="plotly_white", 
                xaxis_title="Fecha", 
                yaxis_title="Monto ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabla (Ajustada para compatibilidad con Pandas Styler)
            st.subheader("📋 Detalle Diario")
            st.dataframe(
                df_res.style.format({'Venta Proyectada': '${:,.2f}'}),
                use_container_width=True
            )
            
            # Descarga
            st.download_button(
                label="📥 Descargar Reporte (CSV)",
                data=df_res.to_csv(index=False).encode('utf-8'),
                file_name=f"proyeccion_{fecha_inicio}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.error("Error: Verifica que el modelo (.json) y los datos (.csv) estén en GitHub.")

else:
    st.info("💡 Selecciona una fecha y presiona el botón para iniciar la proyección.")

# --- 7. PIE DE PÁGINA ---
st.divider()
st.caption(f"© {datetime.now().year} | Suministros 1979 C.A.")

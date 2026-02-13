import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Predicción Ventas | Planificacion de la demanda | Departamento de Cadenas de Suministros",
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

# --- 2. PARÁMETROS DE CONEXIÓN Y RUTAS ---
SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020"
DB_PRINCIPAL = "EnterpriseAdminDb"

ruta_base = os.path.dirname(__file__)
# Asegúrate de que este nombre coincida con tu archivo de imagen
ruta_logo = os.path.join(ruta_base, 'image_ce2f2b.png') 

@st.cache_data
def get_historical_data():
    """Extrae datos reales de SQL Server"""
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DB_PRINCIPAL};UID={UID};PWD={PWD};Encrypt=no;TrustServerCertificate=yes;"
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")
    
    SQL = text(f"""
        SELECT I.FechaE, 
               TRY_CONVERT(float, I.Cantidad) * TRY_CONVERT(float, I.Precio) AS MontoNeto 
        FROM [{DB_PRINCIPAL}].[dbo].[SAITEMFAC] AS I 
        WHERE I.TipoFac = 'A' AND I.FechaE >= '2025-01-01'
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(SQL, conn, parse_dates=["FechaE"])
    
    df["FechaE"] = pd.to_datetime(df["FechaE"]).dt.normalize()
    pw_clean = df.groupby("FechaE")["MontoNeto"].sum().to_frame()
    # Resample para asegurar que no falten días (rellena con 0)
    return pw_clean.resample('D').asfreq().fillna(0)

@st.cache_resource
def load_model():
    """Carga el modelo XGBoost guardado (.json)"""
    ruta_modelo = os.path.join(ruta_base, 'modelo_ventas.json')
    if os.path.exists(ruta_modelo):
        model = xgb.XGBRegressor()
        model.load_model(ruta_modelo)
        return model
    return None

def create_features(df):
    """Ingeniería de variables idéntica al entrenamiento"""
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
        st.image(ruta_logo, width=200)
    else:
        st.subheader("📦 Suministros 1979 C.A.")

with col_titulo:
    st.title("Sistema de Proyección de Demanda")
    st.write("Análisis dinámico basado en historial de SQL Server y XGBoost")

st.markdown("---")

# --- 4. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    # Fecha desde la cual comenzará la proyección de 30 días
    fecha_inicio = st.date_input("Proyectar 30 días desde:", datetime.now())
    st.divider()
    btn_calcular = st.button(" Sincronizar y Calcular", use_container_width=True)
    st.info("Se consultará el SQL Server para obtener los datos más recientes.")

# --- 5. LÓGICA DE PROYECCIÓN (CASCADA RECURSIVA) ---
# Carga inicial de datos y modelo
pw_clean = get_historical_data()
model = load_model()

if btn_calcular:
    if model is not None:
        with st.spinner("Calculando proyección..."):
            # Copiamos el historial para ir agregando las predicciones
            df_loop = pw_clean.copy()
            features_cols = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']
            results = []
            
            # Bucle de 30 días
            current_date = pd.Timestamp(fecha_inicio)
            
            for _ in range(30):
                # 1. Crear la fila para la fecha actual (inicialmente en 0)
                df_loop.loc[current_date, 'MontoNeto'] = 0
                
                # 2. Generar las variables (lags y rolling mean) basadas en el historial + predicciones previas
                df_with_features = create_features(df_loop)
                
                # 3. Seleccionar solo la fila que vamos a predecir
                X_input = df_with_features.loc[[current_date], features_cols]
                
                # 4. Realizar la predicción
                pred = model.predict(X_input)[0]
                pred = max(0, float(pred)) # Evitar ventas negativas
                
                # 5. Actualizar el DataFrame de trabajo con la predicción (para el siguiente lag)
                df_loop.loc[current_date, 'MontoNeto'] = pred
                
                # 6. Guardar en la lista de resultados
                results.append({
                    'Fecha': current_date.strftime('%Y-%m-%d'),
                    'Día': current_date.day_name(),
                    'Venta Proyectada': pred
                })
                
                # Avanzar al siguiente día
                current_date += timedelta(days=1)
            
            # Convertir resultados a DataFrame
            df_res = pd.DataFrame(results)
            total_30d = df_res['Venta Proyectada'].sum()

            # --- 6. VISUALIZACIÓN DE RESULTADOS ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("💰 TOTAL PROYECTADO (30 DÍAS)", f"${total_30d:,.2f}")
            with m2:
                st.metric("📅 INICIO DE PROYECCIÓN", fecha_inicio.strftime('%d/%m/%Y'))
            with m3:
                promedio = total_30d / 30
                st.metric("📊 PROMEDIO DIARIO", f"${promedio:,.2f}")

            # Gráfico de tendencia
            st.subheader("📈 Tendencia de Ventas Proyectada")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Fecha'], 
                y=df_res['Venta Proyectada'],
                mode='lines+markers',
                line=dict(color='#FF4B4B', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 75, 75, 0.1)',
                name="Venta"
            ))
            fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Monto ($)")
            st.plotly_chart(fig, use_container_width=True)

            # Tabla Detallada con formato y gradiente
            st.subheader("📋 Detalle de Ventas Diarias")
            st.dataframe(
                df_res.style.background_gradient(subset=['Venta Proyectada'], cmap='Oranges')
                .format({'Venta Proyectada': '${:,.2f}'}),
                use_container_width=True,
                height=450
            )
            
            # Botón de descarga
            st.download_button(
                label="📥 Descargar Reporte (CSV)",
                data=df_res.to_csv(index=False).encode('utf-8'),
                file_name=f"proyeccion_suministros_{fecha_inicio}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.error("❌ No se encontró el archivo 'modelo_ventas.json'. Por favor, verifica que el modelo esté en la misma carpeta que este script.")
else:
    st.info("💡 Selecciona una fecha en el calendario y presiona el botón 'Sincronizar y Calcular' para generar la proyección de 30 días.")

# --- 7. PIE DE PÁGINA ---
st.divider()
st.caption(f"© {datetime.now().year}  Suministros | Proyección de Demanda ")

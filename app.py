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

# Estilo visual personalizado
st.markdown("""
    <style>
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PARÁMETROS DE CONEXIÓN Y RUTAS ---
SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020"
DB_PRINCIPAL = "EnterpriseAdminDb"

ruta_base = os.path.dirname(__file__)
ruta_logo = os.path.join(ruta_base, 'image_ce2f2b.png') 

@st.cache_data(show_spinner=False)
def get_historical_data():
    """Extrae datos reales de SQL Server con manejo de errores"""
    try:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER};DATABASE={DB_PRINCIPAL};"
            f"UID={UID};PWD={PWD};Encrypt=no;TrustServerCertificate=yes;"
            f"Connection Timeout=30;"
        )
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")
        
        SQL = text(f"""
            SELECT I.FechaE, 
                   TRY_CONVERT(float, I.Cantidad) * TRY_CONVERT(float, I.Precio) AS MontoNeto 
            FROM [{DB_PRINCIPAL}].[dbo].[SAITEMFAC] AS I 
            WHERE I.TipoFac = 'A' AND I.FechaE >= '2025-01-01'
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(SQL, conn, parse_dates=["FechaE"])
        
        if df.empty:
            return None

        df["FechaE"] = pd.to_datetime(df["FechaE"]).dt.normalize()
        pw_clean = df.groupby("FechaE")["MontoNeto"].sum().to_frame()
        return pw_clean.resample('D').asfreq().fillna(0)
    except Exception as e:
        st.sidebar.error(f"Error de conexión SQL: {e}")
        return None

@st.cache_resource
def load_model():
    """Carga el modelo XGBoost guardado (.json)"""
    ruta_modelo = os.path.join(ruta_base, 'modelo_ventas.json')
    if os.path.exists(ruta_modelo):
        try:
            model = xgb.XGBRegressor()
            model.load_model(ruta_modelo)
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
    return None

def create_features(df):
    """Ingeniería de variables para la predicción recursiva"""
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
    st.write("Análisis dinámico | Departamento de Cadenas de Suministros")

st.markdown("---")

# --- 4. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    fecha_inicio = st.date_input("Proyectar 30 días desde:", datetime.now())
    st.divider()
    btn_calcular = st.button("🚀 Sincronizar y Calcular", use_container_width=True)
    st.info("La proyección utiliza un modelo XGBoost con lógica de cascada recursiva.")

# --- 5. LÓGICA DE PROYECCIÓN ---
pw_clean = get_historical_data()
model = load_model()

if btn_calcular:
    if pw_clean is None:
        st.error("No se pudo conectar a la base de datos local. Verifique la VPN o conexión de red.")
    elif model is None:
        st.error("Archivo 'modelo_ventas.json' no encontrado en el servidor.")
    else:
        with st.spinner("Procesando datos y generando proyección..."):
            df_loop = pw_clean.copy()
            features_cols = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']
            results = []
            
            current_date = pd.Timestamp(fecha_inicio)
            
            # Bucle de 30 días
            for _ in range(30):
                df_loop.loc[current_date, 'MontoNeto'] = 0
                df_with_features = create_features(df_loop)
                
                # Seleccionar fila actual para predecir
                X_input = df_with_features.loc[[current_date], features_cols]
                
                # Predicción y limpieza
                pred = max(0, float(model.predict(X_input)[0]))
                
                # Alimentar el bucle para el siguiente lag
                df_loop.loc[current_date, 'MontoNeto'] = pred
                
                results.append({
                    'Fecha': current_date.strftime('%Y-%m-%d'),
                    'Día': current_date.day_name(locale='es_ES'),
                    'Venta Proyectada': pred
                })
                current_date += timedelta(days=1)
            
            df_res = pd.DataFrame(results)
            total_30d = df_res['Venta Proyectada'].sum()

            # --- 6. VISUALIZACIÓN ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("💰 TOTAL PROYECTADO", f"${total_30d:,.2f}")
            with m2:
                st.metric("📅 FECHA INICIO", fecha_inicio.strftime('%d/%m/%Y'))
            with m3:
                st.metric("📊 PROMEDIO DIARIO", f"${(total_30d/30):,.2f}")

            # Gráfico Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_res['Fecha'], 
                y=df_res['Venta Proyectada'],
                mode='lines+markers',
                line=dict(color='#FF4B4B', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 75, 75, 0.1)',
                name="Venta Est."
            ))
            fig.update_layout(
                template="plotly_white", 
                title="Pronóstico de Ventas (Próximos 30 Días)",
                xaxis_title="Fecha",
                yaxis_title="Monto ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabla de detalles
            st.subheader("📋 Detalle Diario de Proyección")
            st.dataframe(
                df_res.style.background_gradient(subset=['Venta Proyectada'], cmap='Oranges')
                .format({'Venta Proyectada': '${:,.2f}'}),
                use_container_width=True,
                height=450
            )
            
            # Exportación
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar Reporte CSV",
                csv,
                f"proyeccion_{fecha_inicio}.csv",
                "text/csv",
                use_container_width=True
            )
else:
    st.info("Seleccione la fecha de inicio en el panel izquierdo para calcular el pronóstico de demanda.")

# --- 7. PIE DE PÁGINA ---
st.divider()
st.caption(f"© {datetime.now().year} Suministros 1979 C.A. | Planificación de la Demanda | XGBoost v2.1")

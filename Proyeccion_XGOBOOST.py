import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import os
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. CONFIGURACIÓN DE CONEXIÓN ---
SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020"
DB_PRINCIPAL = "EnterpriseAdminDb"

def make_engine(database):
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={database};UID={UID};PWD={PWD};Encrypt=no;TrustServerCertificate=yes;"
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")

# --- 2. EXTRACCIÓN DE DATOS ---
print("🚀 Extrayendo datos desde SQL Server...")
engine = make_engine(DB_PRINCIPAL)
SQL = text(f"""
    SELECT I.FechaE, 
           TRY_CONVERT(float, I.Cantidad) * TRY_CONVERT(float, I.Precio) AS MontoNeto 
    FROM [{DB_PRINCIPAL}].[dbo].[SAITEMFAC] AS I 
    WHERE I.TipoFac = 'A' AND I.FechaE >= '2025-01-01'
""")

with engine.connect() as conn:
    df_raw = pd.read_sql(SQL, conn, parse_dates=["FechaE"])

# --- 3. PREPARACIÓN Y LIMPIEZA ---
df_raw["FechaE"] = pd.to_datetime(df_raw["FechaE"]).dt.normalize()
pw_clean = df_raw.groupby("FechaE")["MontoNeto"].sum().to_frame()
pw_clean = pw_clean.resample('D').asfreq().fillna(0)

def create_features(df):
    df = df.copy()
    df['dia_semana'] = df.index.dayofweek
    df['dia_mes']    = df.index.day
    df['mes']        = df.index.month
    df['es_finde']   = df['dia_semana'].isin([5, 6]).astype(int)
    for lag in [1, 2, 7, 14]:
        df[f'lag_{lag}'] = df['MontoNeto'].shift(lag)
    df['rolling_mean_7'] = df['MontoNeto'].shift(1).rolling(window=7).mean()
    return df

features = ['dia_semana', 'dia_mes', 'mes', 'es_finde', 'lag_1', 'lag_2', 'lag_7', 'lag_14', 'rolling_mean_7']

# --- 4. FASE DE BACKTEST (TESTEO DEL ÚLTIMO MES) ---
print("🧪 Iniciando fase de Backtest (Validación)...")
fecha_final_real = pw_clean.index.max()
fecha_inicio_backtest = fecha_final_real - timedelta(days=30)

df_train_bt = pw_clean[pw_clean.index < fecha_inicio_backtest]
df_train_bt_ready = create_features(df_train_bt).dropna()

model_bt = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42)
model_bt.fit(df_train_bt_ready[features], df_train_bt_ready['MontoNeto'])

# Bucle de predicción para el mes de testeo
df_bt_loop = df_train_bt.copy()
for _ in range(31):
    n_f = df_bt_loop.index.max() + timedelta(days=1)
    if n_f > fecha_final_real: break
    df_bt_loop.loc[n_f, 'MontoNeto'] = 0
    temp_feats = create_features(df_bt_loop)
    pred = model_bt.predict(temp_feats.loc[[n_f], features])[0]
    df_bt_loop.loc[n_f, 'MontoNeto'] = max(0, pred)

pred_backtest = df_bt_loop[df_bt_loop.index >= fecha_inicio_backtest]

# --- 5. FASE DE PROYECCIÓN FUTURA (PRÓXIMOS 30 DÍAS) ---
print("🔮 Entrenando modelo final y proyectando futuro...")
df_ready_full = create_features(pw_clean).dropna()
model_final = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42)
model_final.fit(df_ready_full[features], df_ready_full['MontoNeto'])

df_proy_loop = pw_clean.copy()
for _ in range(30):
    n_f = df_proy_loop.index.max() + timedelta(days=1)
    df_proy_loop.loc[n_f, 'MontoNeto'] = 0
    temp_feats = create_features(df_proy_loop)
    pred = model_final.predict(temp_feats.loc[[n_f], features])[0]
    df_proy_loop.loc[n_f, 'MontoNeto'] = max(0, pred)

proyeccion_futura = df_proy_loop[df_proy_loop.index > fecha_final_real]

# --- 6. EXPORTACIÓN Y RUTA DE GUARDADO ---
# Definimos la ruta: Usamos la carpeta donde está este script
try:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
except:
    directorio_actual = os.getcwd() # Para Jupyter o consolas interactivas

nombre_archivo = "Backtest_y_Proyeccion_Ventas_XGBoost.xlsx"
ruta_final = os.path.join(directorio_actual, nombre_archivo)

# Consolidación de datos para el Excel
df_real_export = pw_clean.copy()
df_real_export.columns = ['Venta Real']

# Unimos todo
df_final_excel = df_real_export.copy()
df_final_excel['Testeo_Mes_Pasado'] = pred_backtest['MontoNeto']

# Añadimos las filas del futuro
df_futuro_export = proyeccion_futura.rename(columns={'MontoNeto': 'Proyeccion_Futura'})
df_consolidado = pd.concat([df_final_excel, df_futuro_export])

# Guardar en Excel
df_consolidado.reset_index().to_excel(ruta_final, index=False)

# --- 7. RESUMEN EN PANTALLA ---
print("\n" + "="*30)
print("📊 RESUMEN DE RESULTADOS")
print("="*30)
print(f"Venta Real Últimos 30 días: {pw_clean[pw_clean.index >= fecha_inicio_backtest]['MontoNeto'].sum():,.2f}")
print(f"Predicción del Testeo (Backtest): {pred_backtest['MontoNeto'].sum():,.2f}")
print(f"PROYECCIÓN PRÓXIMOS 30 DÍAS: {proyeccion_futura['MontoNeto'].sum():,.2f}")
print("-" * 30)
print(f"📍 ARCHIVO GUARDADO EN: {ruta_final}")
print("="*30)

# Opcional: Mostrar gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(x=pw_clean.index, y=pw_clean['MontoNeto'], name='Venta Real', line=dict(color='#1f77b4')))
fig.add_trace(go.Scatter(x=pred_backtest.index, y=pred_backtest['MontoNeto'], name='Testeo (Backtest)', line=dict(color='orange', dash='dot')))
fig.add_trace(go.Scatter(x=proyeccion_futura.index, y=proyeccion_futura['MontoNeto'], name='Proyección Futura', line=dict(color='red', width=3)))
fig.update_layout(template="plotly_white", hovermode="x unified", title="Backtest y Proyección Ventas")
fig.show()

import pickle
with open ("modelo.pkl","wb") as f:
        pickle.dump(model_final,f)


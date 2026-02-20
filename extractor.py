import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os

# --- CONFIGURACIÓN ---
SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020" # Asegúrate de que sean correctas
DB_PRINCIPAL = "EnterpriseAdminDb"

# Esto asegura que el CSV se guarde en la misma carpeta que este script
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_SALIDA = os.path.join(ruta_actual, "ventas_historicas.csv")

def extraer_datos():
    print(f"--- Iniciando Proceso de Extracción ---")
    print(f"Carpeta de destino: {ruta_actual}")
    
    try:
        # 1. Configurar Conexión
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER};"
            f"DATABASE={DB_PRINCIPAL};"
            f"UID={UID};"
            f"PWD={PWD};"
            "Encrypt=no;"
            "TrustServerCertificate=yes;"
        )
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")
        
        # 2. Definir Query
        SQL = text(f"""
            SELECT I.FechaE, 
                   TRY_CONVERT(float, I.Cantidad) * TRY_CONVERT(float, I.Precio) AS MontoNeto 
            FROM [{DB_PRINCIPAL}].[dbo].[SAITEMFAC] AS I 
            WHERE I.TipoFac = 'A' AND I.FechaE >= '2025-01-01'
        """)
        
        # 3. Ejecutar y leer
        with engine.connect() as conn:
            print("Conectando a SQL Server...")
            df = pd.read_sql(SQL, conn)
        
        if df.empty:
            print("⚠️ La consulta no devolvió datos. Revisa las fechas o el TipoFac.")
            return

        # 4. Procesamiento
        print(f"Datos extraídos: {len(df)} filas.")
        df["FechaE"] = pd.to_datetime(df["FechaE"]).dt.normalize()
        pw_clean = df.groupby("FechaE")["MontoNeto"].sum().to_frame()
        
        # Rellenar días faltantes con 0
        pw_clean = pw_clean.resample('D').asfreq().fillna(0)
        
        # 5. Guardar CSV (Aquí es donde se crea el archivo)
        pw_clean.to_csv(ARCHIVO_SALIDA)
        
        if os.path.exists(ARCHIVO_SALIDA):
            print(f"✅ ¡ÉXITO! Archivo creado en: {ARCHIVO_SALIDA}")
            print(f"Tamaño del archivo: {os.path.getsize(ARCHIVO_SALIDA)} bytes")
        else:
            print("❌ El archivo no se creó por un problema de permisos en la carpeta.")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {str(e)}")

if __name__ == "__main__":
    extraer_datos()
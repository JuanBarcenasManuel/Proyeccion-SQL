import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os
import subprocess
from datetime import datetime

# --- 1. CONFIGURACIÓN DE CONEXIÓN ---
SERVER = "192.168.150.6,2431"
UID, PWD = "Jbarcenas", "Juanbarcenas2020"
DB_PRINCIPAL = "EnterpriseAdminDb"

# Rutas automáticas
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_SALIDA = os.path.join(ruta_actual, "ventas_historicas.csv")

def extraer_datos():
    """Extrae datos de SQL y genera el CSV local"""
    print(f"[{datetime.now()}] Iniciando extracción...")
    try:
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DB_PRINCIPAL};UID={UID};PWD={PWD};Encrypt=no;TrustServerCertificate=yes;"
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(conn_str)}")
        
        SQL = text(f"""
            SELECT I.FechaE, 
                   TRY_CONVERT(float, I.Cantidad) * TRY_CONVERT(float, I.Precio) AS MontoNeto 
            FROM [{DB_PRINCIPAL}].[dbo].[SAITEMFAC] AS I 
            WHERE I.TipoFac = 'A' AND I.FechaE >= '2025-01-01'
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(SQL, conn)
        
        if df.empty:
            print("⚠️ No hay datos nuevos.")
            return False

        # Procesamiento
        df["FechaE"] = pd.to_datetime(df["FechaE"]).dt.normalize()
        pw_clean = df.groupby("FechaE")["MontoNeto"].sum().to_frame()
        pw_clean = pw_clean.resample('D').asfreq().fillna(0)
        
        # Guardar CSV
        pw_clean.to_csv(ARCHIVO_SALIDA)
        print(f"✅ CSV guardado localmente en: {ARCHIVO_SALIDA}")
        return True

    except Exception as e:
        print(f"❌ Error en SQL: {e}")
        return False

def subir_a_github():
    """Ejecuta los comandos de Git para subir el archivo a la nube"""
    print("🚀 Sincronizando con GitHub...")
    try:
        # 1. Cambiar al directorio del script
        os.chdir(ruta_actual)
        
        # 2. Comandos de Git
        subprocess.run(["git", "add", "ventas_historicas.csv"], check=True)
        
        fecha_msg = datetime.now().strftime('%Y-%m-%d %H:%M')
        subprocess.run(["git", "commit", "-m", f"Actualización automática: {fecha_msg}"], check=True)
        
        # Intenta subir a la rama 'main' (cambia a 'master' si es necesario)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        print("✅ ¡GitHub actualizado! La app de Streamlit se refrescará en breve.")
    except Exception as e:
        print(f"❌ Error al subir a GitHub: {e}")
        print("Asegúrate de que la carpeta es un repositorio de Git y tienes permisos.")

if __name__ == "__main__":
    if extraer_datos():
        subir_a_github()
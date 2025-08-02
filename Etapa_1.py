import pandas as pd
import math

# Cargar archivo inicial y seleccionar variables relevantes
archivo_original = 'Datos_Iniciales.txt'
columnas_utiles = [
    'Date', 'Time', 'Global_active_power',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]

df = pd.read_csv(archivo_original, sep=';', usecols=columnas_utiles)

# Combinar fecha y hora en una sola columna tipo datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Eliminar columnas originales de fecha y hora
df.drop(columns=['Date', 'Time'], inplace=True)

# Reordenar columnas
df = df[['Datetime', 'Global_active_power',
         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]

# Convertir a tipo numérico e interpolar valores faltantes linealmente
columnas_numericas = [
    'Global_active_power', 'Sub_metering_1',
    'Sub_metering_2', 'Sub_metering_3'
]
df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors='coerce')
df[columnas_numericas] = df[columnas_numericas].interpolate(method='linear')


def guardar_partes(df: pd.DataFrame, filas_por_archivo: int, carpeta_salida: str = "Datos/") -> None:
    """
    Divide un DataFrame en partes y guarda cada una como archivo separado.
    
    Args:
        df (pd.DataFrame): DataFrame a dividir y guardar.
        filas_por_archivo (int): Cantidad de filas por archivo.
        carpeta_salida (str): Carpeta donde se guardarán los archivos.
    """
    total_filas = len(df)
    total_archivos = math.ceil(total_filas / filas_por_archivo)

    for i in range(total_archivos):
        inicio = i * filas_por_archivo
        fin = (i + 1) * filas_por_archivo
        df_parte = df.iloc[inicio:fin]
        nombre_archivo = f"{carpeta_salida}datos_limpios_parte_{i+1}.txt"
        df_parte.to_csv(nombre_archivo, sep=';', index=False)
        print(f"Guardado: {nombre_archivo} ({len(df_parte)} filas)")


# Guardar los datos limpios divididos en partes
guardar_partes(df, filas_por_archivo=100_000)
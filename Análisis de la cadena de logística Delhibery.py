
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el conjunto de datos
df = pd.read_csv('delhivery.csv')

# Mostrar las primeras filas del marco de datos
print(df.head())

# Comprobar si faltan valores y tipos de datos
# Mostrar resumen de valores faltantes
missing_values = df.isnull().sum()
print('Resumen de valores faltantes:')
print(missing_values)

# Mostrar los tipos de datos de cada columna
dtypes = df.dtypes
print('Resumen de tipos de datos:')
print(dtypes)

# Complete los valores faltantes en 'nombre_fuente' y 'nombre_destino' con 'Desconocido'
df['source_name'].fillna('Unknown', inplace=True)
df['destination_name'].fillna('Unknown', inplace=True)

# Verifique los cambios mostrando el recuento de valores únicos en estas columnas
print('Valores únicos en source_name después de completar los valores faltantes:')
print(df['source_name'].unique())
print('Valores únicos en nombre_destino después de completar los valores faltantes:')
print(df['destination_name'].unique())

#Los valores faltantes en las columnas nombre_origen y nombre_destino se han completado correctamente con "Desconocido".

#A continuación, procedamos con un análisis de datos exploratorio (EDA) para obtener información sobre las operaciones logísticas de Delhivery. Comenzaremos analizando la distribución de las 
#duraciones y distancias de los viajes, así como los factores que afectan los tiempos de entrega.

#Comencemos visualizando la distribución de las duraciones y distancias de los viajes.

# Trazar la distribución de las duraciones de los viajes
plt.figure(figsize=(12, 6))
sns.histplot(df['actual_time'], bins=50, kde=True, color = "fuchsia", edgecolor = "white")
plt.title('Distribución de la duración del viaje\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Duración del viaje (minutos)\n')
plt.ylabel('Frecuencia\n')
plt.show()

# Trazar la distribución de las distancias de viaje
plt.figure(figsize=(12, 6))
sns.histplot(df['actual_distance_to_destination'], bins=50, kde=True, color = "crimson", edgecolor = "orange")
plt.title('Distribución de distancias de viaje\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Distancia del viaje (km)\n')
plt.ylabel('Frecuencia\n')
plt.show()

#Estas visualizaciones proporcionan una descripción general de las duraciones y distancias de los viajes en el conjunto de datos. A # continuación, analicemos los factores que afectan los 
#tiempos de entrega. Comenzaremos examinando la relación entre la duración del viaje y # la distancia.

#Aquí está la visualización que muestra la relación entre la duración del viaje y la distancia:
# Trazar la relación entre la duración del viaje y la distancia
plt.figure(figsize=(12, 6))
sns.scatterplot(x='actual_distance_to_destination', y='actual_time', data=df)
plt.title('Relación entre duración del viaje y distancia\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Distancia del viaje (km)\n')
plt.ylabel('Duración del viaje (minutos)\n')
plt.show()

#Este diagrama de dispersión nos ayuda a comprender cómo varía la duración del viaje con la distancia. A continuación, analicemos el impacto de los diferentes tipos de rutas en el viaje.

#Aquí está la visualización que muestra el impacto del tipo de ruta en la duración del viaje:
# Trazar el impacto del tipo de ruta en la duración del viaje
plt.figure(figsize=(12, 6))
sns.boxplot(x='route_type', y='actual_time', data=df)
plt.title('Impacto del tipo de ruta en la duración del viaje\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de ruta\n')
plt.ylabel('Duración del viaje (minutos)\n')
plt.xticks(rotation=45)
plt.show()

#A continuación, analicemos el efecto de los factores de corte en la duración del viaje. Esto nos ayudará a comprender cómo los tiempos límite # influyen en el rendimiento de la entrega.
# Trazar el impacto del factor de corte en la duración del viaje
plt.figure(figsize=(12, 6))
sns.boxplot(x='cutoff_factor', y='actual_time', data=df)
plt.title('Impacto del factor de corte en la duración del viaje\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Factor de corte\n')
plt.ylabel('Duración del viaje (minutos)\n')
plt.xticks(rotation=45)
plt.show()

#Esta es una visualización que muestra el impacto del factor de corte en la duración del viaje.

#Este diagrama de caja nos ayuda a comprender cómo los diferentes factores de corte influyen en la duración del viaje.

#A continuación, analicemos la eficiencia de los viajes comparando el tiempo real tomado con el tiempo estimado por OSRM (Open Source Routing # Machine). Este # nos ayudará a identificar 
#cualquier discrepancia y áreas de mejora en la optimización de rutas.

# Trazar la comparación entre el tiempo real y el tiempo estimado de OSRM
plt.figure(figsize=(12, 6))
sns.scatterplot(x='osrm_time', y='actual_time', data=df)
plt.title('Comparación entre el tiempo real y el tiempo estimado de OSRM\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tiempo estimado de OSRM (minutos)\n')
plt.ylabel('Tiempo real (minutos)\n')
plt.show()

# Este diagrama de dispersión nos ayuda a identificar cualquier discrepancia entre los tiempos reales y estimados, lo que puede ser útil para # mejorar la optimización de la ruta.

#A continuación, analicemos la eficiencia de los viajes comparando la distancia real recorrida con la distancia estimada de OSRM.

# Trazar la comparación entre la distancia real y la distancia estimada de OSRM
plt.figure(figsize=(12, 6))
sns.scatterplot(x='osrm_distance', y='actual_distance_to_destination', data=df)
plt.title('Comparación entre la distancia real y la distancia estimada de OSRM\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Distancia estimada de OSRM (km)\n')
plt.ylabel('Distancia real (km)')
plt.show()

#Esta es una visualización que compara la distancia real recorrida con la distancia estimada de OSRM.

#Este diagrama de dispersión nos ayuda a identificar cualquier discrepancia entre las distancias reales y estimadas, lo que puede resultar # útil para mejorar la optimización de la ruta.

#A continuación, analicemos la eficiencia del segmento comparando el tiempo real del segmento con el tiempo estimado de OSRM del segmento.

# Trazar la comparación entre el tiempo real del segmento y el tiempo estimado de OSRM del segmento
plt.figure(figsize=(12, 6))
sns.scatterplot(x='segment_osrm_time', y='segment_actual_time', data=df)
plt.title('Comparación entre el tiempo real del segmento y el tiempo estimado de OSRM del segmento\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tiempo estimado de OSRM del segmento (minutos)\n')
plt.ylabel('Tiempo real del segmento (minutos)\n')
plt.show()

# Trazar la comparación entre la distancia real del segmento y la distancia estimada de OSRM del segmento
plt.figure(figsize=(12, 6))
sns.scatterplot(x='segment_osrm_distance', y='segment_actual_time', data=df)
plt.title('Comparación entre la distancia real del segmento y la distancia estimada OSRM del segmento\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Segmento OSRM Distancia estimada (km)\n')
plt.ylabel('Distancia real del segmento (km)\n')
plt.show()

# Calcular discrepancias
df['time_discrepancy'] = df['actual_time'] - df['osrm_time']
df['distance_discrepancy'] = df['actual_distance_to_destination'] - df['osrm_distance']

# Realizar análisis de correlación
correlation_matrix = df[['time_discrepancy', 'distance_discrepancy', 'cutoff_factor', 'start_scan_to_end_scan', 'factor', 'segment_actual_time', 'segment_osrm_time', 'segment_osrm_distance', 'segment_factor']].corr()

# Mostrar la matriz de correlación
print('Matriz de correlación:')
print(correlation_matrix)

# Prepare los datos para el análisis de regresión
X = df[['cutoff_factor', 'start_scan_to_end_scan', 'factor', 'segment_actual_time', 'segment_osrm_time', 'segment_osrm_distance', 'segment_factor']]
y_time = df['time_discrepancy']
y_distance = df['distance_discrepancy']

# Divida los datos en conjuntos de entrenamiento y prueba
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X, y_time, test_size=0.2, random_state=42)
X_train_distance, X_test_distance, y_train_distance, y_test_distance = train_test_split(X, y_distance, test_size=0.2, random_state=42)

# Entrene el modelo de regresión lineal para la discrepancia temporal
model_time = LinearRegression()
model_time.fit(X_train_time, y_train_time)

# Predecir y evaluar el modelo para la discrepancia temporal
y_pred_time = model_time.predict(X_test_time)
mse_time = mean_squared_error(y_test_time, y_pred_time)
r2_time = r2_score(y_test_time, y_pred_time)

# Entrene el modelo de regresión lineal para la discrepancia de distancia
model_distance = LinearRegression()
model_distance.fit(X_train_distance, y_train_distance)

# Predecir y evaluar el modelo para la discrepancia de distancia
y_pred_distance = model_distance.predict(X_test_distance)
mse_distance = mean_squared_error(y_test_distance, y_pred_distance)
r2_distance = r2_score(y_test_distance, y_pred_distance)

# Mostrar los resultados de la evaluación
print('Modelo de discrepancia de tiempo:')
print('Error medio cuadrado:', mse_time)
print('R-cuadrado:', r2_time)

print('Modelo de discrepancia de distancia:')
print('Error medio cuadrado:', mse_distance)
print('R-cuadrado:', r2_distance)

# Servicios-logisticos
Análisis de operaciones logísticas de la empresa Delhibery

Se hizo un análisis avanzado y modelado predictivo para optimizar las operaciones logísticas de Delhivery, una empresa líder en la cadena de suministro de la India. El proyecto incluyó la evaluación de discrepancias en tiempos y distancias de entrega reales y estimados, con el objetivo de identificar áreas de mejora en la optimización de rutas y factores operativos.

**Preparación y limpieza de datos:**
- Manejo de valores faltantes mediante imputación con marcadores de posición.
- Conversión de marcas de tiempo a formato datetime para facilitar el análisis.
- Cálculo de variables clave como discrepancias entre tiempos y distancias reales y estimados.

**Análisis exploratorio de datos (EDA):**
- Visualización de distribuciones de duración y distancia de los viajes utilizando Seaborn y Matplotlib.
- Análisis de relaciones entre duración del viaje, distancia, tipo de ruta y factores de corte.
- Comparación de métricas reales con estimaciones generadas por OSRM (Open Source Routing Machine).

**Modelado predictivo:**
- Creación de modelos de regresión lineal para identificar factores clave que contribuyen a discrepancias de tiempos y distancias.
- Evaluación de modelos mediante métricas como R² y MSE (Error Cuadrático Medio).
- Interpretación de resultados para proponer mejoras operativas.

**Insights clave:**
- Identificación de factores críticos como "cutoff_factor" y "start_scan_to_end_scan" que explican más del 85% de la varianza en discrepancias de tiempos y distancias.
- Priorización de mejoras en tiempos límite y escaneos para optimizar entregas.

**Herramientas y tecnologías utilizadas:**
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, OSRM, Modelos de Regresión.

**Resultados obtenidos:**
- Desarrollo de modelos predictivos con alta precisión (R² > 0.85) para analizar discrepancias logísticas.
- Generación de insights prácticos que permiten a Delhivery optimizar rutas y tiempos de entrega, mejorando la eficiencia operativa.

# Clasificación Automática de Tickets con NLP

## Índice

- [Índice](#índice)
- [Introducción](#introducción)
  - [Métodos Utilizados](#métodos-utilizados)
  - [Tecnologías](#tecnologías)
- [Descarga y Configuración](#descarga-y-configuración)
  - [Requisitos Previos](#requisitos-previos)
  - [Cómo Ejecutar](#cómo-ejecutar)
- [Declaración del Problema](#declaración-del-problema)
  - [Objetivo General](#objetivo-general)
  - [Preparación de Datos:](#preparación-de-datos)
  - [Construcción y Evaluación del Modelo](#construcción-y-evaluación-del-modelo)
  - [Conclusiones](#conclusiones)
    - [Regresión Ridge](#regresión-ridge)
    - [Regresión Lasso](#regresión-lasso)
    - [Regresión ElasticNet](#regresión-lasso)
    - [Las Variables Más Significativas Son:](#las-variables-más-significativas-son)

## Introducción

Las quejas de los clientes en el sector financiero son un aspecto crucial para la mejora continua de los productos y servicios ofrecidos. Estas quejas no solo reflejan insatisfacciones, sino que también brindan valiosa información sobre posibles áreas de mejora. Según Kumar et al. (2018), la correcta gestión de las quejas no solo resuelve problemas inmediatos, sino que también permite construir una relación más sólida con los clientes, lo que lleva a una mayor lealtad y satisfacción. Resolver quejas de manera eficiente es clave para mantener la competitividad en un mercado tan dinámico como el financiero, donde los clientes buscan una atención al cliente rápida y eficaz.


### Métodos Utilizados
Recopilación de datos.
El conjunto de datos para el estudio está compuesto por 1,460 registros y 81 columnas, la primera columna Id, es un serial
numérico asignado a cada registro por lo que no se la toma en cuenta en el análisis. El detalle de las columnas y contenido
de las columnas se puede encontrar en el archivo "data_description.txt".

Análisis de la calidad de los datos:
Identificacion de datos faltantes,análisis de calidad

Análisis Exploratorio de Datos (EDA):
Análisis de la variable objetivo: Salesprice
Análisis de variables numéricas
Análisis de Correlación
Análisis de variables categóricas

Desarrollo de Modelos Estadisticos:
Regresion Lineal 
Ridge
Lasso
Elasticnet

### Tecnologías
* Python
* Pandas
* Numpy 
* Matplotlib
* Pylab 
* Seaborn 
* Sklearn
* Statsmodels

## Descarga y Configuración
### Requisitos Previos

Este proyecto necesita que Anaconda esté instalado en la computadora.
Para más detalles sobre la instalación, visite: https://docs.anaconda.com/anaconda/install/index.html


### Cómo Ejecutar

Puede descargar el código fuente clonando este repositorio usando Git:

1. Abra su aplicación Terminal favorita (Unix, Linux o Macos), como Terminal, Comando, Consola, iTerm2, etc.

2. Clone el repositorio

```
git clone <GITHUB_REPO_URL>
```

3. Abra el archivo notebook ** MDSv5_ML_P1_Regresion Avanzada.ipynb** en Anaconda.

```
jupyter notebook <MDSv5_ML_P1_Regresion Avanzada.ipynb>
```


## Declaración del Problema

En el competitivo mercado inmobiliario australiano, los precios de las propiedades pueden variar ampliamente en
función de factores económicos, sociales y específicos del inmueble, como la ubicación, el tamaño, la antigüedad 
y el estado de mantenimiento. Para capitalizar estas variaciones de manera rentable, la empresa busca identificar
propiedades cuyo valor actual esté por debajo de su valor de mercado real, lo que representa una oportunidad de 
compra estratégica. 
Esta necesidad de predecir con precisión los precios surge de la intención de invertir únicamente en propiedades 
que, según un análisis, ofrecen el potencial de ser revendidas con una ganancia considerable. Con un conjunto de
datos disponible de ventas y la aplicación de modelos estadísticos avanzados, la empresa pretende no solo evitar 
compras de alto riesgo, sino también optimizar su portafolio de inversiones para maximizar el retorno


### Objetivo General

Desarrollar un modelo estadístico predictivo de regresión avanzada para estimar con mayor precisión el precio de
venta de propiedades residenciales en el mercado australiano, con el fin de identificar oportunidades de inversión 
en las que los inmuebles se encuentren subvalorados con respecto a su valor real.

### Preparación de Datos:

1. Entendimiento de los Datos
2. Manipulación y limpieza de Datos.
2.1 Dropping-Data
2.2 Derived-Data
3. Analisis Exploratorio de Datos: (EDA).
3.1 Plot numerical data
3.2 Plot categorical data
4. Tratamiento de variables categóricas 
4.1 Calculo los valores faltantes
4.2 Identificion de outliers

### Construcción y Evaluación del Modelo

1. Primer modelo solo con variables numericas 
1.1 División de datos de entrenamiento y prueba.
1.2 Ajustar el modelo 
1.3 Preparación del modelo usando Regresión Lineal
1.4 Prediccion y evaluacion

2. Segundo modelo considerando las variables numéricas y categóricas fundamentales
2.1 Realizar One-Hot Encoding para las variables categóricas, convertimos a DUMMY
2.2 División de datos de entrenamiento y prueba.
2.3 Ajustar el modelo 
2.4 Preparación del modelo usando Regresión Lineal
2.5 Prediccion y evaluacion

3. Analisis de multicolinealidad modelo numéricas y categóricas
4. Conclusiones de multicolinealidad - Método del Factor Inflador de Varianza (VIF)

5. Tercer modelo Ridge
5.1 División de datos de entrenamiento y prueba.
5.2 Ajustar el modelo 
5.3 Preparación del modelo usando ridge_model.predict
5.4 Prediccion y evaluacion
5.5 Prueba de multicolinealidad para el modelo de RIDGE correccion del modelo

6. Tercer modelo Lasso
6.1 División de datos de entrenamiento y prueba.
6.2 Ajustar el modelo 
6.3 Preparación del modelo usando lasso_model.predict
6.4 Prediccion y evaluacion
6.5 Prueba de multicolinealidad para el modelo de LASSO correccion del modelo

7. Tercer modelo ElasticNet
7.1 División de datos de entrenamiento y prueba.
7.2 Ajustar el modelo 
7.3 Preparación del modelo usando ElasticNet
7.4 Prediccion y evaluacion
7.5 Prueba de multicolinealidad para el modelo de ElasticNet correccion del modelo

8. Análisis de Precision seleccion del mejor modelo.
7. Evaluación y Valoración del Modelo Lasso.
8. Predicción del modelo Lasso.
9. Conclusión y Análisis Final.

### Conclusiones

### Conclusions
R2_Score modelo solo con variables numericas 0.8288, el modelo explica aproximadamente el 82.88% de la variabilidad total en 
   los precios de venta de las propiedades
R2_Score variables numéricas y categóricas fundamentales 0.8890 indica que el modelo explica ahora el 88.90% de la variabilidad 
   en el precio de venta. Esto indica que las variables seleccionadas numéricas más las categorías son efectivas para
   predecir el precio de venta. Sin embargo, el 11.1% restante de la variabilidad en el precio no es explicado por el modelo.
R2_Score for Ridge regresion 0.7667 indica que el modelo Ridge explica el 76.67% de la variabilidad en los precios de venta 
   de las propiedades en el conjunto de prueba.  
R2_Score for Lasso regresion 0.8891, lo que indica que el modelo Lasso explica aproximadamente el 88.91% de la variabilidad 
   en los precios de venta (SalesPrice). Esto implica que las variables seleccionadas tras la regularización capturan muy bien
   la relación con el precio, manteniendo una capacidad predictiva alta.
R2_Score for ElasticNet regresion 0.8923 implica que el modelo explica el 89.23% de la variabilidad en el precio de venta de 
   las propiedades.

#### Ridge Regression
* **Optimal Lambda Value:100.0 ##
* **R2 :** 0.7667##
* **RMSE :** 32445.60.##

#### Lasso Regression
* **Optimal Lambda Value:** 0.1####
* **R2 :**  0.8890##
* **RMSE :**       23289.56##

#### ElasticNet Regression
* **Optimal Lambda Value:** 0.1####
* **R2 :**  0.8926##
* **RMSE :**       22911.34##

#### Las Variables Más Significativas Son:
Positivas
* MSZoning_FV
* SaleCondition_Alloca
* SaleType_Otros
* Neighborhood_StoneBr
* Neighborhood_Veenker
Negativas
* RoofStyle_Hip
* RoofStyle_Gable
* RoofStyle_Gambrel
* Functional_Otros
* RoofMatl_Otros

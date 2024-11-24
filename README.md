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
En el contexto de la empresa financiera, las quejas a menudo se presentan en forma de datos no estructurados en tickets de atención. Estos tickets incluyen una variedad de problemas que los clientes experimentan con los productos, como tarjetas de crédito, servicios bancarios y préstamos. Según Zhang et al. (2017), la presencia de datos textuales no estructurados puede representar un desafío para las empresas, ya que su análisis manual es intensivo en recursos y tiempo, lo que retrasa la capacidad de respuesta. Esto genera una sobrecarga de trabajo para el personal de atención, lo que puede afectar negativamente la eficiencia y la experiencia del cliente.
Con el crecimiento de las empresas y la expansión de su base de clientes, la carga de trabajo relacionada con el manejo de quejas aumenta. En este contexto, el uso de tecnología para automatizar el proceso de clasificación de quejas es fundamental. Pérez et al. (2019) destacan que la implementación de sistemas automatizados que pueden categorizar y priorizar tickets de queja permite a las empresas no solo reducir la carga de trabajo manual, sino también mejorar la precisión y velocidad en la resolución de problemas. La automatización, a través de técnicas como el procesamiento de lenguaje natural (PLN), puede transformar los datos no estructurados en información útil de manera mucho más eficiente que los métodos tradicionales.
Además, la automatización del análisis de quejas tiene el potencial de proporcionar información en tiempo real sobre las áreas de mejora de los productos y servicios. Según Batra et al. (2020), los sistemas automatizados de clasificación de quejas no solo permiten una respuesta más rápida, sino que también generan datos valiosos para la mejora continua. Estos sistemas pueden identificar tendencias en las quejas de los clientes, lo que ayuda a la empresa a anticipar problemas antes de que escalen. De esta manera, la implementación de soluciones basadas en inteligencia artificial y análisis de datos en la gestión de quejas se convierte en un factor clave para el éxito a largo plazo de la empresa financiera.

## Situación del problema
La empresa financiera tiene acumulación y el manejo manual de tickets de atención al cliente, estos tickets contienen quejas y solicitudes de los clientes, usualmente redactadas en lenguaje natural y relacionadas con diversos productos y servicios como tarjetas de crédito, banca y préstamos/hipotecas.
El proceso tradicional requiere que múltiples empleados analicen y clasifiquen manualmente cada ticket, lo que resulta en un uso intensivo de recursos humanos, tiempos prolongados de respuesta y un margen de error significativo en la categorización. A medida que la base de clientes crece, la cantidad de tickets aumenta exponencialmente, haciendo que este enfoque manual se vuelva ineficiente y afecte la capacidad de la empresa para atender rápidamente a los clientes, poniendo en riesgo su satisfacción y fidelidad.
Además, la falta de un sistema automatizado limita la capacidad de la empresa para identificar patrones y problemas recurrentes en sus productos y servicios, lo que impide la implementación de mejoras proactivas. Este enfoque reactivo no solo dificulta la resolución ágil de quejas, sino que también reduce la capacidad de la organización para obtener ventajas competitivas mediante la innovación basada en las necesidades de los clientes.
En resumen, la empresa enfrenta un desafío crítico: optimizar el manejo de tickets para garantizar la satisfacción del cliente mientras minimiza costos operativos y mejora su capacidad de respuesta.

### Métodos Utilizados
Recopilación de datos.
El conjunto de datos para el estudio está compuesto por 78,313 registros y 22 columnas, la data esta almacenada en una base de datos en formato JSON, por lo que, para su tratamiento y análisis, necesitamos convertirlos a un formato de dataframe, utilizando para ese fin la librería REQUEST de Python.

Análisis de la calidad de los datos:
Identificacion de datos faltantes,análisis de calidad

Análisis Exploratorio de Datos (EDA):
Renombrado de columnas
Preparación del texto para el modelado (conversión a minúsculas)
Lematización y extración de POS tags.
Análisis de exploratorio del texto (longitud de carácteres, palabras más frecuentes, nube de palabras, unigramas, bigramas y trigramas)

Desarrollo de Modelos:
Modelo de Regresión Logística
Modelo de Árbol de Decisión
Modelo Random Forest
Modelo de Naive Bayes

### Tecnologías
* Python
* Pandas
* Numpy
* NLTK
* Spacy
* Matplotlib
* Plotly
* Seaborn 
* Wordcloud
* Sklearn
* 
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

La empresa financiera tiene acumulación y el manejo manual de tickets de atención al cliente, estos tickets contienen quejas y solicitudes de los clientes, usualmente redactadas en lenguaje natural y relacionadas con diversos productos y servicios como tarjetas de crédito, banca y préstamos/hipotecas.
El proceso tradicional requiere que múltiples empleados analicen y clasifiquen manualmente cada ticket, lo que resulta en un uso intensivo de recursos humanos, tiempos prolongados de respuesta y un margen de error significativo en la categorización. A medida que la base de clientes crece, la cantidad de tickets aumenta exponencialmente, haciendo que este enfoque manual se vuelva ineficiente y afecte la capacidad de la empresa para atender rápidamente a los clientes, poniendo en riesgo su satisfacción y fidelidad.
Además, la falta de un sistema automatizado limita la capacidad de la empresa para identificar patrones y problemas recurrentes en sus productos y servicios, lo que impide la implementación de mejoras proactivas. Este enfoque reactivo no solo dificulta la resolución ágil de quejas, sino que también reduce la capacidad de la organización para obtener ventajas competitivas mediante la innovación basada en las necesidades de los clientes.
En resumen, la empresa enfrenta un desafío crítico: optimizar el manejo de tickets para garantizar la satisfacción del cliente mientras minimiza costos operativos y mejora su capacidad de respuesta.


### Objetivo General

Desarrollar un modelo basado en técnicas de Procesamiento de Lenguaje Natural, para clasificar de manera eficiente y precisa las quejas de los clientes en cinco categorías principales: tarjetas de crédito/tarjetas de prepago, servicios de cuentas bancarias, reporte de robo/disputa, hipotecas/préstamos y otros, de tal forma que este modelo permitirá optimizar el sistema de tickets de atención al cliente, facilitando la identificación rápida de problemas, mejorando la oferta de servicios y fortaleciendo la capacidad de respuesta de la empresa Financiera.

### Preparación de Datos:

1. Recopilación de datos.
2. Análisis Exploratorio de Datos (EDA).
2.1 Renombrar columnas
2.2 Preparación del texto para el modelado
2.3 Lematización y extracción de POS tags
2.4 Análisis exploratorio de datos para famliarizarse con la información
3. Modelado mediante Non-Negative Matrix Factorization (NMF).
3.1 Definición del mejor número de tópicos
3.1 Topicos establecidos

### Construcción y Evaluación del Modelo

1. Modelo de Regresión Logística 
1.1 División de datos de entrenamiento y prueba.
1.2 Ajustar el modelo 
1.3 Preparación del modelo usando Regresión Lineal
1.4 Prediccion y evaluacion


1. Modelo de Árbol de Decisión 
1.1 División de datos de entrenamiento y prueba.
1.2 Ajustar el modelo 
1.3 Preparación del modelo usando Regresión Lineal
1.4 Prediccion y evaluacion


1. Modelo Random Forest
1.1 División de datos de entrenamiento y prueba.
1.2 Ajustar el modelo 
1.3 Preparación del modelo usando Regresión Lineal
1.4 Prediccion y evaluacion

1. Modelo Naive Bayes 
1.1 División de datos de entrenamiento y prueba.
1.2 Ajustar el modelo 
1.3 Preparación del modelo usando Regresión Lineal
1.4 Prediccion y evaluacion












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

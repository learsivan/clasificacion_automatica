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
    - [Regresión Logística](#regresión-logística)
    - [Árbol de Decisión](#árbol-de-decisión)
    - [Random Forest](#random-forest)
    - [Naive Bayes:](#naive-bayes)

## Introducción

En un entorno empresarial competitivo, la satisfacción del cliente es fundamental, y una gestión eficiente de las quejas es clave para lograrla. Este proyecto busca desarrollar un modelo de clasificación que organice automáticamente las quejas de los clientes según los productos o servicios relacionados. Al categorizar los tickets de manera precisa, las empresas pueden priorizar y redirigir rápidamente cada caso al equipo adecuado, mejorando los tiempos de respuesta y optimizando la resolución de problemas. Este enfoque no solo eleva la eficiencia operativa, sino que también refuerza la confianza del cliente al demostrar un compromiso proactivo con sus necesidades.

Las quejas de los clientes en el sector financiero son un aspecto crucial para la mejora continua de los productos y servicios ofrecidos. Estas quejas no solo reflejan insatisfacciones, sino que también brindan valiosa información sobre posibles áreas de mejora. Según Kumar et al. (2018), la correcta gestión de las quejas no solo resuelve problemas inmediatos, sino que también permite construir una relación más sólida con los clientes, lo que lleva a una mayor lealtad y satisfacción. Resolver quejas de manera eficiente es clave para mantener la competitividad en un mercado tan dinámico como el financiero, donde los clientes buscan una atención al cliente rápida y eficaz.

Las quejas a menudo se presentan en forma de datos no estructurados en tickets de atención. Estos tickets incluyen una variedad de problemas que los clientes experimentan con los productos. Según Zhang et al. (2017), la presencia de datos textuales no estructurados puede representar un desafío para las empresas, ya que su análisis manual es intensivo en recursos y tiempo, lo que retrasa la capacidad de respuesta. A medida que la empresa crece el reto del procesamiento de dtos aumenta por el incremento en la carga de trabajo respecto al manejo de quejas. Pérez et al. (2019) destacan que la implementación de sistemas automatizados que pueden categorizar y priorizar tickets de queja permite a las empresas no solo reducir la carga de trabajo manual, sino también mejorar la precisión y velocidad en la resolución de problemas. La automatización, a través de técnicas como el procesamiento de lenguaje natural (PLN), puede transformar los datos no estructurados en información útil de manera mucho más eficiente que los métodos tradicionales.

### Métodos Utilizados
Recopilación de datos.
El conjunto de datos para el estudio está compuesto por 78,313 registros y 22 columnas, la data esta almacenada en una base de datos en formato JSON ("complaints.json"), por lo que, para su tratamiento y análisis, se debe convertir a un formato de dataframe, utilizando para ese fin la librería REQUEST de Python.

Análisis Exploratorio de Datos (EDA):
* Renombrado de columnas.
* Preparación del texto para el modelado (conversión a minúsculas).
* Lematización y extración de POS tags.
* Análisis de exploratorio del texto (longitud de carácteres, palabras más frecuentes, nube de palabras, unigramas, bigramas y trigramas).

Desarrollo de Modelos:
* Modelo de Regresión Logística.
* Modelo de Árbol de Decisión.
* Modelo Random Forest.
* Modelo de Naive Bayes.

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

3. Abra el archivo notebook ** Proyecto_2_NLP_Clasificacion_Automatica_de_Tickets.ipynb** en Anaconda.

```
jupyter notebook <Proyecto_2_NLP_Clasificacion_Automatica_de_Tickets.ipynb>
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

### Construcción y Evaluación de los Modelos

1. Modelo de Regresión Logística 
1.1 Entrenamiento y testeo
1.2 Aplicación
1.3 Evaluación

2. Modelo de Árbol de Decisión 
2.1 Entrenamiento y testeo
2.2 Aplicación
2.3 Evaluación

3. Modelo Random Forest
3.1 Entrenamiento y testeo
3.2 Aplicación
3.3 Evaluación

4. Modelo Naive Bayes 
4.1 Entrenamiento y testeo
4.2 Aplicación
4.3 Evaluación

### Conclusiones

La regresión logística fue seleccionada como el mejor modelo debido a su desempeño superior y su adecuación al problema de clasificación de texto procesado, respaldado por lo siguiente: 

* Desempeño métrico sobresaliente: La precisión fue del 96.58%, mostrando que clasifica correctamente la mayoría de las categorías. El F1-Score de 96.57% muestra un equilibrio entre precisión y recall. Las métricas asociadas (precisión, recall y F1-score) están alineadas, lo que confirma un rendimiento robusto y consistente.

* Adecuación al problema: La Regresión Logística es un modelo lineal que funciona bien con datos representados en vectores dispersos de alta dimensionalidad, como los obtenidos mediante la transformación TF-IDF en este caso. La clasificación de texto (considerando que las categorías estan bien definidas), es linealmente separable, por lo que se ajusta a las capacidades del modelo.

* Simplicidad y eficiencia: Computacionalmente es mas eficiente que los otros modelos, especialmente con un conjunto grande de datos; asimismo, el tiempo de entrenamiento y predicción es màs rápido. Los coeficientes del modelo proporcionan una manera clara de entender la importancia de las características en la clasificación, lo cual facilita la generación se insights adicionales.

* Escalabilidad y producción: Es altamente escalable y fácil de integrar en sistemas productivos. Su capacidad para manejar grandes cantidades de datos y realizar predicciones rápidamente la hace adecuada para escenarios donde el número de tickets puede crecer significativamente.

La Regresión Logística fue seleccionada como el modelo final debido a su desempeño superior en todas las métricas relevantes, además de su eficiencia computacional e interpretabilidad. Este modelo demostró ser adecuado para la clasificación de tickets en tiempo real, cumpliendo con el objetivo del proyecto.

En un escenario de implementación del modelo propuesto, éste proporcionará a la empresa financiera una herramienta robusta para automatizar la clasificación de quejas de los clientes. Esto optimiza los tiempos de respuesta, reduce la carga de trabajo manual y mejora la experiencia del cliente. Además, el enfoque propuesto puede ser actualizado con nuevos datos para mantener su relevancia y efectividad a lo largo del tiempo.

#### Regresión Logística
* **Exactitud :** 96.58%##
* **F1-Score :** 96.57%##

#### Árbol de Decisión
* **Exactitud :** 82.37%##
* **F1-Score :** 82.36%##

#### Random Forest
* **Exactitud :** 87.25%##
* **F1-Score :** 87.18%##

#### Naive Bayes
* **Exactitud :** 76.67%
* **F1-Score :** 75.63%

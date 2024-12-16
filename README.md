# 📊 **Prueba técnica**

## 📖 **Descripción**
Nos ha contratado una plataforma de streaming para mejorar la calidad de su contenido y la experiencia de sus usuarios. Este proyecto tiene como objetivo aplicar técnicas de análisis de datos para identificar las películas y cortometrajes más populares y mejor valorados desde 1990 hasta la fecha, además de desarrollar un sistema de recomendación para personalizar la experiencia de los usuarios.

Objetivos principales:
Identificar las películas y cortometrajes más destacados en la plataforma basándose en calificaciones, número de visualizaciones y reseñas.
Analizar la evolución de las preferencias de los usuarios a lo largo de los años para identificar tendencias clave en la industria cinematográfica.
Diseñar y desarrollar un sistema de recomendación para sugerir contenido relevante a los usuarios.
Proporcionar recomendaciones específicas para la promoción de contenido en las distintas secciones de la plataforma.
La documentación a la API la encontrarás aquí.
2. Fases del Proyecto
Fase 1: Extracción de Datos de API de MoviesDataset
En esta fase se utilizará la API de MoviesDataset para extraer información relevante sobre películas y cortometrajes.

Requerimientos:

Películas desde 1990 hasta la actualidad.
Géneros: Drama, Comedy, Action, Fantasy, Horror, Mystery, Romance, Thriller.
Información necesaria:
Tipo (corto o película).
Nombre.
Año y mes de estreno.
ID de la película.
Nota: Los datos extraídos deberán almacenarse en una lista de tuplas.

Fase 2: Extracción de Detalles de Películas con Selenium
Utiliza Selenium para obtener información adicional de las películas listadas previamente.

Información requerida:

Calificación de IMDB.
Dirección (director o directores).
Guionistas.
Argumento.
Duración (en minutos).
Nota: Los datos obtenidos deberán almacenarse en una lista de tuplas.

Fase 3: Extracción de Detalles de Actores con Selenium
Obtén información sobre los 10 actores principales de cada película o corto utilizando Selenium.

Información requerida:

Nombre.
Año de nacimiento.
Por qué es conocido.
Roles (actuación, dirección, etc.).
Premios.
Nota: La información deberá almacenarse en una lista de tuplas.

Fase 4: Creación de un Sistema de Recomendación
Diseña y desarrolla un sistema de recomendación basado en los datos recopilados.

Modelos recomendados:

Modelo colaborativo: Basado en las preferencias de los usuarios.
Modelo basado en contenido: Considerando géneros, directores y calificaciones.
El sistema debe permitir sugerir películas personalizadas para diferentes tipos de usuarios.

Fase 5: Creación de una Base de Datos
Organiza toda la información recopilada en una base de datos SQL bien estructurada. Define las tablas y relaciones necesarias para almacenar los datos de manera eficiente.

Fase 6: Inserción de Datos en la Base de Datos
Inserta todos los datos recopilados en la base de datos diseñada.

---

## 🗂️ **Estructura del Proyecto**
```plaintext
├── data/                # Datos crudos y procesados
├── notebooks/           # Notebooks de Jupyter con análisis y visualizaciones
├── src/                 # Scripts de procesamiento y modelado
├── results/             # Gráficos y reportes finales
├── README.md            # Descripción del proyecto
```

---

## 🛠️ **Instalación y Requisitos**
Este proyecto utiliza **Python 3.8 o superior**. Las dependencias necesarias son:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [scikit-learn-extra](https://scikit-learn.org/stable/)


## 🧑‍💻 **Análisis Realizado**
En proceso de redacción pendiente

---

## 📊 **Resultados y Conclusiones**
En proceso de redacción pendiente

---

## 🔄 **Próximos Pasos**
- Incluir más datos históricos y externos para mejorar los modelos.
- Implementar técnicas avanzadas de feature engineering.
- Explorar el impacto de campañas de marketing en los clusters menos rentables.

---


## 🤝 **Contribuciones**
Las contribuciones son bienvenidas. Si deseas mejorar el proyecto, por favor abre un pull request o una issue en este repositorio.

--- 

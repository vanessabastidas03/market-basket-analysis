# 🛒 Market Basket Analysis - Proyecto Final de Minería de Datos

Aplicación web interactiva que aplica técnicas de minería de datos para identificar patrones de compra mediante reglas de asociación, utilizando los algoritmos **Apriori** y **FP-Growth**.

## 📋 Descripción

Este proyecto implementa un análisis de canasta de mercado sobre el dataset **Groceries Dataset** de Kaggle, siguiendo la metodología **CRISP-DM**. Permite identificar productos que frecuentemente se compran juntos, generando reglas de asociación útiles para estrategias comerciales.

## 🎯 Objetivos

- Aplicar el algoritmo **Apriori** para minería de reglas de asociación.
- Aplicar el algoritmo **FP-Growth** como alternativa eficiente.
- Calcular y comparar métricas: **support, confidence y lift**.
- Visualizar reglas de asociación de forma interactiva.
- Generar combinaciones de productos recomendadas.

## 🛠️ Tecnologías

- **Python 3.10+**
- **Streamlit** - Dashboard interactivo
- **mlxtend** - Algoritmos Apriori y FP-Growth
- **Pandas / NumPy** - Manipulación de datos
- **Plotly / Matplotlib / Seaborn** - Visualizaciones
- **NetworkX** - Grafos de reglas

## 🚀 Instalación local

```bash
# Clonar repositorio
git clone https://github.com/TU-USUARIO/market-basket-analysis.git
cd market-basket-analysis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run streamlit_app.py
```

## 📊 Dataset

**Groceries Dataset** - Kaggle  
https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset

- 38,765 transacciones
- 167 productos únicos
- Período: 2014-2015

## 📁 Estructura del proyecto
market-basket-analysis/
├── data/                  # Datasets crudos y procesados
├── src/                   # Código fuente (modelos y utilidades)
├── pages/                 # Páginas del dashboard Streamlit
├── notebooks/             # Análisis exploratorio
├── docs/                  # Documentación CRISP-DM
└── streamlit_app.py       # Punto de entrada

## 📈 Metodología CRISP-DM

El proyecto sigue las 6 fases de CRISP-DM:

1. **Comprensión del negocio**
2. **Comprensión de los datos**
3. **Preparación de los datos**
4. **Modelado**
5. **Evaluación**
6. **Despliegue**

Ver detalles completos en [`docs/CRISP-DM.md`](docs/CRISP-DM.md).

## 👤 Autor

Vanessa Bastidas - Proyecto Final Minería de Datos


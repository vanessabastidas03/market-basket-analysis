"""
Punto de entrada principal del dashboard de Market Basket Analysis.
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import run_preprocessing
from src.utils import load_processed_data

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Procesar datos si no existen
processed_file = Path('data/processed/transactions_encoded.pkl')
if not processed_file.exists():
    with st.spinner('Procesando datos por primera vez...'):
        run_preprocessing()

# Cargar datos
try:
    df = load_processed_data()
except FileNotFoundError:
    st.error("No se encontró el dataset. Verifica que `data/raw/Groceries_dataset.csv` exista.")
    st.stop()

# --- ENCABEZADO ---
st.title("Market Basket Analysis")
st.markdown("#### Análisis de patrones de compra mediante reglas de asociación")
st.markdown("---")

# --- INTRODUCCIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Descripción del proyecto

    Este dashboard implementa técnicas de **minería de datos** para identificar patrones
    de compra en un supermercado. A partir de transacciones históricas, se generan
    **reglas de asociación** que permiten descubrir qué productos tienden a adquirirse
    conjuntamente.

    Se aplican dos algoritmos clásicos de minería de itemsets frecuentes:

    - **Apriori**: algoritmo fundamental basado en la generación iterativa de candidatos.
      Es didáctico y ampliamente documentado en la literatura.
    - **FP-Growth**: alternativa eficiente que utiliza una estructura de árbol (FP-tree),
      eliminando la necesidad de generar candidatos de forma explícita.

    #### Métricas de evaluación de reglas

    | Métrica | Definición |
    |---------|-----------|
    | **Support** | Proporción de transacciones que contienen el itemset. |
    | **Confidence** | Probabilidad condicional P(B | A): que B se compre dado que A fue comprado. |
    | **Lift** | Razón entre la confianza observada y la esperada bajo independencia. Lift > 1 indica asociación positiva. |

    #### Metodología

    El proyecto sigue las fases de **CRISP-DM** (Cross-Industry Standard Process for Data Mining),
    desde la comprensión del negocio hasta la evaluación y presentación de resultados.
    """)

with col2:
    st.info("""
    **Dataset: Groceries Dataset**

    Fuente: Kaggle

    - Periodo: 2014 — 2015
    - Productos únicos: 167
    - Clientes únicos: ~3.900
    - Transacciones: ~14.000
    """)

st.markdown("---")

# --- KPIs PRINCIPALES ---
st.subheader("Resumen estadístico del dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transacciones", f"{df.shape[0]:,}")
col2.metric("Productos únicos", f"{df.shape[1]:,}")
col3.metric("Items totales", f"{int(df.sum().sum()):,}")
col4.metric("Items por transacción (promedio)", f"{df.sum(axis=1).mean():.2f}")

st.markdown("---")

# --- NAVEGACIÓN ---
st.subheader("Estructura del dashboard")
st.markdown("""
Utilice el menú lateral para navegar entre las secciones del proyecto:

1. **Exploración de datos** — Análisis exploratorio: distribuciones, frecuencias y correlaciones entre productos.
2. **Reglas Apriori** — Generación y visualización de reglas de asociación con el algoritmo Apriori.
3. **Reglas FP-Growth** — Generación y visualización de reglas de asociación con el algoritmo FP-Growth.
4. **Comparación** — Análisis comparativo empírico de ambos algoritmos (tiempo de ejecución y resultados).
5. **Generador de recomendaciones** — Sistema de recomendación de productos basado en reglas de asociación.
""")

st.markdown("---")
st.markdown("**Autora:** Vanessa Bastidas &nbsp;|&nbsp; Proyecto Final — Minería de Datos")

# --- SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.markdown("**Autora:** Vanessa Bastidas")
st.sidebar.markdown("Proyecto Final — Minería de Datos")
st.sidebar.markdown("---")
st.sidebar.markdown("[Dataset en Kaggle](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)")

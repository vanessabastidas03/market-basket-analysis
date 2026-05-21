"""
Punto de entrada principal del dashboard de Market Basket Analysis.
"""
import streamlit as st
from pathlib import Path
import sys

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import run_preprocessing
from src.utils import load_processed_data

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Procesar datos si no existen
processed_file = Path('data/processed/transactions_encoded.pkl')
if not processed_file.exists():
    with st.spinner('⏳ Procesando datos por primera vez... (esto solo ocurre una vez)'):
        run_preprocessing()

# Cargar datos
try:
    df = load_processed_data()
except FileNotFoundError:
    st.error("❌ No se encontró el dataset. Verifica que `data/raw/Groceries_dataset.csv` exista.")
    st.stop()

# --- HEADER ---
st.title("🛒 Market Basket Analysis")
st.markdown("### Análisis de patrones de compra con reglas de asociación")
st.markdown("---")

# --- INTRODUCCIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 📖 Sobre el proyecto
    
    Este dashboard implementa **técnicas de minería de datos** para identificar 
    productos que frecuentemente se compran juntos en un supermercado. 
    
    Se utilizan dos algoritmos clásicos de **reglas de asociación**:
    
    - **🔍 Apriori**: algoritmo fundamental, genera candidatos iterativamente.
    - **⚡ FP-Growth**: alternativa eficiente basada en árboles FP-tree.
    
    ### 🎯 Métricas calculadas
    
    - **Support**: frecuencia del itemset en todas las transacciones.
    - **Confidence**: probabilidad de que B se compre si A se compra.
    - **Lift**: cuánto aumenta la probabilidad de B cuando A está presente.
    
    ### 📚 Metodología
    
    Se sigue **CRISP-DM** (Cross-Industry Standard Process for Data Mining) en sus 6 fases.
    """)

with col2:
    st.info("""
    ### 📊 Dataset
    
    **Groceries Dataset** (Kaggle)
    
    - 📅 Período: 2014-2015
    - 🛍️ Productos únicos: 167
    - 👥 Clientes únicos: ~3,900
    - 🧾 Transacciones: ~14,000
    """)

st.markdown("---")

# --- KPIs PRINCIPALES ---
st.subheader("📈 Resumen del dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🧾 Transacciones", f"{df.shape[0]:,}")
col2.metric("🛍️ Productos únicos", f"{df.shape[1]:,}")
col3.metric("📦 Ítems totales", f"{int(df.sum().sum()):,}")
col4.metric("📊 Ítems/transacción", f"{df.sum(axis=1).mean():.2f}")

st.markdown("---")

# --- NAVEGACIÓN ---
st.subheader("🧭 Navegación")
st.markdown("""
Usa el menú lateral para navegar entre las páginas:

1. **📊 Exploración**: Análisis exploratorio del dataset.
2. **🔍 Reglas Apriori**: Reglas generadas con Apriori.
3. **⚡ Reglas FP-Growth**: Reglas generadas con FP-Growth.
4. **⚖️ Comparación**: Apriori vs FP-Growth.
5. **🛒 Generador**: Recomendador de productos.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Autor")
st.sidebar.markdown("Proyecto Final - Minería de Datos")
st.sidebar.markdown("### 🔗 Enlaces")
st.sidebar.markdown("[📦 Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)")
st.sidebar.markdown("[💻 GitHub](https://github.com/)")
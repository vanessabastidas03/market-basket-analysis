"""
Página de Análisis Exploratorio de Datos.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_processed_data, load_transactions_list, get_top_products

st.set_page_config(page_title="Exploración de Datos", page_icon=None, layout="wide")

# --- ENCABEZADO ---
st.title("Análisis Exploratorio de Datos")
st.markdown("""
Esta sección corresponde a la **Fase 2 de CRISP-DM: Comprensión de los Datos**.
El objetivo es caracterizar el dataset *Groceries* para fundamentar las decisiones
de parametrización de los algoritmos de minería de reglas de asociación.

Se analizan las siguientes dimensiones:
- Frecuencia de compra por producto.
- Distribución del tamaño de las transacciones.
- Cobertura acumulada de productos (principio de Pareto).
- Correlaciones entre los productos más frecuentes.
""")
st.markdown("---")

# Cargar datos
df = load_processed_data()
transactions = load_transactions_list()

# --- KPIs GENERALES ---
st.subheader("Indicadores generales del dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transacciones", f"{df.shape[0]:,}")
col2.metric("Productos únicos", f"{df.shape[1]:,}")
col3.metric("Items totales", f"{int(df.sum().sum()):,}")
col4.metric("Items por transacción (promedio)", f"{df.sum(axis=1).mean():.2f}")

st.info(
    "El promedio de items por transacción indica el tamaño típico de una canasta de compra. "
    "Un valor bajo sugiere un dataset disperso, lo que implica usar soportes mínimos bajos "
    "al aplicar los algoritmos de minería de datos."
)

st.markdown("---")

# --- TOP PRODUCTOS ---
st.subheader("Productos más frecuentes")
st.markdown("""
El siguiente gráfico muestra los productos con mayor frecuencia de aparición en las transacciones
del dataset. Los productos ubicados en la parte superior del gráfico son los que más clientes
adquieren y, por tanto, son candidatos naturales a aparecer como consecuentes en las reglas de
asociación generadas.
""")

top_n = st.slider("Número de productos a visualizar:", 5, 50, 20)
top_products = get_top_products(df, top_n)

fig = px.bar(
    x=top_products.values,
    y=top_products.index,
    orientation='h',
    labels={'x': 'Frecuencia absoluta', 'y': 'Producto'},
    title=f"Top {top_n} productos más comprados",
    color=top_products.values,
    color_continuous_scale='Viridis'
)
fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
st.plotly_chart(fig, width='stretch')

st.success(
    f"El producto más frecuente es **{top_products.index[0]}**, "
    f"presente en {top_products.iloc[0]:,} transacciones del dataset."
)

st.markdown("---")

# --- DISTRIBUCIÓN DEL TAMAÑO DE TRANSACCIONES ---
st.subheader("Distribución del tamaño de las transacciones")
st.markdown("""
Este gráfico muestra la distribución del número de productos por transacción (canasta).
Permite comprender la estructura del dataset:

- Si la mayoría de transacciones contienen pocos productos (1-3 items), el dataset es **disperso**,
  lo que requiere usar valores bajos de `min_support`.
- Transacciones grandes generan potencialmente más reglas, pero también mayor ruido estadístico.

La tabla adjunta presenta las estadísticas descriptivas de esta distribución.
""")

transaction_sizes = [len(t) for t in transactions]
df_sizes = pd.DataFrame({'tamaño': transaction_sizes})

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        df_sizes,
        x='tamaño',
        nbins=20,
        title="Distribución del número de items por transacción",
        labels={'tamaño': 'Items por transacción', 'count': 'Frecuencia'},
        color_discrete_sequence=['#1f77b4']
    )
    st.plotly_chart(fig_hist, width='stretch')

with col2:
    st.markdown("#### Estadísticas descriptivas")
    stats = df_sizes['tamaño'].describe()
    st.dataframe(stats.to_frame().style.format("{:.2f}"))
    st.caption("La mediana indica el tamaño típico de transacción en el dataset.")

st.markdown("---")

# --- COBERTURA DE PRODUCTOS ---
st.subheader("Curva de Pareto: cobertura acumulada de ventas")
st.markdown("""
El siguiente gráfico muestra el porcentaje acumulado de ventas a medida que se agregan
productos ordenados de mayor a menor frecuencia. Permite aplicar el **principio de Pareto (80/20)**:

- Una curva que asciende rápidamente indica que **pocos productos concentran la mayoría de las ventas**.
- La línea de referencia al 80% permite identificar cuántos productos representan el 80% del volumen total.

Este análisis orienta la selección del parámetro `min_support` en los algoritmos.
""")

product_freq = df.sum().sort_values(ascending=False)
cumulative = product_freq.cumsum() / product_freq.sum() * 100

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=list(range(1, len(cumulative) + 1)),
    y=cumulative.values,
    mode='lines',
    fill='tozeroy',
    line=dict(color='#2ecc71', width=2)
))
fig_cum.update_layout(
    title="Curva de Pareto: porcentaje acumulado de ventas",
    xaxis_title="Número de productos (ordenados por frecuencia descendente)",
    yaxis_title="Porcentaje acumulado de ventas (%)",
    height=400
)
fig_cum.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
st.plotly_chart(fig_cum, width='stretch')

n_products_80 = (cumulative <= 80).sum() + 1
st.info(
    f"Solo **{n_products_80} productos** (de {len(cumulative)} en total) concentran "
    f"el **80% de las ventas**, confirmando el principio de Pareto en este dataset."
)

st.markdown("---")

# --- HEATMAP DE CORRELACIÓN ---
st.subheader("Mapa de calor: correlaciones entre productos frecuentes")
st.markdown("""
El mapa de calor muestra las correlaciones de Pearson entre los productos más vendidos.
Funciona como una vista previa de las asociaciones que los algoritmos Apriori y FP-Growth
explorarán de forma más rigurosa:

- **Tonos rojos**: correlación positiva — los productos tienden a comprarse en las mismas transacciones.
- **Tonos azules**: correlación negativa — los productos rara vez coinciden en una misma transacción.
- **Tonos blancos**: ausencia de relación lineal entre los productos.
""")

n_corr = st.slider("Número de productos para el mapa de calor:", 5, 30, 15)
top_corr_products = get_top_products(df, n_corr).index.tolist()
df_top = df[top_corr_products]
corr_matrix = df_top.corr()

fig_heat = px.imshow(
    corr_matrix,
    text_auto='.2f',
    aspect='auto',
    color_continuous_scale='RdBu_r',
    title=f"Correlaciones entre los {n_corr} productos más frecuentes"
)
fig_heat.update_layout(height=700)
st.plotly_chart(fig_heat, width='stretch')

st.markdown("---")

# --- CONCLUSIONES ---
st.subheader("Conclusiones del análisis exploratorio")
st.markdown("""
A partir del análisis exploratorio se extraen las siguientes observaciones:

1. El dataset es **disperso**: la mayoría de transacciones contienen pocos productos,
   lo que implica usar valores bajos de soporte mínimo en los algoritmos.
2. Existe una **concentración de ventas** en un subconjunto reducido de productos,
   coherente con el principio de Pareto.
3. Los productos más frecuentes aparecerán predominantemente como **consecuentes**
   en las reglas de asociación generadas.
4. Las correlaciones del mapa de calor ofrecen indicios preliminares de las
   asociaciones que los algoritmos identificarán de forma cuantitativa.

El siguiente paso consiste en aplicar los algoritmos **Apriori** y **FP-Growth**
para extraer reglas formales de asociación.
""")

st.caption("Datos procesados desde `Groceries_dataset.csv` | Metodología CRISP-DM")

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

st.set_page_config(page_title="Exploración", page_icon="📊", layout="wide")

# --- ENCABEZADO ---
st.title("📊 Análisis Exploratorio de Datos")
st.markdown("""
### 🎯 ¿Qué encontrarás en esta página?

Esta sección corresponde a la **Fase 2 de CRISP-DM: Comprensión de los Datos**.  
Aquí exploramos las características del dataset *Groceries* para responder preguntas como:

- 🛍️ ¿Cuáles son los productos más vendidos?
- 📦 ¿Cuántos productos suele llevar un cliente por transacción?
- 📈 ¿Cuántos productos concentran la mayor parte de las ventas?
- 🔥 ¿Qué productos tienden a comprarse en las mismas transacciones?

Estos hallazgos nos guiarán en la elección de los parámetros del modelo de reglas de asociación.
""")
st.markdown("---")

# Cargar datos
df = load_processed_data()
transactions = load_transactions_list()

# --- KPIs RÁPIDOS ---
st.subheader("📌 Datos generales del dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🧾 Transacciones", f"{df.shape[0]:,}")
col2.metric("🛍️ Productos únicos", f"{df.shape[1]:,}")
col3.metric("📦 Ítems totales", f"{int(df.sum().sum()):,}")
col4.metric("📊 Ítems/transacción", f"{df.sum(axis=1).mean():.2f}")

st.info("""
💡 **Interpretación**: el dataset tiene miles de transacciones pero un número limitado de 
productos. El promedio de ítems por transacción nos indica qué tan grande es una compra típica.
""")

st.markdown("---")

# --- TOP PRODUCTOS ---
st.subheader("🏆 Top productos más vendidos")
st.markdown("""
**¿Qué muestra este gráfico?**  
Los productos con mayor frecuencia de compra en todo el dataset.

**¿Cómo lo interpreto?**  
Los productos en la parte superior son los que **más clientes compran**. Generalmente aparecen 
como **consecuentes** en las reglas de asociación porque son comunes en muchas canastas.
""")

top_n = st.slider("Selecciona cuántos productos mostrar:", 5, 50, 20)
top_products = get_top_products(df, top_n)

fig = px.bar(
    x=top_products.values,
    y=top_products.index,
    orientation='h',
    labels={'x': 'Frecuencia', 'y': 'Producto'},
    title=f"Top {top_n} productos más comprados",
    color=top_products.values,
    color_continuous_scale='Viridis'
)
fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
st.plotly_chart(fig, width='stretch')

st.success(f"✅ **Observación**: El producto más vendido es **{top_products.index[0]}**, "
           f"presente en {top_products.iloc[0]:,} transacciones.")

st.markdown("---")

# --- DISTRIBUCIÓN DEL TAMAÑO DE TRANSACCIONES ---
st.subheader("📦 Tamaño de las transacciones")
st.markdown("""
**¿Qué muestra este gráfico?**  
La distribución del número de productos por transacción (canasta).

**¿Cómo lo interpreto?**  
- Si la mayoría de transacciones son pequeñas (1-3 productos), el dataset es **disperso**.
- Esto significa que debemos usar **soportes bajos** (ej: 0.005) al aplicar Apriori/FP-Growth.
- Transacciones grandes generan más reglas, pero también más ruido.
""")

transaction_sizes = [len(t) for t in transactions]
df_sizes = pd.DataFrame({'tamaño': transaction_sizes})

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        df_sizes,
        x='tamaño',
        nbins=20,
        title="Distribución del número de ítems por transacción",
        labels={'tamaño': 'Ítems por transacción', 'count': 'Frecuencia'},
        color_discrete_sequence=['#1f77b4']
    )
    st.plotly_chart(fig_hist, width='stretch')

with col2:
    st.markdown("### 📊 Estadísticas descriptivas")
    stats = df_sizes['tamaño'].describe()
    st.dataframe(stats.to_frame().style.format("{:.2f}"))
    st.caption("📌 La mediana indica el tamaño 'típico' de transacción.")

st.markdown("---")

# --- COBERTURA DE PRODUCTOS ---
st.subheader("📈 Curva de Pareto: cobertura de productos")
st.markdown("""
**¿Qué muestra este gráfico?**  
El porcentaje acumulado de ventas conforme agregamos productos ordenados de mayor a menor.

**¿Cómo lo interpreto?**  
Este gráfico aplica el **principio de Pareto (80/20)**:
- ¿Cuántos productos concentran el 80% de las ventas?
- Si la curva sube rápido, **pocos productos dominan las ventas** (mercado concentrado).
- Si la curva sube lento, las ventas están **distribuidas** entre muchos productos.
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
    title="Curva de Pareto: % acumulado de ventas",
    xaxis_title="Número de productos (ordenados)",
    yaxis_title="% acumulado de ventas",
    height=400
)
fig_cum.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
st.plotly_chart(fig_cum, width='stretch')

n_products_80 = (cumulative <= 80).sum() + 1
st.info(f"💡 **Hallazgo clave**: Solo **{n_products_80} productos** "
        f"(de {len(cumulative)}) representan el **80% de las ventas**. "
        f"Esto confirma el principio de Pareto en este dataset.")

st.markdown("---")

# --- HEATMAP ---
st.subheader("🔥 Mapa de calor: correlaciones entre top productos")
st.markdown("""
**¿Qué muestra este gráfico?**  
Las correlaciones de Pearson entre los productos más vendidos.

**¿Cómo lo interpreto?**  
- **Rojo intenso**: productos que tienden a comprarse juntos (correlación positiva).
- **Azul intenso**: productos que rara vez se compran juntos (correlación negativa).
- **Blanco**: sin relación.

Es una **vista previa** de las asociaciones que descubrirán Apriori y FP-Growth.
""")

n_corr = st.slider("Productos para el heatmap:", 5, 30, 15)
top_corr_products = get_top_products(df, n_corr).index.tolist()
df_top = df[top_corr_products]
corr_matrix = df_top.corr()

fig_heat = px.imshow(
    corr_matrix,
    text_auto='.2f',
    aspect='auto',
    color_continuous_scale='RdBu_r',
    title=f"Correlación entre los {n_corr} productos más vendidos"
)
fig_heat.update_layout(height=700)
st.plotly_chart(fig_heat, width='stretch')

st.markdown("---")

# --- CONCLUSIONES ---
st.subheader("📝 Conclusiones del análisis exploratorio")
st.markdown("""
A partir de esta exploración podemos concluir que:

1. ✅ El dataset es **disperso**: la mayoría de transacciones son pequeñas.
2. ✅ Existe una **concentración de ventas** en pocos productos (Pareto).
3. ✅ Los productos más vendidos serán candidatos naturales a aparecer en muchas reglas.
4. ✅ Las correlaciones del heatmap nos dan pistas de **qué reglas podríamos encontrar**.

➡️ **Siguiente paso**: aplicar los algoritmos **Apriori** y **FP-Growth** en las páginas correspondientes.
""")

st.caption("💾 Datos procesados desde `Groceries_dataset.csv` siguiendo metodología CRISP-DM.")
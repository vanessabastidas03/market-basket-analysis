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

st.title("📊 Análisis Exploratorio de Datos")
st.markdown("Exploración inicial del dataset Groceries para entender su estructura.")
st.markdown("---")

# Cargar datos
df = load_processed_data()
transactions = load_transactions_list()

# --- TOP PRODUCTOS ---
st.subheader("🏆 Top productos más vendidos")

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
st.plotly_chart(fig, use_container_width=True)

# --- DISTRIBUCIÓN DEL TAMAÑO DE TRANSACCIONES ---
st.subheader("📦 Tamaño de las transacciones")

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
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("### 📊 Estadísticas")
    stats = df_sizes['tamaño'].describe()
    st.dataframe(stats.to_frame().style.format("{:.2f}"))

# --- COBERTURA DE PRODUCTOS ---
st.subheader("📈 Cobertura de productos")

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
st.plotly_chart(fig_cum, use_container_width=True)

n_products_80 = (cumulative <= 80).sum() + 1
st.info(f"💡 **Insight**: Solo **{n_products_80} productos** (de {len(cumulative)}) "
        f"representan el **80% de las ventas**. Esto sigue el principio de Pareto.")

# --- HEATMAP DE CORRELACIÓN (top productos) ---
st.subheader("🔥 Mapa de calor: correlaciones entre top productos")

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
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")
st.caption("💾 Datos procesados desde `Groceries_dataset.csv` siguiendo metodología CRISP-DM.")
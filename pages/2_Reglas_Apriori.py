"""
Página de visualización de reglas con Apriori.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_processed_data, format_itemset
from src.apriori_model import run_apriori, generate_rules

st.set_page_config(page_title="Reglas Apriori", page_icon=None, layout="wide")

st.title("Reglas de Asociación — Algoritmo Apriori")
st.markdown("""
El algoritmo **Apriori** es uno de los métodos clásicos para la minería de reglas de asociación.
Opera en dos etapas: primero identifica los itemsets frecuentes mediante un proceso iterativo
(generando y podando candidatos), y luego extrae reglas de asociación a partir de dichos itemsets.

#### Métricas de evaluación

| Métrica | Interpretación |
|---------|---------------|
| **Support** | Fracción de transacciones que contienen el itemset completo (antecedente + consecuente). |
| **Confidence** | Probabilidad de que el consecuente se compre dado que el antecedente fue comprado. |
| **Lift** | Fuerza de la asociación. Lift > 1 indica que los ítems se compran juntos más de lo esperado por azar. |

#### Uso de esta página

1. Ajuste los parámetros en la barra lateral izquierda.
2. Haga clic en **Ejecutar Apriori** para generar las reglas.
3. Explore los resultados en la tabla, el diagrama de dispersión y el grafo de relaciones.

**Nota:** valores bajos de `min_support` producen más reglas pero incrementan el tiempo de cómputo.
""")
st.markdown("---")

df = load_processed_data()

# --- PARÁMETROS ---
st.sidebar.header("Parámetros del algoritmo")
min_support = st.sidebar.slider("Soporte mínimo", 0.001, 0.1, 0.005, 0.001, format="%.3f")
min_confidence = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.1, 0.05)
min_lift = st.sidebar.slider("Lift mínimo", 0.5, 5.0, 1.0, 0.1)
max_len = st.sidebar.selectbox("Tamaño máximo de itemset", [None, 2, 3, 4], index=2)

st.sidebar.markdown("---")
st.sidebar.info(
    "Soportes más bajos generan más itemsets, pero el algoritmo tarda más. "
    "Un lift > 1 indica asociación positiva entre los productos."
)

# --- EJECUCIÓN ---
if st.button("Ejecutar Apriori", type="primary"):
    with st.spinner("Ejecutando Apriori..."):
        itemsets, elapsed = run_apriori(df, min_support=min_support, max_len=max_len)

    if itemsets.empty:
        st.warning("No se encontraron itemsets con los parámetros seleccionados. Reduzca el soporte mínimo.")
        st.stop()

    rules = generate_rules(itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift].reset_index(drop=True)

    st.session_state['apriori_rules'] = rules
    st.session_state['apriori_itemsets'] = itemsets
    st.session_state['apriori_time'] = elapsed

# --- MOSTRAR RESULTADOS ---
if 'apriori_rules' in st.session_state:
    rules = st.session_state['apriori_rules']
    itemsets = st.session_state['apriori_itemsets']
    elapsed = st.session_state['apriori_time']

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tiempo de ejecución", f"{elapsed:.3f} s")
    col2.metric("Itemsets frecuentes", len(itemsets))
    col3.metric("Reglas generadas", len(rules))
    col4.metric("Lift máximo", f"{rules['lift'].max():.2f}" if not rules.empty else "N/A")

    if rules.empty:
        st.warning("No se generaron reglas con los umbrales definidos. Ajuste los parámetros.")
        st.stop()

    st.markdown("---")

    # --- TABLA DE REGLAS ---
    st.subheader("Reglas de asociación encontradas")
    st.markdown("""
    La tabla muestra todas las reglas que superan los umbrales definidos.
    Cada fila representa una regla del tipo *"Si el cliente compra [Antecedente],
    también comprará [Consecuente]"*.
    Las reglas están ordenadas por **lift** de forma descendente.
    """)

    rules_display = rules.copy()
    rules_display['antecedents'] = rules_display['antecedents'].apply(format_itemset)
    rules_display['consequents'] = rules_display['consequents'].apply(format_itemset)
    rules_display = rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules_display.columns = ['Antecedente (Si compra...)', 'Consecuente (También compra...)', 'Support', 'Confidence', 'Lift']

    st.dataframe(
        rules_display.style.format({
            'Support': '{:.4f}',
            'Confidence': '{:.4f}',
            'Lift': '{:.2f}'
        }).background_gradient(subset=['Lift'], cmap='Greens'),
        width="stretch",
        height=400
    )

    csv = rules_display.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar reglas (CSV)", csv, "apriori_rules.csv", "text/csv")

    st.markdown("---")

    # --- VISUALIZACIÓN: SCATTER ---
    st.subheader("Diagrama de dispersión: Support vs Confidence")
    st.markdown("""
    Cada punto representa una regla de asociación. El color y el tamaño del punto
    corresponden al valor de **lift**. Las reglas más valiosas se ubican en la
    esquina superior derecha: alta confianza y buen soporte.
    """)

    rules_plot = rules.copy()
    rules_plot['antecedents_str'] = rules_plot['antecedents'].apply(lambda x: ", ".join(list(x)))
    rules_plot['consequents_str'] = rules_plot['consequents'].apply(lambda x: ", ".join(list(x)))
    rules_plot = rules_plot[['support', 'confidence', 'lift', 'antecedents_str', 'consequents_str']]

    fig_scatter = px.scatter(
        rules_plot,
        x='support',
        y='confidence',
        color='lift',
        size='lift',
        hover_data={'antecedents_str': True, 'consequents_str': True},
        color_continuous_scale='Viridis',
        title="Distribución de reglas de asociación — Apriori"
    )
    fig_scatter.update_traces(
        hovertemplate='<b>Support:</b> %{x:.4f}<br><b>Confidence:</b> %{y:.4f}<br><b>Lift:</b> %{marker.color:.2f}<extra></extra>'
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # --- TOP 10 REGLAS ---
    st.subheader("Top 10 reglas por Lift")
    st.markdown("Las diez reglas con mayor lift representan las asociaciones más fuertes encontradas en el dataset.")

    top10 = rules_plot.head(10).copy()
    top10['rule'] = top10.apply(
        lambda r: f"{r['antecedents_str']} → {r['consequents_str']}",
        axis=1
    )

    fig_top = px.bar(
        top10,
        x='lift',
        y='rule',
        orientation='h',
        color='confidence',
        color_continuous_scale='Plasma',
        title="Top 10 reglas por Lift — Apriori"
    )
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_top, width="stretch")

    # --- GRAFO DE RELACIONES ---
    st.subheader("Grafo de relaciones (Top 20 reglas)")
    st.markdown("""
    El grafo dirigido representa las relaciones entre productos derivadas de las 20 reglas
    con mayor lift. Cada nodo es un producto o conjunto de productos; cada arista indica
    que existe una regla de asociación entre el antecedente y el consecuente.
    """)

    G = nx.DiGraph()
    top20 = rules.head(20)

    for _, row in top20.iterrows():
        ante = format_itemset(row['antecedents'])
        cons = format_itemset(row['consequents'])
        G.add_edge(ante, cons, weight=row['lift'])

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title("Grafo de reglas de asociación — Top 20 por Lift (Apriori)", fontsize=14)
    ax.axis('off')
    st.pyplot(fig)

else:
    st.info("Configure los parámetros en la barra lateral y haga clic en **Ejecutar Apriori** para generar las reglas.")

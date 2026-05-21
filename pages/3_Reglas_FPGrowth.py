"""
Página de visualización de reglas con FP-Growth.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_processed_data, format_itemset
from src.fpgrowth_model import run_fpgrowth, generate_rules_fpgrowth

st.set_page_config(page_title="Reglas FP-Growth", page_icon=None, layout="wide")

st.title("Reglas de Asociación — Algoritmo FP-Growth")
st.markdown("""
**FP-Growth** (Frequent Pattern Growth) es una alternativa más eficiente al algoritmo Apriori.
En lugar de generar candidatos de forma iterativa, construye una estructura de árbol compacta
denominada *FP-tree* que permite extraer todos los itemsets frecuentes en exactamente dos
recorridos sobre el dataset.

#### Ventajas frente a Apriori

- No genera candidatos explícitamente, lo que reduce el costo computacional.
- Requiere solo dos lecturas del dataset.
- Es significativamente más rápido en datasets de gran volumen o con soporte mínimo bajo.

#### Métricas de evaluación

| Métrica | Interpretación |
|---------|---------------|
| **Support** | Fracción de transacciones que contienen el itemset completo. |
| **Confidence** | Probabilidad de que el consecuente se compre dado que el antecedente fue comprado. |
| **Lift** | Fuerza de la asociación. Lift > 1 indica que los ítems se compran juntos más de lo esperado por azar. |

#### Uso de esta página

1. Ajuste los parámetros en la barra lateral izquierda.
2. Haga clic en **Ejecutar FP-Growth** para generar las reglas.
3. Explore los resultados en la tabla, el diagrama de dispersión y el ranking de reglas.
""")
st.markdown("---")

df = load_processed_data()

# --- PARÁMETROS ---
st.sidebar.header("Parámetros del algoritmo")
min_support = st.sidebar.slider("Soporte mínimo", 0.001, 0.1, 0.005, 0.001, format="%.3f")
min_confidence = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.1, 0.05)
min_lift = st.sidebar.slider("Lift mínimo", 0.5, 5.0, 1.0, 0.1)
max_len = st.sidebar.selectbox("Tamaño máximo de itemset", [None, 2, 3, 4], index=2)

if st.button("Ejecutar FP-Growth", type="primary"):
    with st.spinner("Ejecutando FP-Growth..."):
        itemsets, elapsed = run_fpgrowth(df, min_support=min_support, max_len=max_len)

    if itemsets.empty:
        st.warning("No se encontraron itemsets con los parámetros seleccionados. Reduzca el soporte mínimo.")
        st.stop()

    rules = generate_rules_fpgrowth(itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift].reset_index(drop=True)

    # Convertir frozenset a cadena de texto para evitar error de serialización JSON
    rules['antecedents_str'] = rules['antecedents'].apply(format_itemset)
    rules['consequents_str'] = rules['consequents'].apply(format_itemset)

    st.session_state['fp_rules'] = rules
    st.session_state['fp_itemsets'] = itemsets
    st.session_state['fp_time'] = elapsed

if 'fp_rules' in st.session_state:
    rules = st.session_state['fp_rules']
    itemsets = st.session_state['fp_itemsets']
    elapsed = st.session_state['fp_time']

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
    Cada fila representa una regla del tipo *"Si el cliente compra [Antecedente],
    también comprará [Consecuente]"*. Las reglas están ordenadas por **lift** de forma descendente.

    | Métrica | Significado práctico |
    |---------|---------------------|
    | Support | Frecuencia de la combinación en el dataset. |
    | Confidence | Probabilidad de B cuando A está presente en la canasta. |
    | Lift | Lift > 1 indica asociación positiva; cuanto mayor, más fuerte la relación. |
    """)

    rules_display = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
    rules_display.columns = ['Antecedente', 'Consecuente', 'Support', 'Confidence', 'Lift']

    st.dataframe(
        rules_display.style.format({
            'Support': '{:.4f}',
            'Confidence': '{:.4f}',
            'Lift': '{:.2f}'
        }).background_gradient(subset=['Lift'], cmap='Oranges'),
        width='stretch',
        height=400
    )

    csv = rules_display.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar reglas (CSV)", csv, "fpgrowth_rules.csv", "text/csv")

    st.markdown("---")

    # --- SCATTER ---
    st.subheader("Diagrama de dispersión: Support vs Confidence")
    st.markdown("""
    Cada punto representa una regla de asociación. El color y tamaño del punto reflejan el
    valor de **lift**. Las reglas con mayor relevancia se ubican en la esquina superior derecha
    del diagrama: alta confianza y soporte considerable.
    """)

    fig = px.scatter(
        rules,
        x='support',
        y='confidence',
        color='lift',
        size='lift',
        hover_data=['antecedents_str', 'consequents_str'],
        color_continuous_scale='Plasma',
        title="Distribución de reglas de asociación — FP-Growth"
    )
    st.plotly_chart(fig, width='stretch')

    # --- TOP 10 ---
    st.subheader("Top 10 reglas por Lift")
    st.markdown("Las diez reglas con mayor lift representan las asociaciones más fuertes identificadas por FP-Growth.")

    top10 = rules.head(10).copy()
    top10['rule'] = top10['antecedents_str'] + ' → ' + top10['consequents_str']

    fig_top = px.bar(
        top10,
        x='lift',
        y='rule',
        orientation='h',
        color='confidence',
        color_continuous_scale='Plasma',
        title="Top 10 reglas por Lift — FP-Growth"
    )
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_top, width='stretch')

else:
    st.info("Configure los parámetros en la barra lateral y haga clic en **Ejecutar FP-Growth** para generar las reglas.")

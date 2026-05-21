"""
Generador de combinaciones de productos.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_processed_data, format_itemset
from src.fpgrowth_model import run_fpgrowth, generate_rules_fpgrowth

st.set_page_config(page_title="Generador de Recomendaciones", page_icon=None, layout="wide")

st.title("Generador de Recomendaciones de Productos")
st.markdown("""
Esta herramienta implementa un **sistema de recomendación** basado en reglas de asociación,
equivalente al mecanismo utilizado por plataformas de comercio electrónico bajo el principio:

> *"Los clientes que adquirieron estos productos también compraron..."*

#### Funcionamiento del sistema

1. El usuario selecciona uno o varios productos que representan su **canasta de compra**.
2. El sistema identifica todas las reglas de asociación cuyo **antecedente** sea un subconjunto
   de la selección del usuario.
3. Se filtran las reglas cuyos **consecuentes** no estén ya en la canasta del usuario.
4. Los resultados se ordenan por **lift** de forma descendente y se presentan como recomendaciones.

#### Aplicaciones en el ámbito empresarial

- Diseño del *layout* de tiendas: ubicar productos asociados en secciones contiguas.
- Promociones cruzadas: "Compre A y obtenga B con descuento".
- Recomendaciones en plataformas de comercio electrónico.
- Campañas de marketing segmentadas según el historial de compra.
""")
st.markdown("---")

df = load_processed_data()

# --- PARÁMETROS ---
with st.sidebar:
    st.header("Parámetros del modelo")
    st.caption("Defina los umbrales mínimos que deben cumplir las reglas utilizadas para recomendar.")
    min_support = st.slider("Soporte mínimo", 0.001, 0.05, 0.003, 0.001, format="%.3f")
    min_confidence = st.slider("Confianza mínima", 0.0, 1.0, 0.05, 0.01)
    min_lift = st.slider("Lift mínimo", 0.5, 5.0, 1.0, 0.1)
    top_n = st.slider("Número de recomendaciones", 1, 20, 5)

# --- GENERAR REGLAS (con cache) ---
@st.cache_data(show_spinner=False)
def get_rules(min_sup, min_conf, min_l):
    itemsets, _ = run_fpgrowth(df, min_support=min_sup)
    if itemsets.empty:
        return pd.DataFrame()
    rules = generate_rules_fpgrowth(itemsets, metric='confidence', min_threshold=min_conf)
    rules = rules[rules['lift'] >= min_l].reset_index(drop=True)
    return rules

with st.spinner("Cargando reglas de asociación..."):
    rules = get_rules(min_support, min_confidence, min_lift)

if rules.empty:
    st.warning("No se encontraron reglas con los umbrales definidos. Reduzca el soporte mínimo o la confianza mínima.")
    st.stop()

st.success(f"{len(rules)} reglas disponibles para generar recomendaciones.")

# --- SELECCIÓN DE PRODUCTOS ---
st.subheader("Construcción de la canasta de compra")
st.markdown("""
Seleccione uno o más productos que representen los artículos que el cliente ya tiene en su canasta.
El sistema buscará productos frecuentemente asociados a su selección.
""")

all_products = sorted(df.columns.tolist())

selected_products = st.multiselect(
    "Seleccione productos:",
    options=all_products,
    default=['whole milk'] if 'whole milk' in all_products else [],
    help="Estos productos representan la canasta de compra actual del cliente."
)

if not selected_products:
    st.info("Seleccione al menos un producto para obtener recomendaciones.")
    st.stop()

# --- BUSCAR RECOMENDACIONES ---
st.markdown("---")
st.subheader("Recomendaciones generadas")
st.markdown("Resultados ordenados por relevancia (lift descendente).")

selected_set = frozenset(selected_products)

matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(selected_set))].copy()
matching_rules = matching_rules[
    matching_rules['consequents'].apply(lambda x: x.isdisjoint(selected_set))
]

if matching_rules.empty:
    st.warning("No se encontraron recomendaciones para la combinación seleccionada.")
    st.markdown("""
    **Sugerencias para obtener resultados:**
    - Reduzca el **soporte mínimo** en la barra lateral.
    - Reduzca la **confianza mínima**.
    - Seleccione productos de alta frecuencia como *whole milk*, *other vegetables* o *rolls/buns*.
    """)
    st.stop()

matching_rules['recomendados'] = matching_rules['consequents'].apply(format_itemset)
matching_rules['antecedente_aplicado'] = matching_rules['antecedents'].apply(format_itemset)

best_recommendations = (
    matching_rules.sort_values('lift', ascending=False)
                  .drop_duplicates(subset=['recomendados'])
                  .head(top_n)
)

# --- MOSTRAR RECOMENDACIONES ---
for idx, row in best_recommendations.iterrows():
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        with col1:
            st.markdown(f"**{row['recomendados']}**")
            st.caption(f"Basado en: {row['antecedente_aplicado']}")
        with col2:
            st.metric("Confianza", f"{row['confidence']*100:.1f}%")
        with col3:
            st.metric("Lift", f"{row['lift']:.2f}")
        with col4:
            st.metric("Support", f"{row['support']*100:.2f}%")
        st.markdown("---")

# --- INTERPRETACIÓN ---
with st.expander("Interpretación de las métricas de recomendación"):
    st.markdown("""
    **Confianza**: probabilidad de que el cliente adquiera el producto recomendado,
    dado que ya tiene en su canasta los productos del antecedente.

    **Lift**: indica cuánto más probable es la compra conjunta respecto a lo esperado
    si los productos fueran independientes. Un lift de 2.0 significa que los productos
    se compran juntos el doble de lo esperado por azar.

    **Support**: proporción de transacciones del dataset en las que aparece esta
    combinación de productos. Un soporte alto indica que la asociación es frecuente y robusta.

    **Criterio de ordenación**: las recomendaciones se presentan ordenadas por lift de mayor
    a menor, priorizando las asociaciones más fuertes. En caso de consecuentes duplicados,
    se conserva únicamente la regla con mayor lift.
    """)

# --- TABLA COMPLETA ---
st.markdown("---")
st.subheader("Detalle de todas las reglas aplicables")
st.markdown("Tabla completa de reglas que coinciden con la selección actual, limitada a las primeras 50 entradas.")

display_df = matching_rules[['antecedente_aplicado', 'recomendados', 'support', 'confidence', 'lift']].head(50)
display_df.columns = ['Antecedente (canasta)', 'Producto sugerido', 'Support', 'Confidence', 'Lift']
st.dataframe(
    display_df.style.format({'Support': '{:.4f}', 'Confidence': '{:.4f}', 'Lift': '{:.2f}'}),
    width='stretch'
)

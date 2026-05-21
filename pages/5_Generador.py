"""
Generador de combinaciones de productos.
Recomienda productos basándose en reglas de asociación.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_processed_data, format_itemset
from src.fpgrowth_model import run_fpgrowth, generate_rules_fpgrowth

st.set_page_config(page_title="Generador", page_icon="🛒", layout="wide")

st.title("🛒 Generador de Combinaciones de Productos")
st.markdown("""
Selecciona productos en tu canasta y recibe **recomendaciones** basadas en reglas de asociación.
El sistema utiliza FP-Growth para encontrar los productos más probablemente asociados a tu selección.
""")
st.markdown("---")

df = load_processed_data()

# --- PARÁMETROS GENERALES ---
with st.sidebar:
    st.header("⚙️ Parámetros del modelo")
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

with st.spinner("⏳ Cargando reglas..."):
    rules = get_rules(min_support, min_confidence, min_lift)

if rules.empty:
    st.warning("⚠️ No hay reglas con esos umbrales. Reduce el soporte mínimo o la confianza.")
    st.stop()

st.success(f"✅ {len(rules)} reglas disponibles para recomendar.")

# --- SELECCIÓN DE PRODUCTOS ---
st.subheader("🛍️ Construye tu canasta")

all_products = sorted(df.columns.tolist())

selected_products = st.multiselect(
    "Selecciona uno o más productos:",
    options=all_products,
    default=['whole milk'] if 'whole milk' in all_products else [],
    help="Estos productos representan los que tienes en tu canasta de compra."
)

if not selected_products:
    st.info("👆 Selecciona al menos un producto para recibir recomendaciones.")
    st.stop()

# --- BUSCAR RECOMENDACIONES ---
st.markdown("---")
st.subheader("🎯 Recomendaciones para tu canasta")

selected_set = frozenset(selected_products)

# Filtrar reglas cuyo antecedente esté contenido en la selección
matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(selected_set))].copy()

# Filtrar también: el consecuente NO debe estar ya en la canasta
matching_rules = matching_rules[
    matching_rules['consequents'].apply(lambda x: x.isdisjoint(selected_set))
]

if matching_rules.empty:
    st.warning("⚠️ No se encontraron recomendaciones para esta combinación. Intenta:")
    st.markdown("- Reducir el soporte/confianza mínimos en la barra lateral.")
    st.markdown("- Seleccionar productos más populares.")
    st.stop()

# Agregar columna 'producto recomendado' (puede haber más de uno)
matching_rules['recomendados'] = matching_rules['consequents'].apply(format_itemset)
matching_rules['antecedente_aplicado'] = matching_rules['antecedents'].apply(format_itemset)

# Para evitar duplicados, agrupar por consecuente y tomar el de mayor lift
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
            st.markdown(f"### 🛒 {row['recomendados']}")
            st.caption(f"Porque compraste: *{row['antecedente_aplicado']}*")
        with col2:
            st.metric("Confianza", f"{row['confidence']*100:.1f}%")
        with col3:
            st.metric("Lift", f"{row['lift']:.2f}")
        with col4:
            st.metric("Support", f"{row['support']*100:.2f}%")
        st.markdown("---")

# --- INTERPRETACIÓN ---
with st.expander("ℹ️ ¿Cómo interpretar estas recomendaciones?"):
    st.markdown("""
    - **Confianza**: probabilidad de que el cliente compre el producto recomendado dado que ya tiene los seleccionados.
    - **Lift > 1**: el producto se compra MÁS de lo esperado junto con tu selección (asociación positiva).
    - **Support**: qué tan frecuente es esta combinación en todas las transacciones.
    
    **Ejemplo de uso comercial**:
    - 📌 Ubicar productos recomendados cerca en el supermercado.
    - 📌 Crear promociones tipo "compra X y lleva Y al 50%".
    - 📌 Recomendar productos en e-commerce ("clientes también compraron").
    """)

# --- TABLA COMPLETA ---
st.markdown("---")
st.subheader("📋 Todas las reglas aplicables")
display_df = matching_rules[['antecedente_aplicado', 'recomendados', 'support', 'confidence', 'lift']].head(50)
display_df.columns = ['Tu canasta contiene', 'Producto sugerido', 'Support', 'Confidence', 'Lift']
st.dataframe(
    display_df.style.format({'Support': '{:.4f}', 'Confidence': '{:.4f}', 'Lift': '{:.2f}'}),
    use_container_width=True
)
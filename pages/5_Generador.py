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

st.set_page_config(page_title="Generador", page_icon="🛒", layout="wide")

st.title("🛒 Generador de Combinaciones de Productos")
st.markdown("""
### 🎯 ¿Qué hace este generador?

Es un **sistema de recomendación** basado en reglas de asociación.  
Simula lo que hacen plataformas como Amazon o Mercado Libre cuando muestran:

> *"Los clientes que compraron esto, también compraron..."*

### 🔧 ¿Cómo funciona?

1. **Tú seleccionas** uno o varios productos (tu "canasta virtual").
2. El sistema busca **reglas de asociación** cuyo *antecedente* esté en tu canasta.
3. Te recomienda los productos del *consecuente* con mayor **lift** y **confidence**.

### 📊 ¿Qué significan las métricas?

| Métrica | Significado en términos prácticos |
|---------|-----------------------------------|
| **Confianza** | Probabilidad de que compres el producto recomendado si llevas los seleccionados. |
| **Lift** | Cuánto más probable es comprar el recomendado *junto* a tu selección, vs comprarlo por separado. |
| **Support** | Qué tan frecuente es esa combinación en TODAS las transacciones. |

### 💡 ¿Cómo interpretar las recomendaciones?

- **Lift > 1** → asociación positiva (el producto se compra MÁS con tu selección).
- **Lift = 1** → independencia (la asociación es por azar).
- **Lift < 1** → asociación negativa (los productos no van bien juntos).

➡️ **Selecciona productos abajo y descubre qué te recomienda el modelo.**
""")
st.markdown("---")

df = load_processed_data()

# --- PARÁMETROS ---
with st.sidebar:
    st.header("⚙️ Parámetros del modelo")
    st.caption("Ajusta los umbrales de las reglas que se considerarán para recomendar.")
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
st.markdown("""
Selecciona uno o más productos que representen lo que **ya tienes en tu canasta de compras**.  
El sistema buscará productos asociados a tu selección.
""")

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
st.subheader("🎯 Recomendaciones personalizadas")
st.markdown("""
Estas son las recomendaciones del modelo, **ordenadas por relevancia (lift)**.  
Cada recomendación incluye la regla de asociación que la justifica.
""")

selected_set = frozenset(selected_products)

matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(selected_set))].copy()
matching_rules = matching_rules[
    matching_rules['consequents'].apply(lambda x: x.isdisjoint(selected_set))
]

if matching_rules.empty:
    st.warning("⚠️ No se encontraron recomendaciones para esta combinación.")
    st.markdown("""
    **Sugerencias para obtener recomendaciones:**
    - Reduce el **soporte mínimo** en la barra lateral.
    - Reduce la **confianza mínima**.
    - Prueba con productos más populares (whole milk, other vegetables, rolls/buns).
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
with st.expander("ℹ️ ¿Cómo se relacionan las recomendaciones con las reglas de asociación?"):
    st.markdown("""
    **Conexión directa con el modelo de reglas:**
    
    Cada recomendación viene de una regla del tipo:
    > *Si compraste {antecedente}, entonces probablemente comprarás {consecuente}*
    
    El sistema:
    1. Toma tu canasta (productos seleccionados).
    2. Busca reglas cuyo **antecedente** sea un subconjunto de tu canasta.
    3. Filtra para que el **consecuente** NO esté ya en tu canasta.
    4. Ordena por **lift** (mayor asociación primero).
    5. Elimina duplicados y muestra el top N.
    
    **Aplicaciones reales en negocio:**
    - 📌 **Layout de tiendas**: ubicar productos recomendados cerca.
    - 🎁 **Promociones cruzadas**: "compra X y lleva Y al 50%".
    - 🤖 **E-commerce**: recomendaciones tipo "clientes también compraron".
    - 📧 **Marketing por email**: campañas personalizadas según historial.
    """)

# --- TABLA COMPLETA ---
st.markdown("---")
st.subheader("📋 Todas las reglas aplicables a tu canasta")
st.markdown("Lista completa de reglas que coinciden con tu selección. Útil para análisis detallado.")

display_df = matching_rules[['antecedente_aplicado', 'recomendados', 'support', 'confidence', 'lift']].head(50)
display_df.columns = ['Tu canasta contiene', 'Producto sugerido', 'Support', 'Confidence', 'Lift']
st.dataframe(
    display_df.style.format({'Support': '{:.4f}', 'Confidence': '{:.4f}', 'Lift': '{:.2f}'}),
    width='stretch'
)
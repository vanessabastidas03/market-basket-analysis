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

st.set_page_config(page_title="FP-Growth", page_icon="⚡", layout="wide")

st.title("⚡ Reglas de Asociación - FP-Growth")
st.markdown("Genera reglas de asociación usando el algoritmo **FP-Growth** (más eficiente que Apriori).")
st.markdown("---")

df = load_processed_data()

# --- PARÁMETROS ---
st.sidebar.header("⚙️ Parámetros del algoritmo")
min_support = st.sidebar.slider("Soporte mínimo", 0.001, 0.1, 0.005, 0.001, format="%.3f")
min_confidence = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.1, 0.05)
min_lift = st.sidebar.slider("Lift mínimo", 0.5, 5.0, 1.0, 0.1)
max_len = st.sidebar.selectbox("Tamaño máximo de itemset", [None, 2, 3, 4], index=2)

if st.button("⚡ Ejecutar FP-Growth", type="primary"):
    with st.spinner("⏳ Ejecutando FP-Growth..."):
        itemsets, elapsed = run_fpgrowth(df, min_support=min_support, max_len=max_len)
    
    if itemsets.empty:
        st.warning("⚠️ No se encontraron itemsets.")
        st.stop()
    
    rules = generate_rules_fpgrowth(itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift].reset_index(drop=True)
    
    st.session_state['fp_rules'] = rules
    st.session_state['fp_itemsets'] = itemsets
    st.session_state['fp_time'] = elapsed

if 'fp_rules' in st.session_state:
    rules = st.session_state['fp_rules']
    itemsets = st.session_state['fp_itemsets']
    elapsed = st.session_state['fp_time']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Tiempo", f"{elapsed:.3f} s")
    col2.metric("📦 Itemsets", len(itemsets))
    col3.metric("📜 Reglas", len(rules))
    col4.metric("🎯 Lift máximo", f"{rules['lift'].max():.2f}" if not rules.empty else "N/A")
    
    if rules.empty:
        st.warning("⚠️ No se generaron reglas.")
        st.stop()
    
    st.markdown("---")
    
    # Tabla
    st.subheader("📋 Reglas encontradas")
    rules_display = rules.copy()
    rules_display['antecedents'] = rules_display['antecedents'].apply(format_itemset)
    rules_display['consequents'] = rules_display['consequents'].apply(format_itemset)
    rules_display = rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules_display.columns = ['Antecedente', 'Consecuente', 'Support', 'Confidence', 'Lift']
    
    st.dataframe(
        rules_display.style.format({
            'Support': '{:.4f}',
            'Confidence': '{:.4f}',
            'Lift': '{:.2f}'
        }).background_gradient(subset=['Lift'], cmap='Oranges'),
        use_container_width=True,
        height=400
    )
    
    csv = rules_display.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar reglas (CSV)", csv, "fpgrowth_rules.csv", "text/csv")
    
    st.markdown("---")
    
    # Scatter
    st.subheader("📊 Support vs Confidence (color = Lift)")
    fig = px.scatter(
        rules, x='support', y='confidence', color='lift', size='lift',
        hover_data={'antecedents': True, 'consequents': True},
        color_continuous_scale='Plasma',
        title="Distribución de reglas FP-Growth"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 10
    st.subheader("🏆 Top 10 reglas por Lift")
    top10 = rules.head(10).copy()
    top10['rule'] = top10.apply(
        lambda r: f"{format_itemset(r['antecedents'])} → {format_itemset(r['consequents'])}",
        axis=1
    )
    fig_top = px.bar(
        top10, x='lift', y='rule', orientation='h',
        color='confidence', color_continuous_scale='Plasma'
    )
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_top, use_container_width=True)

else:
    st.info("👈 Configura los parámetros y haz clic en **Ejecutar FP-Growth**.")
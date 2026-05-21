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
st.markdown("""
Genera reglas de asociación usando el algoritmo **FP-Growth**, una alternativa más eficiente que Apriori.

**FP-Growth** construye un árbol llamado *FP-tree* que permite encontrar itemsets frecuentes sin 
generar candidatos explícitamente. Esto lo hace **mucho más rápido** en datasets grandes.
""")
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
    
    # 🔧 CONVERTIR FROZENSET A STRING para evitar error de serialización JSON
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
    col1.metric("⏱️ Tiempo", f"{elapsed:.3f} s")
    col2.metric("📦 Itemsets", len(itemsets))
    col3.metric("📜 Reglas", len(rules))
    col4.metric("🎯 Lift máximo", f"{rules['lift'].max():.2f}" if not rules.empty else "N/A")
    
    if rules.empty:
        st.warning("⚠️ No se generaron reglas.")
        st.stop()
    
    st.markdown("---")
    
    # --- TABLA DE REGLAS ---
    st.subheader("📋 Reglas encontradas")
    st.markdown("""
    Cada fila representa una **regla de asociación** del tipo *"Si el cliente compra A, también comprará B"*.
    
    - **Support**: qué tan frecuente es la regla en el dataset.
    - **Confidence**: probabilidad de B cuando A está presente.
    - **Lift**: cuánto más probable es B con A. *Lift > 1 = asociación positiva.*
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
    st.download_button("📥 Descargar reglas (CSV)", csv, "fpgrowth_rules.csv", "text/csv")
    
    st.markdown("---")
    
    # --- SCATTER ---
    st.subheader("📊 Support vs Confidence (color = Lift)")
    st.markdown("""
    Cada punto es una regla. Los puntos en la **esquina superior derecha** son los más valiosos:
    alta confianza, buen soporte y lift elevado.
    """)
    
    # ✅ USAR LAS COLUMNAS STRING, NO LOS FROZENSETS
    fig = px.scatter(
        rules,
        x='support',
        y='confidence',
        color='lift',
        size='lift',
        hover_data=['antecedents_str', 'consequents_str'],
        color_continuous_scale='Plasma',
        title="Distribución de reglas FP-Growth"
    )
    st.plotly_chart(fig, width='stretch')
    
    # --- TOP 10 ---
    st.subheader("🏆 Top 10 reglas por Lift")
    st.markdown("Las 10 reglas con mayor *lift*: las asociaciones más fuertes entre productos.")
    
    top10 = rules.head(10).copy()
    top10['rule'] = top10['antecedents_str'] + ' → ' + top10['consequents_str']
    
    fig_top = px.bar(
        top10,
        x='lift',
        y='rule',
        orientation='h',
        color='confidence',
        color_continuous_scale='Plasma'
    )
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_top, width='stretch')

else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en **Ejecutar FP-Growth**.")
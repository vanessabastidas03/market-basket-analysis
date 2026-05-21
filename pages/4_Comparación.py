"""
Página de comparación entre Apriori y FP-Growth.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_processed_data
from src.comparison import compare_algorithms, get_comparison_summary

st.set_page_config(page_title="Comparación", page_icon="⚖️", layout="wide")

st.title("⚖️ Comparación: Apriori vs FP-Growth")
st.markdown("""
### 🎯 ¿Qué encontrarás en esta página?

Esta página compara **empíricamente** los dos algoritmos clásicos de minería de reglas de asociación:

- 🔍 **Apriori** (1994): el algoritmo fundamental. Genera candidatos iterativamente.
- ⚡ **FP-Growth** (2000): construye un *FP-tree* sin generar candidatos. Más eficiente.

### 🧪 ¿Qué se compara aquí?

| Dimensión | ¿Qué significa? |
|-----------|-----------------|
| ⏱️ **Tiempo de ejecución** | Cuánto tarda cada algoritmo en procesar el dataset. |
| 📦 **Itemsets generados** | Cantidad de conjuntos de productos frecuentes encontrados. |
| 📜 **Reglas generadas** | Cantidad de reglas A → B obtenidas tras filtrar por confianza y lift. |
| ⚡ **Speedup** | Cuántas veces más rápido es FP-Growth respecto a Apriori. |

### 💡 ¿Cómo interpretar los resultados?

- Ambos algoritmos **producen los mismos itemsets** (validación matemática).
- FP-Growth debería ser **más rápido**, sobre todo con `min_support` bajos.
- Si los itemsets coinciden, se valida la **correctitud** de ambas implementaciones.

📌 Configura distintos valores de `min_support` para ver cómo cambia el rendimiento.
""")
st.markdown("---")

df = load_processed_data()

# --- CONFIGURACIÓN ---
st.sidebar.header("⚙️ Configuración")
support_values_input = st.sidebar.text_input(
    "Valores de min_support (separados por coma)",
    "0.001, 0.005, 0.01, 0.02, 0.05",
    help="Probaremos cada algoritmo con estos valores de soporte mínimo."
)
min_confidence = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.1, 0.05)
min_lift = st.sidebar.slider("Lift mínimo", 0.5, 5.0, 1.0, 0.1)

st.sidebar.info("💡 Soportes más bajos = más reglas pero más tiempo de procesamiento.")

if st.button("🚀 Ejecutar comparación", type="primary"):
    try:
        support_values = [float(s.strip()) for s in support_values_input.split(',')]
    except ValueError:
        st.error("❌ Formato inválido. Usa números separados por coma.")
        st.stop()
    
    with st.spinner("⏳ Comparando algoritmos... (puede tardar varios minutos)"):
        comparison = compare_algorithms(df, support_values, min_confidence, min_lift)
        st.session_state['comparison'] = comparison

if 'comparison' in st.session_state:
    comp = st.session_state['comparison']
    summary = get_comparison_summary(comp)
    
    # --- KPIs ---
    st.subheader("🎯 Resumen ejecutivo")
    st.markdown("Una vista rápida del ganador en eficiencia.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Algoritmo más rápido", summary.get('winner_speed', 'N/A'))
    col2.metric("⚡ Speedup promedio (FP/Apriori)", f"{summary.get('avg_speedup', 0):.2f}x")
    col3.metric("✅ Mismos itemsets", "Sí" if summary.get('same_results') else "No")
    
    st.markdown("---")
    
    # --- TABLA ---
    st.subheader("📋 Tabla comparativa detallada")
    st.markdown("""
    Cada fila muestra los resultados para un valor distinto de `min_support`.  
    La columna **speedup** indica cuántas veces más rápido es FP-Growth.
    """)
    
    st.dataframe(comp.style.format({
        'min_support': '{:.4f}',
        'apriori_time_s': '{:.4f}',
        'fpgrowth_time_s': '{:.4f}',
        'speedup': '{:.2f}'
    }).background_gradient(subset=['speedup'], cmap='RdYlGn'), width='stretch')
    
    csv = comp.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar comparación (CSV)", csv, "comparison.csv", "text/csv")
    
    st.markdown("---")
    
    # --- TIEMPOS ---
    st.subheader("⏱️ Tiempos de ejecución")
    st.markdown("""
    **¿Qué muestra este gráfico?**  
    Cómo varía el tiempo de cada algoritmo al cambiar el `min_support`.
    
    **¿Cómo lo interpreto?**  
    - A medida que **`min_support` baja**, ambos algoritmos tardan más.
    - La línea de **FP-Growth** suele quedar por debajo de Apriori = es más rápido.
    """)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comp['min_support'], y=comp['apriori_time_s'],
        mode='lines+markers', name='Apriori',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10)
    ))
    fig.add_trace(go.Scatter(
        x=comp['min_support'], y=comp['fpgrowth_time_s'],
        mode='lines+markers', name='FP-Growth',
        line=dict(color='#e67e22', width=3),
        marker=dict(size=10)
    ))
    fig.update_layout(
        title="Tiempo de ejecución vs Soporte mínimo",
        xaxis_title="min_support",
        yaxis_title="Tiempo (segundos)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, width='stretch')
    
    # --- BARRAS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Itemsets generados")
        st.caption("Ambos algoritmos deberían generar la misma cantidad: confirma corrección.")
        df_melt = comp.melt(
            id_vars='min_support',
            value_vars=['apriori_itemsets', 'fpgrowth_itemsets'],
            var_name='Algoritmo', value_name='Itemsets'
        )
        df_melt['Algoritmo'] = df_melt['Algoritmo'].map({
            'apriori_itemsets': 'Apriori',
            'fpgrowth_itemsets': 'FP-Growth'
        })
        fig_it = px.bar(
            df_melt, x='min_support', y='Itemsets', color='Algoritmo',
            barmode='group',
            color_discrete_map={'Apriori': '#3498db', 'FP-Growth': '#e67e22'}
        )
        st.plotly_chart(fig_it, width='stretch')
    
    with col2:
        st.subheader("📜 Reglas generadas")
        st.caption("Reglas tras filtrar por confianza y lift mínimos.")
        df_melt_r = comp.melt(
            id_vars='min_support',
            value_vars=['apriori_rules', 'fpgrowth_rules'],
            var_name='Algoritmo', value_name='Reglas'
        )
        df_melt_r['Algoritmo'] = df_melt_r['Algoritmo'].map({
            'apriori_rules': 'Apriori',
            'fpgrowth_rules': 'FP-Growth'
        })
        fig_r = px.bar(
            df_melt_r, x='min_support', y='Reglas', color='Algoritmo',
            barmode='group',
            color_discrete_map={'Apriori': '#3498db', 'FP-Growth': '#e67e22'}
        )
        st.plotly_chart(fig_r, width='stretch')
    
    st.markdown("---")
    
    # --- CONCLUSIONES ---
    st.subheader("📝 Conclusiones de la comparación")
    speedup_val = summary.get('avg_speedup', 1)
    st.markdown(f"""
    Los resultados confirman las siguientes observaciones:
    
    - ⚡ **FP-Growth es ~{speedup_val:.1f}x más rápido** que Apriori en promedio.
    - ✅ Ambos algoritmos producen los **mismos itemsets frecuentes** (validación matemática).
    - 📚 **Cuándo usar Apriori**: datasets pequeños, fines didácticos, fácil de explicar paso a paso.
    - 🚀 **Cuándo usar FP-Growth**: datasets grandes, producción, escenarios con `min_support` bajo.
    
    ➡️ **Recomendación**: Para este dataset y para producción, usar **FP-Growth**.
    """)

else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en **Ejecutar comparación**.")
    
    st.markdown("---")
    st.markdown("### 📚 ¿Cómo funciona internamente cada algoritmo?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🔍 Apriori
        1. Encuentra 1-itemsets frecuentes.
        2. Genera candidatos 2-itemsets.
        3. Poda los no frecuentes.
        4. Repite hasta que no haya más itemsets.
        
        **Desventaja**: muchas pasadas sobre los datos.
        """)
    with col2:
        st.markdown("""
        #### ⚡ FP-Growth
        1. Recorre el dataset 2 veces.
        2. Construye un *FP-tree* compacto.
        3. Extrae itemsets recursivamente del árbol.
        
        **Ventaja**: no genera candidatos explícitamente.
        """)
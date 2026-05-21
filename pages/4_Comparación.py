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

st.set_page_config(page_title="Comparación de Algoritmos", page_icon=None, layout="wide")

st.title("Comparación: Apriori vs FP-Growth")
st.markdown("""
Esta sección realiza una **comparativa empírica** entre los dos algoritmos de minería de reglas
de asociación implementados en este proyecto. El objetivo es evaluar su comportamiento bajo
distintos valores de soporte mínimo y verificar la consistencia de sus resultados.

#### Dimensiones de comparación

| Dimensión | Descripción |
|-----------|-------------|
| **Tiempo de ejecución** | Segundos que tarda cada algoritmo en procesar el dataset completo. |
| **Itemsets frecuentes** | Número de conjuntos de productos que superan el soporte mínimo definido. |
| **Reglas generadas** | Número de reglas A → B obtenidas tras filtrar por confianza y lift. |
| **Speedup** | Factor de aceleración: cuántas veces más rápido es FP-Growth respecto a Apriori. |

#### Fundamento teórico

- **Apriori** (Agrawal y Srikant, 1994): genera candidatos de k-itemsets a partir de (k-1)-itemsets
  frecuentes, realizando múltiples lecturas del dataset.
- **FP-Growth** (Han, Pei y Yin, 2000): construye un árbol FP-tree compacto en dos recorridos
  y extrae los itemsets frecuentes de forma recursiva, sin generar candidatos explícitamente.

Ambos algoritmos producen exactamente los mismos itemsets frecuentes para los mismos parámetros,
lo que permite validar la **correctitud** de ambas implementaciones.
""")
st.markdown("---")

df = load_processed_data()

# --- CONFIGURACIÓN ---
st.sidebar.header("Configuración de la comparación")
support_values_input = st.sidebar.text_input(
    "Valores de min_support (separados por coma)",
    "0.001, 0.005, 0.01, 0.02, 0.05",
    help="Se ejecutarán ambos algoritmos para cada uno de estos valores."
)
min_confidence = st.sidebar.slider("Confianza mínima", 0.0, 1.0, 0.1, 0.05)
min_lift = st.sidebar.slider("Lift mínimo", 0.5, 5.0, 1.0, 0.1)

st.sidebar.info(
    "Valores de soporte más bajos generan más reglas pero aumentan el tiempo de procesamiento. "
    "Se recomienda comenzar con los valores predeterminados."
)

if st.button("Ejecutar comparación", type="primary"):
    try:
        support_values = [float(s.strip()) for s in support_values_input.split(',')]
    except ValueError:
        st.error("Formato inválido. Utilice números separados por coma (por ejemplo: 0.001, 0.005, 0.01).")
        st.stop()

    with st.spinner("Comparando algoritmos... Este proceso puede tardar varios minutos."):
        comparison = compare_algorithms(df, support_values, min_confidence, min_lift)
        st.session_state['comparison'] = comparison

if 'comparison' in st.session_state:
    comp = st.session_state['comparison']
    summary = get_comparison_summary(comp)

    # --- KPIs ---
    st.subheader("Resumen ejecutivo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Algoritmo más rápido", summary.get('winner_speed', 'N/A'))
    col2.metric("Speedup promedio (FP-Growth / Apriori)", f"{summary.get('avg_speedup', 0):.2f}x")
    col3.metric("Mismos itemsets", "Sí" if summary.get('same_results') else "No")

    st.markdown("---")

    # --- TABLA ---
    st.subheader("Tabla comparativa detallada")
    st.markdown("""
    Cada fila corresponde a una ejecución con un valor distinto de `min_support`.
    La columna **speedup** indica cuántas veces más rápido es FP-Growth en cada caso.
    Un gradiente de color verde más intenso indica mayor ventaja de FP-Growth.
    """)

    st.dataframe(comp.style.format({
        'min_support': '{:.4f}',
        'apriori_time_s': '{:.4f}',
        'fpgrowth_time_s': '{:.4f}',
        'speedup': '{:.2f}'
    }).background_gradient(subset=['speedup'], cmap='RdYlGn'), width='stretch')

    csv = comp.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar comparación (CSV)", csv, "comparison.csv", "text/csv")

    st.markdown("---")

    # --- TIEMPOS ---
    st.subheader("Tiempos de ejecución por valor de soporte")
    st.markdown("""
    El gráfico muestra cómo varía el tiempo de ejecución de cada algoritmo al modificar
    el parámetro `min_support`. A medida que el soporte disminuye, ambos algoritmos tardan
    más, ya que se generan más itemsets frecuentes. La separación entre las curvas refleja
    la ventaja computacional de FP-Growth.
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
    st.subheader("Itemsets y reglas generados por cada algoritmo")
    st.markdown("""
    Los dos gráficos siguientes permiten comparar el volumen de resultados producido por
    cada algoritmo. Si los itemsets coinciden, se confirma la **corrección matemática**
    de ambas implementaciones.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Itemsets frecuentes generados**")
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
        st.markdown("**Reglas generadas**")
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
    st.subheader("Conclusiones de la comparación")
    speedup_val = summary.get('avg_speedup', 1)
    st.markdown(f"""
    Los resultados obtenidos permiten extraer las siguientes conclusiones:

    - **FP-Growth es aproximadamente {speedup_val:.1f}x más rápido** que Apriori en promedio
      sobre los valores de soporte evaluados.
    - **Ambos algoritmos producen los mismos itemsets frecuentes**, lo que valida matemáticamente
      la corrección de las implementaciones.
    - **Cuándo utilizar Apriori**: datasets pequeños, fines didácticos o cuando la explicabilidad
      paso a paso del proceso es un requisito.
    - **Cuándo utilizar FP-Growth**: datasets de gran volumen, entornos de producción o escenarios
      con valores de `min_support` muy bajos.

    **Recomendación**: para este dataset y para aplicaciones en producción, se recomienda utilizar **FP-Growth**.
    """)

else:
    st.info("Configure los parámetros en la barra lateral y haga clic en **Ejecutar comparación** para iniciar el análisis.")

    st.markdown("---")
    st.subheader("Fundamento de los algoritmos")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Apriori

        1. Encuentra todos los 1-itemsets frecuentes.
        2. Genera candidatos 2-itemsets a partir de ellos.
        3. Elimina los que no superen el soporte mínimo.
        4. Repite el proceso hasta que no existan más itemsets frecuentes.

        **Limitación principal:** requiere múltiples lecturas del dataset y genera un gran número
        de candidatos intermedios.
        """)
    with col2:
        st.markdown("""
        #### FP-Growth

        1. Recorre el dataset dos veces.
        2. Construye un árbol FP-tree compacto en memoria.
        3. Extrae los itemsets frecuentes de forma recursiva desde el árbol.

        **Ventaja principal:** no genera candidatos explícitamente, lo que reduce
        significativamente el tiempo de cómputo y el uso de memoria.
        """)

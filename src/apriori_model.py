"""
Implementación del algoritmo Apriori para Market Basket Analysis.
"""
import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(df_encoded: pd.DataFrame,
                min_support: float = 0.005,
                use_colnames: bool = True,
                max_len: int = None) -> tuple:
    """
    Ejecuta el algoritmo Apriori para encontrar itemsets frecuentes.
    
    Args:
        df_encoded: DataFrame one-hot encoded.
        min_support: Soporte mínimo (0 a 1).
        use_colnames: Si True, usa nombres de columnas en lugar de índices.
        max_len: Tamaño máximo de itemsets (None = sin límite).
    
    Returns:
        Tupla (frequent_itemsets, tiempo_ejecucion).
    """
    start_time = time.time()
    
    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=use_colnames,
        max_len=max_len
    )
    
    elapsed = time.time() - start_time
    
    # Añadir columna con número de items
    if not frequent_itemsets.empty:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    
    print(f"✅ Apriori: {len(frequent_itemsets)} itemsets frecuentes encontrados en {elapsed:.2f}s")
    return frequent_itemsets, elapsed


def generate_rules(frequent_itemsets: pd.DataFrame,
                   metric: str = 'lift',
                   min_threshold: float = 1.0) -> pd.DataFrame:
    """
    Genera reglas de asociación a partir de itemsets frecuentes.
    
    Args:
        frequent_itemsets: DataFrame con itemsets frecuentes.
        metric: Métrica para filtrar reglas ('support', 'confidence', 'lift').
        min_threshold: Umbral mínimo para la métrica.
    
    Returns:
        DataFrame con reglas de asociación.
    """
    if frequent_itemsets.empty:
        print("⚠️ No hay itemsets frecuentes para generar reglas.")
        return pd.DataFrame()
    
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold
    )
    
    # Ordenar por lift descendente
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    
    print(f"✅ Reglas generadas: {len(rules)}")
    return rules


def filter_rules(rules: pd.DataFrame,
                 min_support: float = 0.0,
                 min_confidence: float = 0.0,
                 min_lift: float = 1.0) -> pd.DataFrame:
    """
    Filtra reglas por umbrales mínimos.
    """
    filtered = rules[
        (rules['support'] >= min_support) &
        (rules['confidence'] >= min_confidence) &
        (rules['lift'] >= min_lift)
    ].copy()
    return filtered.sort_values('lift', ascending=False).reset_index(drop=True)


def get_rule_summary(rules: pd.DataFrame) -> dict:
    """
    Retorna un resumen estadístico de las reglas.
    """
    if rules.empty:
        return {}
    
    return {
        'total_rules': len(rules),
        'avg_support': rules['support'].mean(),
        'avg_confidence': rules['confidence'].mean(),
        'avg_lift': rules['lift'].mean(),
        'max_lift': rules['lift'].max(),
        'min_lift': rules['lift'].min(),
    }


if __name__ == '__main__':
    from src.utils import load_processed_data
    
    df = load_processed_data()
    itemsets, t = run_apriori(df, min_support=0.005)
    print(itemsets.head())
    
    rules = generate_rules(itemsets, metric='lift', min_threshold=1.0)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
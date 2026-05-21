"""
Implementación del algoritmo FP-Growth para Market Basket Analysis.
FP-Growth es más eficiente que Apriori al construir un árbol FP-tree.
"""
import pandas as pd
import time
from mlxtend.frequent_patterns import fpgrowth, association_rules


def run_fpgrowth(df_encoded: pd.DataFrame,
                 min_support: float = 0.005,
                 use_colnames: bool = True,
                 max_len: int = None) -> tuple:
    """
    Ejecuta el algoritmo FP-Growth para encontrar itemsets frecuentes.
    
    A diferencia de Apriori, FP-Growth:
    - No genera candidatos explícitamente.
    - Construye un FP-tree compacto.
    - Suele ser 2-10x más rápido en datasets grandes.
    
    Args:
        df_encoded: DataFrame one-hot encoded.
        min_support: Soporte mínimo (0 a 1).
        use_colnames: Si True, usa nombres de columnas.
        max_len: Tamaño máximo de itemsets.
    
    Returns:
        Tupla (frequent_itemsets, tiempo_ejecucion).
    """
    start_time = time.time()
    
    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=use_colnames,
        max_len=max_len
    )
    
    elapsed = time.time() - start_time
    
    if not frequent_itemsets.empty:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    
    print(f"⚡ FP-Growth: {len(frequent_itemsets)} itemsets frecuentes encontrados en {elapsed:.2f}s")
    return frequent_itemsets, elapsed


def generate_rules_fpgrowth(frequent_itemsets: pd.DataFrame,
                            metric: str = 'lift',
                            min_threshold: float = 1.0) -> pd.DataFrame:
    """
    Genera reglas de asociación a partir de itemsets frecuentes (FP-Growth).
    Usa la misma función association_rules de mlxtend.
    """
    if frequent_itemsets.empty:
        return pd.DataFrame()
    
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold
    )
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    print(f"✅ Reglas FP-Growth: {len(rules)}")
    return rules


if __name__ == '__main__':
    from src.utils import load_processed_data
    
    df = load_processed_data()
    itemsets, t = run_fpgrowth(df, min_support=0.005)
    print(itemsets.head())
    
    rules = generate_rules_fpgrowth(itemsets, metric='lift', min_threshold=1.0)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
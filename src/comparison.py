"""
Módulo de comparación entre Apriori y FP-Growth.
Mide tiempos de ejecución, cantidad de itemsets y reglas.
"""
import pandas as pd
import time
from src.apriori_model import run_apriori, generate_rules
from src.fpgrowth_model import run_fpgrowth, generate_rules_fpgrowth


def compare_algorithms(df_encoded: pd.DataFrame,
                       min_support_values: list = None,
                       min_confidence: float = 0.1,
                       min_lift: float = 1.0) -> pd.DataFrame:
    """
    Compara Apriori vs FP-Growth con diferentes valores de min_support.
    
    Args:
        df_encoded: DataFrame one-hot encoded.
        min_support_values: Lista de valores de soporte a probar.
        min_confidence: Confianza mínima para las reglas.
        min_lift: Lift mínimo para las reglas.
    
    Returns:
        DataFrame con la comparación.
    """
    if min_support_values is None:
        min_support_values = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    results = []
    
    for min_sup in min_support_values:
        print(f"\n🔬 Probando min_support = {min_sup}")
        
        # Apriori
        try:
            apriori_itemsets, apriori_time = run_apriori(df_encoded, min_support=min_sup)
            apriori_rules = generate_rules(apriori_itemsets, metric='confidence', min_threshold=min_confidence)
            apriori_rules = apriori_rules[apriori_rules['lift'] >= min_lift]
            n_apriori_itemsets = len(apriori_itemsets)
            n_apriori_rules = len(apriori_rules)
        except Exception as e:
            print(f"❌ Error en Apriori: {e}")
            apriori_time, n_apriori_itemsets, n_apriori_rules = None, 0, 0
        
        # FP-Growth
        try:
            fp_itemsets, fp_time = run_fpgrowth(df_encoded, min_support=min_sup)
            fp_rules = generate_rules_fpgrowth(fp_itemsets, metric='confidence', min_threshold=min_confidence)
            fp_rules = fp_rules[fp_rules['lift'] >= min_lift]
            n_fp_itemsets = len(fp_itemsets)
            n_fp_rules = len(fp_rules)
        except Exception as e:
            print(f"❌ Error en FP-Growth: {e}")
            fp_time, n_fp_itemsets, n_fp_rules = None, 0, 0
        
        speedup = (apriori_time / fp_time) if (apriori_time and fp_time and fp_time > 0) else None
        
        results.append({
            'min_support': min_sup,
            'apriori_time_s': round(apriori_time, 4) if apriori_time else None,
            'fpgrowth_time_s': round(fp_time, 4) if fp_time else None,
            'speedup': round(speedup, 2) if speedup else None,
            'apriori_itemsets': n_apriori_itemsets,
            'fpgrowth_itemsets': n_fp_itemsets,
            'apriori_rules': n_apriori_rules,
            'fpgrowth_rules': n_fp_rules,
        })
    
    df_comp = pd.DataFrame(results)
    print("\n📊 RESULTADOS DE COMPARACIÓN")
    print(df_comp.to_string(index=False))
    return df_comp


def get_comparison_summary(df_comp: pd.DataFrame) -> dict:
    """
    Resumen ejecutivo de la comparación.
    """
    avg_speedup = df_comp['speedup'].mean()
    
    return {
        'avg_speedup': round(avg_speedup, 2) if pd.notna(avg_speedup) else None,
        'winner_speed': 'FP-Growth' if avg_speedup and avg_speedup > 1 else 'Apriori',
        'same_results': bool((df_comp['apriori_itemsets'] == df_comp['fpgrowth_itemsets']).all()),
    }


if __name__ == '__main__':
    from src.utils import load_processed_data
    df = load_processed_data()
    comparison = compare_algorithms(df)
    print(get_comparison_summary(comparison))
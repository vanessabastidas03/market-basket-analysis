"""
Módulo de preprocesamiento de datos para Market Basket Analysis.
Transforma el dataset crudo en formato de transacciones one-hot encoded.
"""
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from pathlib import Path
import pickle


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset crudo desde un archivo CSV.
    
    Args:
        filepath: Ruta al archivo CSV.
    
    Returns:
        DataFrame con los datos crudos.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset:
    - Elimina valores nulos.
    - Elimina duplicados.
    - Normaliza nombres de productos (minúsculas, sin espacios extra).
    - Convierte la fecha a formato datetime.
    
    Args:
        df: DataFrame crudo.
    
    Returns:
        DataFrame limpio.
    """
    df = df.copy()
    
    # Eliminar nulos
    nulos_iniciales = df.isnull().sum().sum()
    df = df.dropna()
    print(f"🧹 Valores nulos eliminados: {nulos_iniciales}")
    
    # Eliminar duplicados exactos
    duplicados_iniciales = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"🧹 Duplicados eliminados: {duplicados_iniciales}")
    
    # Normalizar nombres de productos
    df['itemDescription'] = df['itemDescription'].str.lower().str.strip()
    
    # Convertir fecha
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    print(f"✅ Dataset limpio: {df.shape[0]} filas")
    return df


def create_transactions(df: pd.DataFrame) -> list:
    """
    Agrupa los productos por cliente y fecha para crear transacciones.
    Una transacción = productos comprados por un cliente en una fecha.
    
    Args:
        df: DataFrame limpio.
    
    Returns:
        Lista de listas, donde cada lista interna es una transacción.
    """
    transactions = (
        df.groupby(['Member_number', 'Date'])['itemDescription']
          .apply(list)
          .tolist()
    )
    print(f"🛒 Transacciones creadas: {len(transactions)}")
    print(f"📦 Tamaño promedio de transacción: {np.mean([len(t) for t in transactions]):.2f}")
    return transactions


def encode_transactions(transactions: list) -> pd.DataFrame:
    """
    Convierte las transacciones a formato one-hot encoded.
    Cada fila = una transacción, cada columna = un producto (True/False).
    
    Args:
        transactions: Lista de listas de productos.
    
    Returns:
        DataFrame one-hot encoded.
    """
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    print(f"✅ Encoding completado: {df_encoded.shape[0]} transacciones × {df_encoded.shape[1]} productos")
    return df_encoded


def save_processed_data(df_encoded: pd.DataFrame, transactions: list, output_dir: str = 'data/processed'):
    """
    Guarda los datos procesados en formato pickle para uso posterior.
    
    Args:
        df_encoded: DataFrame one-hot encoded.
        transactions: Lista de transacciones.
        output_dir: Directorio de salida.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df_encoded.to_pickle(f'{output_dir}/transactions_encoded.pkl')
    with open(f'{output_dir}/transactions_list.pkl', 'wb') as f:
        pickle.dump(transactions, f)
    
    print(f"💾 Datos guardados en {output_dir}/")


def run_preprocessing(input_path: str = 'data/raw/Groceries_dataset.csv',
                      output_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento.
    
    Args:
        input_path: Ruta al CSV crudo.
        output_dir: Directorio de salida.
    
    Returns:
        DataFrame one-hot encoded.
    """
    print("=" * 60)
    print("🔄 INICIANDO PREPROCESAMIENTO")
    print("=" * 60)
    
    df = load_raw_data(input_path)
    df_clean = clean_data(df)
    transactions = create_transactions(df_clean)
    df_encoded = encode_transactions(transactions)
    save_processed_data(df_encoded, transactions, output_dir)
    
    print("=" * 60)
    print("✅ PREPROCESAMIENTO FINALIZADO")
    print("=" * 60)
    return df_encoded


if __name__ == '__main__':
    run_preprocessing()
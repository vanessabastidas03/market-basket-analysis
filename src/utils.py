"""
Funciones auxiliares para el proyecto.
"""
import pandas as pd
import pickle
from pathlib import Path
import streamlit as st


def _regenerate():
    from src.data_preprocessing import run_preprocessing
    run_preprocessing()


@st.cache_data
def load_processed_data(filepath: str = 'data/processed/transactions_encoded.pkl') -> pd.DataFrame:
    """
    Carga el DataFrame one-hot encoded procesado.
    Si el archivo no existe o no puede deserializarse, regenera desde el CSV.
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(filepath)
        return pd.read_pickle(filepath)
    except Exception:
        _regenerate()
        return pd.read_pickle(filepath)


@st.cache_data
def load_transactions_list(filepath: str = 'data/processed/transactions_list.pkl') -> list:
    """
    Carga la lista de transacciones.
    Si el archivo no existe o no puede deserializarse, regenera desde el CSV.
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(filepath)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        _regenerate()
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def format_itemset(itemset) -> str:
    """
    Formatea un frozenset/set de productos como string legible.
    """
    if isinstance(itemset, (frozenset, set)):
        return ', '.join(sorted(itemset))
    return str(itemset)


def get_top_products(df_encoded: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """
    Obtiene los N productos más vendidos.
    """
    return df_encoded.sum().sort_values(ascending=False).head(top_n)
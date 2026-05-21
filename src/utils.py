"""
Funciones auxiliares para el proyecto.
"""
import pandas as pd
import pickle
from pathlib import Path
import streamlit as st


@st.cache_data
def load_processed_data(filepath: str = 'data/processed/transactions_encoded.pkl') -> pd.DataFrame:
    """
    Carga el DataFrame one-hot encoded procesado.
    Usa cache de Streamlit para no recargar en cada interacción.
    """
    return pd.read_pickle(filepath)


@st.cache_data
def load_transactions_list(filepath: str = 'data/processed/transactions_list.pkl') -> list:
    """
    Carga la lista de transacciones.
    """
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
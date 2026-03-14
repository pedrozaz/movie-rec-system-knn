import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple

def create_user_item_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame, pd.Series]:
    """
    Transforms DataFrame into normalized sparse matrix.
    """
    matrix_df = df.pivot_table(index='user_id', columns='movie_id', values='rating')
    user_means = matrix_df.mean(axis=1)
    matrix_centered = matrix_df.sub(user_means, axis=0).fillna(0)
    sparse_matrix = csr_matrix(matrix_centered.values)

    return sparse_matrix, matrix_centered, user_means

def calculate_sparsity(df: pd.DataFrame) -> float:
    """
    Calculates NaN values in sparse matrix.
    """
    total_cells = df.size
    non_zero_cells = df.count().sum()
    sparsity = (1 - (non_zero_cells / total_cells)) * 100

    return sparsity
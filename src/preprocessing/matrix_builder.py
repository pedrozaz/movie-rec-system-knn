from typing import Tuple

import pandas as pd
from scipy.sparse import csr_matrix


def create_user_item_matrix(df: pd.DataFrame, is_item_based: bool = False) -> Tuple[csr_matrix, pd.DataFrame, pd.Series]:
    matrix_df = df.pivot_table(index='user_id', columns='movie_id', values='rating')

    if is_item_based:
        matrix_df = matrix_df.T
        means = matrix_df.mean(axis=1)
    else:
        means = matrix_df.mean(axis=1)

    matrix_centered = matrix_df.sub(means, axis=0).fillna(0)
    sparse_matrix = csr_matrix(matrix_centered.values)

    return sparse_matrix, matrix_centered, means

def calculate_sparsity(df: pd.DataFrame) -> float:
    """
    Calculates NaN values in sparse matrix.
    """
    total_cells = df.size
    non_zero_cells = df.count().sum()
    sparsity = (1 - (non_zero_cells / total_cells)) * 100

    return sparsity
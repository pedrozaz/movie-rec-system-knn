import pandas as pd
import os
from typing import Tuple

def load_movielens_100k(data_path: str = 'data/raw') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads MovieLens 100k dataset
    """
    ratings_path = os.path.join(data_path, 'u.data')
    items_path = os.path.join(data_path, 'u.item')

    if not os.path.exists(ratings_path) or not os.path.exists(items_path):
        raise FileNotFoundError(f"Missing files in {data_path}. Please download u.data and u.item.")

    df_ratings = pd.read_csv(
        ratings_path,
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )

    df_items = pd.read_csv(
        items_path,
        sep='|',
        names=['movie_id', 'title'],
        usecols=[0, 1],
        encoding='latin-1'
    )

    return df_ratings, df_items

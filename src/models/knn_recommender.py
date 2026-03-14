from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

class KNNRecommender:
    def __init__(self, n_neighbors: int = 20, metric: str = 'cosine'):
        self.model = NearestNeighbors(metric=metric, algorithm='brute', n_neighbors=n_neighbors)
        self.matrix = None

    def fit(self, matrix: csr_matrix):
        self.matrix = matrix
        self.model.fit(matrix)

    def get_similar_users(self, user_index: int):
        distances, indices = self.model.kneighbors(
            self.matrix[user_index],
            n_neighbors=self.model.n_neighbors + 1
        )

        return distances.flatten()[1:], indices.flatten()[1:]

    def recommend(self, user_idx: int, matrix_df: pd.DataFrame, user_means: pd.Series, n_recs: int = 5):
        distances, indices = self.get_similar_users(user_idx)
        similarities = 1 - distances

        user_ratings = matrix_df.iloc[user_idx]
        already_watched = user_ratings[user_ratings != 0].index

        neighbor_matrices = matrix_df.iloc[indices]
        weighted_ratings = neighbor_matrices.values * similarities[:,np.newaxis]

        sum_of_similarities = np.abs(similarities).sum()
        prediction_scores = weighted_ratings.sum(axis=0) / (sum_of_similarities + 1e-9)

        recommendations = pd.Series(prediction_scores, index=matrix_df.columns)
        recommendations = recommendations.drop(index=already_watched)
        final_predictions = recommendations + user_means.iloc[user_idx]

        return final_predictions.sort_values(ascending=False).head(n_recs)
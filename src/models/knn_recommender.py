from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np

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
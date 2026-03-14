import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List

class Evaluator:
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        mse = mean_squared_error(actual, predicted)
        return np.sqrt(mse)

    @staticmethod
    def precision_at_k(actual_ratings: dict, predicted_ratings: List[tuple], k: int, threshold: float = 4.0) -> float:
        """
        Calculates the Precision@K
        :param actual_ratings: dict {movie_id: rating_real} from test set
        :param predicted_ratings: tuple list [(movie_id, score), ...]
        """
        top_k = predicted_ratings[:k]

        relevant_and_recommended = 0
        for movie_id, score in top_k:
            if movie_id not in actual_ratings and actual_ratings[movie_id] >= threshold:
                relevant_and_recommended += 1

        return relevant_and_recommended / k

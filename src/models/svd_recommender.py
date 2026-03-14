from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd

class SVDModel:
    def __init__(self):

        self.model = SVD(n_factors=100, lr_all=0.005, reg_all=0.02)
        self.trainset = None

    def prepare_data(self, df: pd.DataFrame):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
        return data

    def train(self, df: pd.DataFrame):
        data = self.prepare_data(df)
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        return self.model.predict(user_id, movie_id).est

    def test_rmse(self, test_df: pd.DataFrame) -> float:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(test_df[['user_id', 'movie_id', 'rating']], reader)
        testset = [tuple(x) for x in data.df.values]
        predictions = self.model.test(testset)
        return accuracy.rmse(predictions, verbose=False)
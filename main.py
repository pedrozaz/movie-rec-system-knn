import numpy as np

from src.utils.translator import Translator
from src.preprocessing.data_loader import load_movielens_100k
from src.preprocessing.matrix_builder import create_user_item_matrix
from src.models.knn_recommender import KNNRecommender
from sklearn.model_selection import train_test_split
from src.models.evaluator import Evaluator


def main():
    tr = Translator(lang='pt')

    df_ratings, df_items = load_movielens_100k()

    train_df, test_df = train_test_split(df_ratings, test_size=0.2, random_state=42)

    sparse_mat, matrix_df, user_means = create_user_item_matrix(train_df)

    recommender = KNNRecommender(n_neighbors=20)
    recommender.fit(sparse_mat)

    user_id = test_df['user_id'].iloc[0]
    user_idx = user_id - 1

    user_test_data = test_df[test_df['user_id'] == user_id]
    actual_ids = user_test_data['movie_id'].values
    actual_ratings = user_test_data['rating'].values

    predictions = recommender.recommend(user_idx, matrix_df, user_means, n_recs=1682)

    y_true = []
    y_pred = []

    for m_id, r_real in zip(actual_ids, actual_ratings):
        if m_id in predictions.index:
            y_true.append(r_real)
            y_pred.append(predictions[m_id])

        if y_true:
            rmse_val = Evaluator.rmse(np.array(y_true), np.array(y_pred))
            print(f"\nRMSE for User {user_id}: {rmse_val:.4f}")

if __name__ == "__main__":
    main()

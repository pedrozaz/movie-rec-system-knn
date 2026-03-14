from src.utils.translator import Translator
from src.preprocessing.data_loader import load_movielens_100k
from src.preprocessing.matrix_builder import create_user_item_matrix
from src.models.knn_recommender import KNNRecommender


def main():
    tr = Translator(lang='pt')

    try:
        df_ratings, df_items = load_movielens_100k()
        sparse_mat, matrix_df, user_means = create_user_item_matrix(df_ratings)

        recommender = KNNRecommender(n_neighbors=20)
        recommender.fit(sparse_mat)

        user_id = 1
        user_idx = user_id - 1
        n_recs = 5

        print(tr.get('generating_recs', user_id=user_id))

        tops_recs = recommender.recommend(user_idx, matrix_df, user_means, n_recs=n_recs)

        print(tr.get('top_n_title', n=n_recs))

        for rank, (movie_id, rating) in enumerate(tops_recs.items(), 1):
            title = df_items[df_items['movie_id'] == movie_id]['title'].values[0]
            print(tr.get('movie_display', rank=rank, title=title, rating=rating))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

from src.utils.translator import Translator
from src.preprocessing.data_loader import load_movielens_100k
from src.preprocessing.matrix_builder import create_user_item_matrix, calculate_sparsity
from src.models.knn_recommender import KNNRecommender


def main():
    tr = Translator(lang='pt')

    try:
        df_ratings, df_items = load_movielens_100k()
        sparse_mat, matrix_df, user_means = create_user_item_matrix(df_ratings)

        metric = 'cosine'
        print(tr.get('model_training', metric=metric))

        recommender = KNNRecommender(n_neighbors=10, metric=metric)
        recommender.fit(sparse_mat)

        user_id_to_test = 1
        user_idx = user_id_to_test - 1

        print(tr.get('finding_neighbors', user_id=user_id_to_test))
        distances, indices = recommender.get_similar_users(user_idx)

        print(tr.get('neighbors_found', count=len(indices)))

        for i in range(len(indices)):
            similarity = 1 - distances[i]
            print(f"Vizinho: Usuário {indices[i] + 1} | Similaridade: {similarity:.4f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

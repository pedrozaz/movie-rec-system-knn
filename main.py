from src.utils.translator import Translator
from src.preprocessing.data_loader import load_movielens_100k
from src.preprocessing.matrix_builder import create_user_item_matrix, calculate_sparsity


def main():
    tr = Translator(lang='pt')

    try:
        df_ratings, df_items = load_movielens_100k()
        print(tr.get('data_info', rows=len(df_ratings)))

        print(tr.get('creating_matrix'))

        sparsity = calculate_sparsity(df_ratings.pivot_table(index='user_id', columns='movie_id', values='rating'))
        print(tr.get('sparsity_info', sparsity=sparsity))

        sparse_mat, matrix_df, user_means = create_user_item_matrix(df_ratings)

        print(tr.get('matrix_info', users=sparse_mat.shape[0], movies=sparse_mat.shape[1]))

        print("\nMatriz Normalizada (amostra):")
        print(matrix_df.iloc[:5, :5])

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

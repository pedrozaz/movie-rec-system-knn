from src.utils.translator import Translator
from src.preprocessing.data_loader import load_movielens_100k


def main():
    lang = 'pt'
    tr = Translator(lang)

    print(f"--- {tr.get('loading_data')} ---")

    try:
        df_ratings, df_items = load_movielens_100k()
        print(tr.get('data_info', rows=len(df_ratings)))

        print("\nTop 5 Ratings:")
        print(df_ratings.head())

        print("\nTop 5 Movies:")
        print(df_items.head())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    main()

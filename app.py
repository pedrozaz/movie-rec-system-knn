import streamlit as st
import pandas as pd
from src.utils.translator import Translator
from src.preprocessing.data_loader import load_movielens_100k
from src.preprocessing.matrix_builder import create_user_item_matrix
from src.models.knn_recommender import KNNRecommender
from src.models.svd_recommender import SVDModel

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

@st.cache_resource
def get_data_and_models():
    df_ratings, df_items = load_movielens_100k()

    sparse_user, matrix_user, user_means = create_user_item_matrix(df_ratings)
    knn_model = KNNRecommender(n_neighbors=20)
    knn_model.fit(sparse_user)

    svd_model = SVDModel()
    svd_model.train(df_ratings)

    return df_ratings, df_items, knn_model, matrix_user, user_means, svd_model

st.sidebar.title("Settings / Configurações")
lang = st.sidebar.selectbox("Language / Idioma", ["pt", "en"])
tr = Translator(lang=lang)

df_ratings, df_items, knn_model, matrix_user, user_means, svd_model = get_data_and_models()

st.title(tr.get('comparing_models'))

user_id = st.number_input("User ID", min_value=1, max_value=943, value=1)
n_recs = st.slider("Number of recommendations", 5, 20, 10)

col1, col2 = st.columns(2)

with col1:
    st.subheader("User-Based KNN")
    if st.button(f"Run KNN for User {user_id}"):
        user_idx = user_id - 1
        recs = knn_model.recommend(user_idx, matrix_user, user_means, n_recs=n_recs)

        for movie_id, rating in recs.items():
            title = df_items[df_items['movie_id'] == movie_id]['title'].values[0]
            st.write(f"**{title}**")
            st.caption(f"Rating: {rating:.2f}")

with col2:
    st.subheader("SVD (Matrix Factorization)")
    if st.button("Compare with SVD"):
        user_idx = user_id - 1
        recs = knn_model.recommend(user_idx, matrix_user, user_means, n_recs=n_recs)

        for movie_id, _ in recs.items():
            title = df_items[df_items['movie_id'] == movie_id]['title'].values[0]
            svd_pred = svd_model.predict_rating(user_id, movie_id)
            st.write(f"**{title}**")
            st.caption(f"SVD Rating: {svd_pred:.2f}")
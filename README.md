# Movie Recommendation System

This project implements and compares different Collaborative Filtering
 approaches using the **MovieLens 100** dataset. Developed for
study purposes, the systems explores neighborhood-based methods
(KNN) and latent factor models (SVD).

## Overview
The goal is to predict user ratings for movies and generate a Top-N
recommendation list. The system addresses the "sparsity problem" (93.7% in this dataset)
through mean-centering and matrix factorization.

### Key Features
- **User-Based KNN**: Finding similar users via Cosine Similarity.
- **Item-Based KNN**: Transposed matrix approach to find movie-to-movie relationships.
- **SVD (Singular Value Decomposition)**: Latent factor model to reduce dimensionality and improve RMSE.
- **Internationalization (i18n)**: Full support for English and Portuguese.
- **Interactive UI**: Built with Streamlit for real-time recommendations.

## Dataset
- **Source**: [GroupLens - MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- **Composition**: 100,000 ratings (1-5) from 943 users on 1,682 movies.
- **Sparsity**: ~93.7%.

## Methodology
### 1. Preprocessing
- **Mean-Centering**: Subtraction of each user's average rating to remove individual bias (optimists vs. pessimists).
- **Sparse Matrix**: Conversion to SciPy `csr_matrix` for memory efficiency.

### 2. Algorithms
- **KNN (K-Nearest Neighbors)**:
  - Metric: Cosine Similarity.
  - Logic: Weighted average of neighbors' ratings.
- **SVD (Matrix Factorization)**:
  - Library: `scikit-surprise`.
  - Factors: 100 latent factors.
  - Optimization: Stochastic Gradient Descent (SGD).

## Evaluation
Evaluation is performed using an 80/20 train-test split.

| Model | RMSE | Metric |
| :--- | :--- | :--- |
| KNN (User-Based) | ~1.01 | Cosine Similarity |
| SVD | **0.9323** | Latent Factors |

## Setup & Execution
This project uses [uv](https://astral.sh/uv) for high-performance Python package management.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pedrozaz/movie-rec-system-knn.git
   cd movie-rec-system-knn

2. **Install dependencies**:
    ```bash
    uv sync
    ```

3. **Download the dataset**:
    Place u.data and u.item in data/raw/

4. **Run the dashboard**:
    ```bash
   uv run streamlit run app.py
   ```

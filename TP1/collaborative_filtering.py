"""
TP1 - Filtrage Collaboratif Item-Item (top-N)
Algorithme :
  1. Construire la matrice utilisateurs × items
  2. Calculer la similarité cosinus entre chaque paire d'items
  3. Pour un utilisateur donné, prédire les notes des items non vus
  4. Retourner le top-N
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ─────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
def download_movielens():
    """Télécharge MovieLens Small (1Mo) si pas déjà présent."""
    ratings_path = os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")
    movies_path  = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
    if os.path.exists(ratings_path) and os.path.exists(movies_path):
        return ratings_path, movies_path

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")
    print("Téléchargement du dataset MovieLens Small...")
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print("Dataset prêt.")
    return ratings_path, movies_path


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retourne :
      - ratings  : userId, movieId, rating
      - movies   : movieId, title, genres
      - matrix   : DataFrame pivot (userId × movieId), NaN = non noté
    """
    ratings_path, movies_path = download_movielens()
    ratings = pd.read_csv(ratings_path)
    movies  = pd.read_csv(movies_path)

    # Matrice utilisateurs × films (NaN pour les films non notés)
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    return ratings, movies, matrix


# ─────────────────────────────────────────
# SIMILARITÉ ITEM-ITEM
# ─────────────────────────────────────────
def compute_item_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la similarité cosinus entre tous les films.
    On remplace les NaN par 0 (un utilisateur qui n'a pas noté = neutre).
    Retourne un DataFrame (movieId × movieId).
    """
    filled = matrix.fillna(0).values  # shape : (n_users, n_items)
    sim_matrix = cosine_similarity(filled.T)  # transposée → items × items
    return pd.DataFrame(sim_matrix, index=matrix.columns, columns=matrix.columns)


# ─────────────────────────────────────────
# PRÉDICTION DE NOTE
# ─────────────────────────────────────────
def predict_rating(user_ratings: pd.Series, item_id: int, sim_df: pd.DataFrame, k: int = 20) -> float:
    """
    Prédit la note que l'utilisateur donnerait à `item_id`.
    Formule : sum(sim(i,j) * r_j) / sum(|sim(i,j)|)
    On prend les k voisins les plus similaires parmi les films notés.
    """
    # Films déjà notés par l'utilisateur
    rated_items = user_ratings.dropna()
    if rated_items.empty:
        return 0.0

    # Similarités entre item_id et tous les films notés
    sims = sim_df.loc[item_id, rated_items.index]

    # On garde les k plus similaires (>0)
    top_k = sims.nlargest(k)
    top_k = top_k[top_k > 0]
    if top_k.empty:
        return 0.0

    numerator   = (top_k * rated_items[top_k.index]).sum()
    denominator = top_k.abs().sum()
    return numerator / denominator if denominator != 0 else 0.0


# ─────────────────────────────────────────
# TOP-N RECOMMANDATIONS
# ─────────────────────────────────────────
def recommend(
    user_id: int,
    matrix: pd.DataFrame,
    sim_df: pd.DataFrame,
    movies: pd.DataFrame,
    n: int = 10,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """
    Retourne les N meilleures recommandations pour `user_id`.
    Exclut les films déjà notés par l'utilisateur.
    """
    if user_id not in matrix.index:
        return pd.DataFrame()

    user_ratings = matrix.loc[user_id]

    # Films non notés par cet utilisateur
    unseen_items = user_ratings[user_ratings.isna()].index

    # Prédiction pour chaque film non vu
    predictions = {
        item_id: predict_rating(user_ratings, item_id, sim_df, k=k_neighbors)
        for item_id in unseen_items
        if item_id in sim_df.index
    }

    top_n = (
        pd.Series(predictions)
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top_n.columns = ["movieId", "predicted_rating"]

    # Ajout des métadonnées (titre, genre)
    result = top_n.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    result["predicted_rating"] = result["predicted_rating"].round(2)
    return result


# ─────────────────────────────────────────
# POINT D'ENTRÉE RAPIDE (test CLI)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Chargement des données...")
    ratings, movies, matrix = load_data()
    print(f"  {ratings.shape[0]:,} notes | {matrix.shape[0]} utilisateurs | {matrix.shape[1]} films")

    print("Calcul de la matrice de similarité item-item...")
    sim_df = compute_item_similarity(matrix)

    user_id = int(input("Entrez un userId (1-610) : "))
    n = int(input("Combien de recommandations ? "))

    recs = recommend(user_id, matrix, sim_df, movies, n=n)
    print(f"\nTop {n} recommandations pour l'utilisateur {user_id} :\n")
    print(recs[["title", "genres", "predicted_rating"]].to_string(index=False))

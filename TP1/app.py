"""
TP1 - Application Streamlit : Système de recommandation Item-Item
Lance avec : streamlit run app.py
"""

import streamlit as st
import pandas as pd
from collaborative_filtering import load_data, compute_item_similarity, recommend

# ─────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Film Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Système de recommandation de films")
st.caption("Filtrage collaboratif Item-Item — Dataset MovieLens")

# ─────────────────────────────────────────
# CHARGEMENT (mis en cache pour ne pas recharger à chaque interaction)
# ─────────────────────────────────────────
@st.cache_data(show_spinner="Chargement du dataset MovieLens...")
def get_data():
    return load_data()


@st.cache_data(show_spinner="Calcul de la matrice de similarité item-item...")
def get_similarity(_matrix):
    return compute_item_similarity(_matrix)


ratings, movies, matrix = get_data()
sim_df = get_similarity(matrix)

# ─────────────────────────────────────────
# SIDEBAR — PARAMÈTRES
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")

    user_id = st.selectbox(
        "Utilisateur",
        options=sorted(matrix.index.tolist()),
        help="Choisissez un utilisateur (1 à 610)",
    )

    n_recs = st.slider(
        "Nombre de recommandations",
        min_value=5, max_value=50, value=10, step=5,
    )

    k_neighbors = st.slider(
        "Nombre de voisins (k)",
        min_value=5, max_value=50, value=20, step=5,
        help="Nombre de films similaires utilisés pour prédire la note",
    )

    st.divider()
    st.markdown(f"**Dataset stats**")
    st.metric("Notes totales", f"{ratings.shape[0]:,}")
    st.metric("Utilisateurs", matrix.shape[0])
    st.metric("Films", matrix.shape[1])

# ─────────────────────────────────────────
# PROFIL UTILISATEUR
# ─────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Profil de l'utilisateur {user_id}")
    user_ratings = matrix.loc[user_id].dropna().sort_values(ascending=False)
    n_rated = len(user_ratings)
    avg_rating = user_ratings.mean()

    m1, m2 = st.columns(2)
    m1.metric("Films notés", n_rated)
    m2.metric("Note moyenne", f"{avg_rating:.2f} / 5")

    top_rated = (
        user_ratings.head(10)
        .reset_index()
        .merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    )
    top_rated.columns = ["movieId", "Note", "Titre", "Genre"]
    st.markdown("**Films les mieux notés par cet utilisateur :**")
    st.dataframe(
        top_rated[["Titre", "Genre", "Note"]],
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────
# RECOMMANDATIONS
# ─────────────────────────────────────────
with col2:
    st.subheader(f"Top {n_recs} recommandations")

    with st.spinner("Calcul des recommandations..."):
        recs = recommend(user_id, matrix, sim_df, movies, n=n_recs, k_neighbors=k_neighbors)

    if recs.empty:
        st.warning("Pas de recommandations disponibles pour cet utilisateur.")
    else:
        # Barre de score colorée
        recs["Score"] = recs["predicted_rating"]

        st.dataframe(
            recs[["title", "genres", "Score"]].rename(
                columns={"title": "Titre", "genres": "Genre", "Score": "Score prédit"}
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score prédit": st.column_config.ProgressColumn(
                    "Score prédit",
                    min_value=0,
                    max_value=5,
                    format="%.2f",
                )
            },
        )

# ─────────────────────────────────────────
# SECTION PÉDAGOGIQUE — Comment ça marche ?
# ─────────────────────────────────────────
with st.expander("Comment fonctionne l'algorithme ?"):
    st.markdown("""
### Filtrage collaboratif Item-Item

**Idée centrale :** Si deux films sont souvent appréciés par les mêmes utilisateurs,
ils sont *similaires*. On peut alors recommander un film similaire à ceux qu'un utilisateur a aimés.

**Étapes :**

1. **Matrice utilisateurs × films** — chaque cellule est la note donnée (NaN si pas noté).

2. **Similarité cosinus** entre films — on compare les colonnes de la matrice :
   $$sim(i, j) = \\frac{\\vec{r_i} \\cdot \\vec{r_j}}{||\\vec{r_i}|| \\cdot ||\\vec{r_j}||}$$

3. **Prédiction de note** pour un film non vu :
   $$\\hat{r}_{u,i} = \\frac{\\sum_{j \\in N(i)} sim(i,j) \\cdot r_{u,j}}{\\sum_{j \\in N(i)} |sim(i,j)|}$$
   où $N(i)$ = les k films les plus similaires à $i$ que l'utilisateur a déjà notés.

4. **Top-N** — on trie par note prédite décroissante et on retourne les N premiers.
    """)

st.divider()
st.caption("TP1 Big Data — Master 1 | Dataset : [MovieLens Small](https://grouplens.org/datasets/movielens/latest/)")

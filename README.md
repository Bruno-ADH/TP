# Big Data — TPs Master 1

## TP0 — Word Count MapReduce (PySpark)

**Objectif :** Compter la fréquence de chaque mot dans le corpus Gutenberg (>1 Go) en utilisant le paradigme MapReduce via PySpark.

### Prérequis

```bash
pip install pyspark
```

### Lancer le TP

```bash
cd TP0

# 1. Télécharger les livres Gutenberg
python download_gutenberg.py --books 100

# 2. Lancer le word count
python word_count_spark.py gutenberg_data/ --top 30
```

### Fichiers

| Fichier | Rôle |
|---|---|
| `word_count_spark.py` | Programme principal MapReduce (PySpark) |
| `download_gutenberg.py` | Télécharge les livres Gutenberg |
| `requirements.txt` | Dépendances Python |

### Sorties

- Affichage du top-N dans le terminal
- Export CSV complet dans `word_counts_result/`

---

## TP1 — Système de recommandation (Filtrage Collaboratif Item-Item)

**Objectif :** Implémenter un système de recommandation de films basé sur le filtrage collaboratif item-item et l'exposer via une interface Streamlit.

**Dataset :** MovieLens Small (100 000 notes, 610 utilisateurs, 9 000 films) — téléchargé automatiquement au premier lancement.

### Prérequis

```bash
cd TP1
pip install -r requirements.txt
```

### Lancer le TP

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur. Sélectionner un utilisateur et régler les paramètres dans la barre latérale.

### Fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit |
| `collaborative_filtering.py` | Algorithme item-item (similarité cosinus, top-N) |
| `requirements.txt` | Dépendances Python |

### Algorithme

1. Construction de la matrice utilisateurs × films
2. Calcul de la similarité cosinus entre items
3. Prédiction de note par moyenne pondérée des k voisins
4. Retour des N films avec le score prédit le plus élevé

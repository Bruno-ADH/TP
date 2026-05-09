# TP0 — Word Count MapReduce (PySpark)

**Objectif :** Compter la fréquence de chaque mot dans le corpus Gutenberg (>1 Go) en utilisant le paradigme MapReduce via PySpark.

## Prérequis

```bash
pip install pyspark
```

## Lancer le TP

```bash
# 1. Télécharger les livres Gutenberg
python download_gutenberg.py --books 100

# 2. Lancer le word count
python word_count_spark.py gutenberg_data/ --top 30
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `word_count_spark.py` | Programme principal MapReduce (PySpark) |
| `download_gutenberg.py` | Télécharge les livres Gutenberg |
| `requirements.txt` | Dépendances Python |

## Sorties

- Affichage du top-N dans le terminal
- Export CSV trié dans `gutenberg_data/word_counts.csv`

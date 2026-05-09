"""
TP0 - Word Count MapReduce avec PySpark
Dataset : Corpus Gutenberg (>1 Go)

Usage:
    python word_count_spark.py <chemin_dossier_ou_fichier> [--top N]
    python word_count_spark.py gutenberg_data/ --top 30
"""

import re
import sys
import argparse
import time
from pyspark import SparkContext, SparkConf


def main():
    parser = argparse.ArgumentParser(description="Word Count MapReduce — PySpark")
    parser.add_argument("path", help="Dossier ou fichier texte à analyser")
    parser.add_argument("--top", type=int, default=20, help="Nombre de mots à afficher (défaut : 20)")
    args = parser.parse_args()

    # ── Initialisation Spark ──────────────────────────────────────────────────
    conf = (
        SparkConf()
        .setAppName("TP0_WordCount_Gutenberg")
        .setMaster("local[*]")           # local[*] = utilise tous les cœurs du CPU
        .set("spark.driver.memory", "2g")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")             # masque les logs verbeux de Spark

    print(f"\n{'='*55}")
    print(f"  TP0 - Word Count MapReduce (PySpark)")
    print(f"  Fichiers : {args.path}")
    print(f"{'='*55}\n")

    t0 = time.time()

    # ── Lecture ───────────────────────────────────────────────────────────────
    # sc.textFile() retourne un RDD où chaque élément = une ligne du fichier.
    # Le chemin peut contenir des wildcards : "data/*.txt"
    path = args.path.rstrip("/\\") + "/*" if not args.path.endswith(".txt") else args.path
    lines = sc.textFile(path)

    # ── PHASE MAP ─────────────────────────────────────────────────────────────
    # flatMap : pour chaque ligne, on émet autant de paires (mot, 1) qu'il y a de mots.
    # "le chat le" → [("le", 1), ("chat", 1), ("le", 1)]
    word_pairs = lines.flatMap(
        lambda line: [
            (word, 1)
            for word in re.findall(r"[a-zA-ZÀ-ÿ]+", line.lower())
        ]
    )

    # ── PHASE SHUFFLE / SORT + REDUCE ─────────────────────────────────────────
    # reduceByKey regroupe par clé (le mot) et applique la fonction d'agrégation.
    # Spark fait automatiquement une pré-agrégation locale (= combiner)
    # AVANT de transférer les données sur le réseau → très efficace.
    word_counts = word_pairs.reduceByKey(lambda count_a, count_b: count_a + count_b)

    # ── TRI et collecte ───────────────────────────────────────────────────────
    # sortBy retourne un RDD trié. takeOrdered prend les N plus grands sans tout charger.
    top_n = word_counts.takeOrdered(args.top, key=lambda x: -x[1])

    t1 = time.time()

    # ── Affichage ─────────────────────────────────────────────────────────────
    total_words = word_pairs.count()
    distinct_words = word_counts.count()

    print(f"Temps de traitement  : {t1 - t0:.2f} secondes")
    print(f"Mots totaux          : {total_words:,}")
    print(f"Mots distincts       : {distinct_words:,}")
    print(f"\nTop {args.top} mots les plus fréquents :\n")
    print(f"{'Rang':<6} {'Mot':<25} {'Occurrences':>12}")
    print("-" * 45)
    for rank, (word, count) in enumerate(top_n, start=1):
        print(f"{rank:<6} {word:<25} {count:>12,}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    out_path = "word_counts_result"
    # saveAsTextFile écrit en parallèle (un fichier par partition)
    word_counts.sortBy(lambda x: -x[1]).map(
        lambda x: f"{x[0]},{x[1]}"
    ).saveAsTextFile(out_path)
    print(f"\nRésultats complets sauvegardés dans : {out_path}/")

    sc.stop()


if __name__ == "__main__":
    main()

"""
TP0 - Word Count MapReduce avec PySpark
Dataset : Corpus Gutenberg (>1 Go)

Usage:
    python word_count_spark.py <chemin_dossier_ou_fichier> [--top N]
    python word_count_spark.py gutenberg_data/ --top 30
"""

import os
import re
import glob
import argparse
import time

from pyspark import SparkContext, SparkConf


def read_file(filepath: str) -> list[str]:
    """Lit un fichier texte et retourne ses lignes. Exécuté par chaque worker Spark."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def main():
    parser = argparse.ArgumentParser(description="Word Count MapReduce — PySpark")
    parser.add_argument("path", help="Dossier ou fichier texte à analyser")
    parser.add_argument("--top", type=int, default=20, help="Nombre de mots à afficher (défaut : 20)")
    args = parser.parse_args()

    # ── Collecte des fichiers ─────────────────────────────────────────────────
    if args.path.endswith(".txt"):
        files = [os.path.abspath(args.path)]
    else:
        files = glob.glob(os.path.join(args.path, "**", "*.txt"), recursive=True)

    if not files:
        print(f"[ERREUR] Aucun fichier .txt trouvé dans : {args.path}")
        return

    # ── Initialisation Spark ──────────────────────────────────────────────────
    conf = (
        SparkConf()
        .setAppName("TP0_WordCount_Gutenberg")
        .setMaster("local[*]")
        .set("spark.driver.memory", "2g")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    print(f"\n{'='*55}")
    print(f"  TP0 - Word Count MapReduce (PySpark)")
    java_ver = os.popen("java -version 2>&1").read().split()[2].strip('"')
    print(f"  {len(files)} fichier(s) | Java {java_ver}")
    print(f"{'='*55}\n")

    t0 = time.time()

    # ── Lecture parallèle des fichiers ────────────────────────────────────────
    # On parallélise la liste des chemins, puis chaque worker lit son fichier.
    # Cela évite d'utiliser le filesystem Hadoop (incompatible Java 21+).
    file_rdd = sc.parallelize(files, numSlices=len(files))
    lines = file_rdd.flatMap(read_file)

    # ── PHASE MAP ─────────────────────────────────────────────────────────────
    # Pour chaque ligne, on émet une paire (mot, 1) par mot trouvé.
    # "le chat le" → [("le", 1), ("chat", 1), ("le", 1)]
    word_pairs = lines.flatMap(
        lambda line: [
            (word, 1)
            for word in re.findall(r"[a-zA-ZÀ-ÿ]+", line.lower())
        ]
    )

    # ── PHASE SHUFFLE / SORT + REDUCE ─────────────────────────────────────────
    # reduceByKey regroupe par clé (le mot) et additionne les occurrences.
    # Spark pré-agrège localement avant le transfert réseau (= combiner).
    word_counts = word_pairs.reduceByKey(lambda a, b: a + b)

    # ── TRI et collecte du top-N ──────────────────────────────────────────────
    top_n = word_counts.takeOrdered(args.top, key=lambda x: -x[1])

    t1 = time.time()

    # ── Stats et affichage ────────────────────────────────────────────────────
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
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.path)) if os.path.isfile(args.path) else os.path.abspath(args.path), "word_counts.csv")
    all_counts = word_counts.sortBy(lambda x: -x[1]).collect()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("mot,occurrences\n")
        for word, count in all_counts:
            f.write(f"{word},{count}\n")
    print(f"\nRésultats complets sauvegardés dans : {out_path}")

    sc.stop()


if __name__ == "__main__":
    main()

"""
TP0 - Téléchargement du corpus Gutenberg
Télécharge les 100 livres les plus populaires du Projet Gutenberg (~500Mo-1Go)

Usage: python download_gutenberg.py [--books N]
"""

import os
import time
import argparse
import urllib.request

# 100 IDs des livres les plus populaires de Gutenberg
POPULAR_BOOK_IDS = [
    1342, 11, 1661, 84, 98, 2701, 174, 1400, 76, 74,
    1952, 345, 5200, 43, 2542, 2591, 16328, 1080, 46, 219,
    2554, 2600, 25344, 1260, 1184, 36, 4300, 100, 1232, 2197,
    120, 844, 514, 244, 768, 30254, 1251, 2413, 55, 996,
    1322, 45, 829, 8800, 203, 730, 1727, 4517, 4363, 2148,
    158, 2814, 1998, 779, 135, 1399, 1934, 3207, 2500, 3296,
    2680, 209, 2097, 160, 2852, 1257, 5740, 3825, 147, 521,
    2148, 863, 1155, 1217, 5765, 7370, 2489, 1064, 19942, 2600,
    3421, 1268, 35, 1250, 28054, 2160, 3207, 8164, 2147, 3825,
    1497, 2701, 766, 4363, 1513, 3207, 135, 2554, 1661, 84,
]

BASE_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
ALT_URL  = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def download_book(book_id: int, out_dir: str) -> bool:
    out_path = os.path.join(out_dir, f"book_{book_id}.txt")
    if os.path.exists(out_path):
        return True

    for url_template in [BASE_URL, ALT_URL]:
        url = url_template.format(id=book_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8", errors="ignore")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  Livre {book_id:>6} ({size_kb:>6.0f} Ko)")
            return True
        except Exception:
            continue
    print(f"  Livre {book_id} — non disponible")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", type=int, default=100, help="Nombre de livres à télécharger")
    parser.add_argument("--out", default="gutenberg_data", help="Dossier de sortie")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids = list(dict.fromkeys(POPULAR_BOOK_IDS))[:args.books]

    print(f"Téléchargement de {len(ids)} livres Gutenberg dans '{args.out}/'...")
    ok = 0
    for book_id in ids:
        if download_book(book_id, args.out):
            ok += 1
        time.sleep(0.5) 
        
    total_mb = sum(
        os.path.getsize(os.path.join(args.out, f)) / (1024 * 1024)
        for f in os.listdir(args.out)
    )
    print(f"\n{ok}/{len(ids)} livres téléchargés — {total_mb:.1f} Mo au total")
    print(f"\nLance ensuite :")
    print(f"  python word_count_spark.py {args.out}/ --top 30")


if __name__ == "__main__":
    main()

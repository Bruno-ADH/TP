# TP1 — Système de recommandation (Filtrage Collaboratif Item-Item)

**Objectif :** Implémenter un système de recommandation de films basé sur le filtrage collaboratif item-item et l'exposer via une interface Streamlit.

**Dataset :** MovieLens Small (100 000 notes, 610 utilisateurs, 9 000 films) — téléchargé automatiquement au premier lancement.

## Prérequis

```bash
pip install -r requirements.txt
```

## Lancer le TP

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur. Sélectionner un utilisateur et régler les paramètres dans la barre latérale.

## Fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit |
| `collaborative_filtering.py` | Algorithme item-item (similarité cosinus, top-N) |
| `requirements.txt` | Dépendances Python |

## Algorithme

1. Construction de la matrice utilisateurs × films
2. Calcul de la similarité cosinus entre items
3. Prédiction de note par moyenne pondérée des k voisins
4. Retour des N films avec le score prédit le plus élevé

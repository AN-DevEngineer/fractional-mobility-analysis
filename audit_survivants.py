#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_survivants.py

Analyse les utilisateurs après prétraitement.
- Génère les graphiques uniquement pour les "survivants" (ceux qui ont un CSV avec des données dans sortie/).
- Diagnostique les causes d'élimination des autres.
Exclut l'utilisateur "182" (données personnelles).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pretraitement.lecture import lire_dossier_utilisateur
from pretraitement.nettoyage import nettoyer_points, supprimer_doublons
from pretraitement.tri import trier_par_temps
from pretraitement.segmentation import segmenter_points
from config import TIME_GAP_THRESHOLD, MIN_POINTS, DISTANCE_MIN


def analyser_raisons_echec(dataset_path, user_id):
    """
    Analyse un utilisateur éliminé pour comprendre pourquoi.
    NB: Heuristique basée sur les seuils globaux. Un segment peut aussi
    être éliminé par le filtre de distance relatif (V5), ce qui est
    classé ici dans "Autre (qualité de segment)".
    """
    user_path = os.path.join(dataset_path, user_id)
    trajets = lire_dossier_utilisateur(user_path)
    
    if not trajets:
        return "Fichiers vides/illisibles"
    
    max_seg_len = 0
    max_dist = 0
    n_points_total = 0
    
    for fichier, points in trajets:
        points = nettoyer_points(points)
        points = trier_par_temps(points)
        points = supprimer_doublons(points)
        if len(points) < 2: 
            continue
        
        n_points_total += len(points)
        segs = segmenter_points(points, TIME_GAP_THRESHOLD)
        
        for seg in segs:
            max_seg_len = max(max_seg_len, len(seg))
            # Calcul de distance approximatif (Haversine rapide via degrés)
            lat = np.array([p[0] for p in seg])
            lon = np.array([p[1] for p in seg])
            d = np.sum(np.sqrt((np.diff(lat)*111)**2 + (np.diff(lon)*111*np.cos(np.radians(lat[:-1])))**2))
            max_dist = max(max_dist, d)
            
    if n_points_total < MIN_POINTS:
        return f"Points insuffisants (total={n_points_total} < {MIN_POINTS})"
    elif max_seg_len < MIN_POINTS:
        return f"Segments trop courts (max={max_seg_len} pts < {MIN_POINTS})"
    elif max_dist < DISTANCE_MIN:
        return f"Distance trop faible (max={max_dist:.1f} km)"
    else:
        return "Éliminé par les filtres qualité (distance relative ou autres)"


def main(dataset_path, sortie_pretraitement, sortie_dir="resultats_survivants"):
    os.makedirs(sortie_dir, exist_ok=True)
    
    # 1. Lister tous les utilisateurs (exclusion de 182)
    tous_users = sorted([
        u for u in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, u)) and u != "182"
    ])
    
    survivants = []
    resultats = []
    
    # 2. Identifier les survivants ET récupérer leurs stats en une seule passe
    for u in tous_users:
        chemin_csv = os.path.join(sortie_pretraitement, f"{u}.csv")
        
        if not os.path.exists(chemin_csv):
            continue
            
        try:
            # Lecture du CSV pour vérifier s'il contient des données
            df = pd.read_csv(chemin_csv)
            
            # Si le CSV est vide (en-têtes seuls), on l'ignore
            if len(df) == 0:
                continue
                
            survivants.append(u)
            resultats.append({
                "user_id": u,
                "n_segments": df["segment_id"].nunique(),
                "n_points": len(df),
                # On utilise le nombre de segments comme proxy du nombre de fichiers sources
                "n_fichiers": df["segment_id"].nunique() 
            })
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de lire {chemin_csv} : {e}")
            continue

    # --- Bilan des survivants ---
    print(f"\n{'='*60}")
    print(f"Utilisateurs totaux (sans 182) : {len(tous_users)}")
    print(f"Utilisateurs survivants        : {len(survivants)}")
    print(f"Utilisateurs éliminés          : {len(tous_users) - len(survivants)}")
    print(f"{'='*60}")

    # --- Analyse des éliminés ---
    elimines = [u for u in tous_users if u not in survivants]
    raisons = []
    
    if elimines:
        print("\nAnalyse des causes d'élimination (cela peut prendre un peu de temps)...")
        for u in elimines:
            raison = analyser_raisons_echec(dataset_path, u)
            raisons.append({"user_id": u, "raison": raison})
            
        df_raisons = pd.DataFrame(raisons)
        chemin_raisons = os.path.join(sortie_dir, "raisons_echec.csv")
        df_raisons.to_csv(chemin_raisons, index=False)
        
        print("\nCAUSES D'ÉLIMINATION :")
        print(df_raisons["raison"].value_counts().to_string())
        print(f"\n[OK] Détail des raisons exporté dans {chemin_raisons}")

    # --- Génération des graphiques pour les survivants ---
    if not resultats:
        print("[ERREUR] Aucun survivant à analyser. Arrêt des graphiques.")
        return

    print("\nGénération des graphiques pour les survivants...")
    df_surv = pd.DataFrame(resultats)
    df_surv = df_surv.sort_values("n_points", ascending=False).reset_index(drop=True)
    
    total_points = df_surv["n_points"].sum()
    df_surv["pct_points"] = (df_surv["n_points"] / total_points) * 100
    df_surv["cum_pct_points"] = df_surv["pct_points"].cumsum()
    
    # 3.1 Pareto
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(range(len(df_surv)), df_surv["n_points"], color="#2ecc71")
    ax1.set_ylabel("Points valides (survivants)", color="#2ecc71")
    ax1.set_xlabel("Utilisateurs (triés)")
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(df_surv)), df_surv["cum_pct_points"], color="#e74c3c", marker="o", markersize=3)
    ax2.set_ylabel("Pourcentage cumulé", color="#e74c3c")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color="black", linestyle="--", alpha=0.5)
    
    plt.title("Pareto : Participation des survivants après prétraitement", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "pareto_survivants.png"), dpi=150)
    plt.close()

    # 3.2 Barres Top 30
    top_30 = df_surv.head(30).iloc[::-1] # Inversé pour avoir le plus grand en haut
    plt.figure(figsize=(10, 10))
    plt.barh(top_30["user_id"], top_30["n_points"], color="#27ae60")
    plt.title("Top 30 des survivants (Points valides)", fontsize=14)
    plt.xlabel("Nombre de points valides")
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "barh_top30_survivants.png"), dpi=150)
    plt.close()

    # 3.3 Pie Chart
    plt.figure(figsize=(20, 20))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_surv)))
    labels = [row['user_id'] if row['pct_points'] > 0.5 else "" for _, row in df_surv.iterrows()]
    plt.pie(df_surv["n_points"], labels=labels, colors=colors, startangle=90, textprops={'fontsize': 10})
    plt.title(f"Répartition des survivants (n={len(df_surv)})", fontsize=20)
    plt.savefig(os.path.join(sortie_dir, "pie_survivants.png"), dpi=150)
    plt.close()
    
    print(f"[OK] Graphiques exportés dans {sortie_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/Geolife Trajectories 1.3/Data", help="Dossier des données brutes GeoLife")
    p.add_argument("--sortie-pretraitement", default="sortie", help="Dossier des CSV générés par 01_pretraitement.py")
    p.add_argument("--sortie", default="resultats_survivants", help="Dossier des résultats et graphiques")
    args = p.parse_args()
    main(args.dataset, args.sortie_pretraitement, args.sortie)
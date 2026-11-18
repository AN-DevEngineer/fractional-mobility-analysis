#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_grille.py

Explore la grille (TIME_GAP_THRESHOLD x MIN_POINTS) sur le dataset GeoLife.
Exclut l'utilisateur "182".
Produit un CSV et une Heatmap.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pretraitement.lecture import lire_dossier_utilisateur
from pretraitement.nettoyage import nettoyer_points, supprimer_doublons
from pretraitement.tri import trier_par_temps

# Grilles de test
GAPS = [300, 450, 600, 750, 900, 1050, 1200, 1500, 1800]
MINS = [100, 150, 200, 250, 300, 350, 400, 500]

def main(dataset_path, sortie_dir="resultats_grille"):
    os.makedirs(sortie_dir, exist_ok=True)
    
    users = sorted(os.listdir(dataset_path))
    # Exclusion de l'utilisateur 182
    users = [u for u in users if u != "182"]
    
    print(f"Analyse de {len(users)} utilisateurs (182 exclu)...")
    
    # Dictionnaires pour accumuler les résultats
    # Clé : (gap, min_pts) -> {"points_conserves": 0, "n_seg": 0, "n_seg_valides": 0}
    stats = {(g, m): {"points_conserves": 0, "n_seg": 0, "n_seg_valides": 0} for g in GAPS for m in MINS}
    total_points_valides = 0
    
    for i, user in enumerate(users):
        user_path = os.path.join(dataset_path, user)
        if not os.path.isdir(user_path): continue
        
        trajets = lire_dossier_utilisateur(user_path)
        for fichier, points in trajets:
            points = nettoyer_points(points)
            points = trier_par_temps(points)
            points = supprimer_doublons(points)
            
            if len(points) < 2: continue
            
            t = np.array([p[3] for p in points], dtype=float)
            dt = np.diff(t)
            total_points_valides += len(points)
            
            # Pour chaque gap, on segmente une seule fois
            for g in GAPS:
                coupures = np.where(dt > g)[0]
                debuts = np.concatenate([[0], coupures + 1])
                fins = np.concatenate([coupures, [len(points) - 1]])
                
                tailles_segments = fins - debuts + 1
                
                # Pour chaque taille min, on vérifie quels segments passent
                for m in MINS:
                    seg_valides = tailles_segments[tailles_segments >= m]
                    stats[(g, m)]["n_seg"] += len(tailles_segments)
                    stats[(g, m)]["n_seg_valides"] += len(seg_valides)
                    stats[(g, m)]["points_conserves"] += seg_valides.sum()

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(users)} utilisateurs traités...")

    # --- Construction du tableau de résultats ---
    resultats = []
    for g in GAPS:
        for m in MINS:
            s = stats[(g, m)]
            pct_points = (s["points_conserves"] / total_points_valides * 100) if total_points_valides > 0 else 0
            resultats.append({
                "gap_s": g,
                "min_points": m,
                "n_segments_valides": s["n_seg_valides"],
                "pct_points_conserves": round(pct_points, 2)
            })
    
    df = pd.DataFrame(resultats)
    chemin_csv = os.path.join(sortie_dir, "grille_gap_minpoints.csv")
    df.to_csv(chemin_csv, index=False)
    print(f"\n[OK] Résultats exportés dans {chemin_csv}")

    # --- Création du Heatmap ---
    # On veut une matrice (gap en X, min_points en Y, pct_points en couleur)
    pivot = df.pivot(index="min_points", columns="gap_s", values="pct_points_conserves")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '% de points conservés'})
    plt.title("Pourcentage de points conservés selon (Gap, Min_Points)\n(GeoLife, sans user 182)", fontsize=14)
    plt.xlabel("TIME_GAP_THRESHOLD (secondes)", fontsize=12)
    plt.ylabel("MIN_POINTS (taille minimale des segments)", fontsize=12)
    plt.gca().invert_yaxis() # Pour avoir les min_points croissants vers le bas
    
    chemin_png = os.path.join(sortie_dir, "grille_gap_minpoints.png")
    plt.tight_layout()
    plt.savefig(chemin_png, dpi=150)
    print(f"[OK] Heatmap exportée dans {chemin_png}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/Geolife Trajectories 1.3/Data", help="Chemin vers le dossier Data de GeoLife")
    p.add_argument("--sortie", default="resultats_grille", help="Dossier de sortie")
    args = p.parse_args()
    main(args.dataset, args.sortie)
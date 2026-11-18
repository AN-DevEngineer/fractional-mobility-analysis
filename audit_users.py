#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_users.py

Analyse la participation de chaque utilisateur au dataset GeoLife.
Produit un CSV et plusieurs graphiques (Pareto, Scatter, Pie, Barres).
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

# Liste des utilisateurs à exclure (votre compte)
USER_A_EXCLURE = ["182", "183"] 

def main(dataset_path, sortie_dir="resultats_users"):
    os.makedirs(sortie_dir, exist_ok=True)
    
    users = sorted(os.listdir(dataset_path))
    users = [u for u in users if u not in USER_A_EXCLURE and os.path.isdir(os.path.join(dataset_path, u))]
    
    print(f"Analyse de {len(users)} utilisateurs...")
    
    resultats = []
    
    for i, user in enumerate(users):
        user_path = os.path.join(dataset_path, user)
        trajets = lire_dossier_utilisateur(user_path)
        
        n_fichiers = len(trajets)
        n_points_bruts = 0
        n_points_valides = 0
        
        for fichier, points in trajets:
            n_points_bruts += len(points)
            points = nettoyer_points(points)
            points = trier_par_temps(points)
            points = supprimer_doublons(points)
            n_points_valides += len(points)
            
        resultats.append({
            "user_id": user,
            "n_fichiers": n_fichiers,
            "n_points_bruts": n_points_bruts,
            "n_points_valides": n_points_valides,
            "n_points_perdus": n_points_bruts - n_points_valides
        })
        
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(users)} utilisateurs traités...")
            
    df = pd.DataFrame(resultats)
    df = df.sort_values("n_points_valides", ascending=False).reset_index(drop=True)
    
    total_points = df["n_points_valides"].sum()
    total_fichiers = df["n_fichiers"].sum()
    
    df["pct_points"] = (df["n_points_valides"] / total_points) * 100
    df["cum_pct_points"] = df["pct_points"].cumsum()
    
    chemin_csv = os.path.join(sortie_dir, "participation_users.csv")
    df.to_csv(chemin_csv, index=False)
    
    print("\n" + "="*60)
    print("RÉSUMÉ DU DATASET")
    print("="*60)
    print(f"Total points valides : {total_points:,}")
    print(f"Total fichiers       : {total_fichiers:,}")
    print(f"Top 5 utilisateurs   : {df['pct_points'].head(5).sum():.1f}% des points")
    print(f"Top 10 utilisateurs  : {df['pct_points'].head(10).sum():.1f}% des points")
    print(f"Top 20 utilisateurs  : {df['pct_points'].head(20).sum():.1f}% des points")
    print(f"Nb d'users pour faire 80% : {(df['cum_pct_points'] <= 80).sum() + 1} / {len(df)}")
    print(f"[OK] CSV exporté dans {chemin_csv}")

    # ==========================================
    # 1. GRAPHIQUE PARETO (Barres + Ligne)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(range(len(df)), df["n_points_valides"], color="#4a90d9", width=0.8)
    ax1.set_ylabel("Nombre de points valides", color="#4a90d9", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#4a90d9")
    ax1.set_xlabel("Utilisateurs (triés)", fontsize=12)
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), df["cum_pct_points"], color="#e74c3c", marker="o", markersize=3, linestyle="-", linewidth=2)
    ax2.set_ylabel("Pourcentage cumulé (%)", color="#e74c3c", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color="black", linestyle="--", alpha=0.5)
    
    plt.title("Diagramme de Pareto : Participation des utilisateurs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "1_pareto_participation.png"), dpi=150)
    plt.close()

    # ==========================================
    # 2. GRAPHIQUE SCATTER (Fichiers vs Points)
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.scatter(df["n_fichiers"], df["n_points_valides"], alpha=0.6, color="#8e44ad", edgecolors="black")
    plt.xlabel("Nombre de fichiers (.plt)", fontsize=12)
    plt.ylabel("Nombre de points valides", fontsize=12)
    plt.title("Relation entre nombre de fichiers et nombre de points", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    for i in range(min(5, len(df))):
        plt.annotate(df.loc[i, "user_id"], (df.loc[i, "n_fichiers"], df.loc[i, "n_points_valides"]), 
                     textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "2_scatter_fichiers_points.png"), dpi=150)
    plt.close()

    # ==========================================
    # 3. GRAPHIQUE PIE CHART (Tous les users)
    # ==========================================
    plt.figure(figsize=(20, 20)) 
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(df)))
    labels = [row['user_id'] if row['pct_points'] > 0.5 else "" for _, row in df.iterrows()]
    def my_autopct(pct): return f'{pct:.1f}%' if pct > 0.5 else ''
        
    plt.pie(df["n_points_valides"], labels=labels, autopct=my_autopct, colors=colors, startangle=90, textprops={'fontsize': 10})
    plt.title(f"Répartition de TOUS les utilisateurs (n={len(df)})\n(Labels < 0.5% masqués)", fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "3_pie_tous_users.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 4. BARRES HORIZONTALES (Top 30)
    # ==========================================
    top_30 = df.head(30).iloc[::-1] # Inverser pour avoir le plus grand en haut
    plt.figure(figsize=(10, 10))
    plt.barh(top_30["user_id"], top_30["n_points_valides"], color="#2980b9")
    plt.xlabel("Nombre de points valides", fontsize=12)
    plt.ylabel("Utilisateur", fontsize=12)
    plt.title("Top 30 des utilisateurs par volume de points", fontsize=14, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "4_barh_top30.png"), dpi=150)
    plt.close()

    # ==========================================
    # 5. BARRES EMPILÉES (Valides vs Perdus)
    # ==========================================
    # On prend le Top 40 pour garder une lisibilité correcte
    top_40 = df.head(40)
    plt.figure(figsize=(15, 6))
    plt.bar(top_40["user_id"], top_40["n_points_valides"], label="Points valides", color="#2ecc71")
    plt.bar(top_40["user_id"], top_40["n_points_perdus"], bottom=top_40["n_points_valides"], label="Points perdus (nettoyage)", color="#e74c3c")
    
    plt.xlabel("Utilisateur (Top 40)", fontsize=12)
    plt.ylabel("Nombre de points", fontsize=12)
    plt.title("Impact du nettoyage : Points valides vs Points perdus", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "5_barres_empilees_nettoyage.png"), dpi=150)
    plt.close()

    # ==========================================
    # 6. BARRES VERTICALES (Tous les utilisateurs)
    # ==========================================
    plt.figure(figsize=(18, 6))
    plt.bar(range(len(df)), df["n_points_valides"], color="#f39c12", width=1.0)
    plt.xlabel("Utilisateurs (triés par volume)", fontsize=12)
    plt.ylabel("Nombre de points valides", fontsize=12)
    plt.title(f"Volume de points pour TOUS les utilisateurs (n={len(df)})", fontsize=14, fontweight="bold")
    plt.xlim(-1, len(df))
    plt.tight_layout()
    plt.savefig(os.path.join(sortie_dir, "6_barres_tous_users.png"), dpi=150)
    plt.close()
    
    print(f"[OK] Les 6 graphiques ont été exportés dans {sortie_dir}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/Geolife Trajectories 1.3/Data")
    p.add_argument("--sortie", default="resultats_users")
    args = p.parse_args()
    main(args.dataset, args.sortie)
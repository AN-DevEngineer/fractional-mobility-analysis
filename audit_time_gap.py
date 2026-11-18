#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from pretraitement.lecture import lire_dossier_utilisateur
from pretraitement.nettoyage import nettoyer_points, supprimer_doublons
from pretraitement.tri import trier_par_temps

SEUILS_TESTES = [300, 600, 900, 1200, 1800]
N_DFA_STABLE = 300

def lire_fichier_tdrive(filepath):
    points = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne: continue
                parts = ligne.split(",")
                if len(parts) < 4: continue
                try:
                    dt_str = parts[1].strip().replace("  ", " ").replace("- ", "-").replace(" -", "-")
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    t = float(dt.timestamp())
                    lon, lat = float(parts[2]), float(parts[3])
                    if lat == 0.0 and lon == 0.0: continue
                    points.append((lat, lon, 0.0, t))
                except Exception:
                    continue
    except OSError:
        pass
    return points

def traiter_points(points, tous_les_dt, stats_seuils):
    """Fonction utilitaire pour traiter une liste de points (commune aux 2 formats)."""
    points = nettoyer_points(points)
    points = trier_par_temps(points)
    points = supprimer_doublons(points)
    
    if len(points) < 2: return
    
    t = np.array([p[3] for p in points], dtype=float)
    dt = np.diff(t)
    tous_les_dt.append(dt)
    
    for s in SEUILS_TESTES:
        coupures = np.where(dt > s)[0]
        debuts = np.concatenate([[0], coupures + 1])
        fins = np.concatenate([coupures, [len(points) - 1]])
        
        for debut, fin in zip(debuts, fins):
            taille = fin - debut + 1
            if taille > 0:
                stats_seuils[s]["n_segments"] += 1
                stats_seuils[s]["tailles"].append(taille)
                stats_seuils[s]["n_points_total"] += taille
                if taille >= N_DFA_STABLE:
                    stats_seuils[s]["n_seg_stables"] += 1
                    stats_seuils[s]["n_points_stables"] += taille

def main(dataset_path, format_donnees, sortie_dir):
    os.makedirs(sortie_dir, exist_ok=True)
    print(f"Analyse du dataset {format_donnees}...")
    
    tous_les_dt = []
    stats_seuils = {s: {"n_segments": 0, "n_seg_stables": 0, "n_points_stables": 0, "n_points_total": 0, "tailles": []} for s in SEUILS_TESTES}
    
    if format_donnees == "geolife":
        users = sorted(os.listdir(dataset_path))
        for i, user in enumerate(users):
            user_path = os.path.join(dataset_path, user)
            if not os.path.isdir(user_path): continue
            for fichier, points in lire_dossier_utilisateur(user_path):
                traiter_points(points, tous_les_dt, stats_seuils)
            if (i+1) % 20 == 0: print(f"  {i+1}/{len(users)} utilisateurs...")
                
    elif format_donnees == "tdrive":
        fichiers = sorted([f for f in os.listdir(dataset_path) if f.endswith(".txt")])
        for i, f in enumerate(fichiers):
            points = lire_fichier_tdrive(os.path.join(dataset_path, f))
            traiter_points(points, tous_les_dt, stats_seuils)
            if (i+1) % 1000 == 0: print(f"  {i+1}/{len(fichiers)} fichiers...")

    # --- RÉSULTATS ---
    dt_global = np.concatenate(tous_les_dt)
    dt_global = dt_global[dt_global > 0]
    
    resultats = []
    for s in SEUILS_TESTES:
        tailles = stats_seuils[s]["tailles"]
        n_total = stats_seuils[s]["n_segments"]
        n_stable = stats_seuils[s]["n_seg_stables"]
        pts_total = stats_seuils[s]["n_points_total"]
        pts_stable = stats_seuils[s]["n_points_stables"]
        
        resultats.append({
            "seuil_s": s, "n_segments": n_total, "n_segments_ge_200": n_stable,
            "pct_segments_ge_200": (n_stable / n_total * 100) if n_total > 0 else 0,
            "taille_mediane": np.median(tailles) if tailles else 0,
            "pct_points_dans_seg_ge_200": (pts_stable / pts_total * 100) if pts_total > 0 else 0
        })
    
    df_res = pd.DataFrame(resultats)
    chemin_csv = os.path.join(sortie_dir, "resultats_time_gap.csv")
    df_res.to_csv(chemin_csv, index=False)
    
    print("\n" + "="*80)
    print(f"RÉSULTATS PAR SEUIL ({format_donnees})")
    print("="*80)
    print(df_res.to_string(index=False))

    # --- GRAPHIQUE ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"Audit du TIME_GAP_THRESHOLD ({format_donnees})", fontsize=14, fontweight="bold")

    bins = np.logspace(0, np.log10(dt_global.max()), 50)
    ax1.hist(dt_global, bins=bins, color="#4a90d9", edgecolor="black", alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Δt (secondes, échelle log)")
    ax1.set_ylabel("Occurrences (log)")
    ax1.set_title("1. Distribution des intervalles de temps bruts (Δt)")
    for s in SEUILS_TESTES:
        ax1.axvline(s, color="red", linestyle="--", alpha=0.5)
        ax1.text(s, ax1.get_ylim()[1]*0.9, f"{s}s", color="red", rotation=90, va="top", ha="right")

    ax2.plot(df_res["seuil_s"], df_res["taille_mediane"], marker="o", label="Taille médiane", color="#e67e22")
    ax2.plot(df_res["seuil_s"], df_res["pct_points_dans_seg_ge_200"], marker="s", label="% points dans N>=200", color="#27ae60")
    ax2.set_xlabel("Seuil de gap (secondes)")
    ax2.set_title("2. Effet du seuil sur la taille des segments et la stabilité DFA")
    ax2.axhline(N_DFA_STABLE, color="red", linestyle="--", alpha=0.5, label="N=200")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(sortie_dir, "audit_time_gap.png"), dpi=150)
    print(f"[OK] Résultats dans {sortie_dir}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["geolife", "tdrive"], required=True)
    p.add_argument("--dataset", required=True, help="Chemin du dossier contenant les données brutes")
    p.add_argument("--sortie", default="resultats_gap", help="Dossier de sortie")
    args = p.parse_args()
    main(args.dataset, args.format, args.sortie)
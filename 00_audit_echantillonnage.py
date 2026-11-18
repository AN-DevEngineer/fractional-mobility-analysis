#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_audit_echantillonnage.py

Audit de l'echantillonnage de GeoLife : a lancer AVANT de figer les seuils de config.py
(MIN_POINTS, TIME_GAP_THRESHOLD...). Pour chaque fichier .plt :
  - pas de temps median et irregularite (cv_dt)
  - mode d'enregistrement : "temps" (1-5 s), "distance" (5-10 m) ou "mixte"
  - part d'intervalles longs (> 15 s, > 60 s, > 600 s)

Meme chaine de lecture que 01_pretraitement.py (lecture -> nettoyage -> tri -> doublons), sans
segmentation ni filtre : on regarde les donnees telles qu'elles arrivent.

Usage :
    python 00_audit_echantillonnage.py
    python 00_audit_echantillonnage.py --dataset "chemin/vers/Data" --utilisateurs 000 001
Sortie : resultats_H/audit_echantillonnage.csv + resume a l'ecran.
"""

import argparse
import os

import numpy as np
import pandas as pd

from pretraitement.echantillonnage import profil_echantillonnage
from pretraitement.geodesie import haversine_vectorise
from pretraitement.lecture import lire_dossier_utilisateur
from pretraitement.nettoyage import nettoyer_points, supprimer_doublons
from pretraitement.tri import trier_par_temps

DATASET_PATH = os.path.join("dataset", "Geolife Trajectories 1.3", "Data")
SORTIE_AUDIT = os.path.join("resultats_H", "audit_echantillonnage.csv")
SEUILS_GAP_S = (15, 60, 600)


def auditer_utilisateur(user_id, user_path):
    """Une ligne par fichier .plt de l'utilisateur."""
    lignes = []
    for fichier, points in lire_dossier_utilisateur(user_path):
        points = supprimer_doublons(trier_par_temps(nettoyer_points(points)))
        if len(points) < 3:
            continue
        a = np.asarray(points, dtype=float)                 # colonnes : lat, lon, alt, timestamp
        dt = np.diff(a[:, 3])
        d = haversine_vectorise(a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1]) * 1000.0
        ligne = dict(utilisateur=user_id, fichier=os.path.basename(fichier), n_points=len(points),
                     **profil_echantillonnage(dt, d))
        for g in SEUILS_GAP_S:
            ligne[f"part_dt_sup_{g}s"] = float(np.mean(dt > g))
        lignes.append(ligne)
    return lignes


def main():
    analyseur = argparse.ArgumentParser(description="Audit de l'echantillonnage de GeoLife")
    analyseur.add_argument("--dataset", default=DATASET_PATH)
    analyseur.add_argument("--utilisateurs", nargs="*", default=None, help="restreindre (ex. 000 001)")
    analyseur.add_argument("--sortie", default=SORTIE_AUDIT)
    args = analyseur.parse_args()

    if not os.path.exists(args.dataset):
        print(f"[ERREUR] Dataset introuvable : {args.dataset}")
        return
    lignes = []
    for nom in sorted(os.listdir(args.dataset)):
        if args.utilisateurs and nom not in args.utilisateurs:
            continue
        chemin = os.path.join(args.dataset, nom)
        if os.path.isdir(chemin):
            lignes += auditer_utilisateur(nom, chemin)
            print(f"  {nom} : {sum(1 for l in lignes if l['utilisateur'] == nom)} fichier(s)", flush=True)
    df = pd.DataFrame(lignes)
    if df.empty:
        print("Aucun fichier exploitable.")
        return
    os.makedirs(os.path.dirname(args.sortie) or ".", exist_ok=True)
    df.to_csv(args.sortie, index=False)

    print(f"\n{len(df)} fichiers, {df['utilisateur'].nunique()} utilisateurs -> {args.sortie}")
    print("\nMode d'enregistrement (fichiers | points) :")
    print(df.groupby("mode").agg(fichiers=("fichier", "count"), points=("n_points", "sum")).to_string())
    print("\nPas median (s), quantiles 5/25/50/75/95 :", df["pas_median"].quantile([.05, .25, .5, .75, .95]).round(1).tolist())
    for g in SEUILS_GAP_S:
        part = np.average(df[f"part_dt_sup_{g}s"], weights=df["n_points"])
        print(f"Part d'intervalles > {g:>3d} s (ponderee par les points) : {part:.3%}")
    print("\n-> choisir TIME_GAP_THRESHOLD et MIN_POINTS dans config.py d'apres ces parts, puis les figer.")


if __name__ == "__main__":
    main()

# ./outils/chargement_segments.py
#
# Lecture des CSV de sortie/ (une ligne par point, colonnes de 01_pretraitement.py) et regroupement par segment.
# Meme format que lire_fichier_csv() de 03_estimation_H.py ; sert aux nouvelles expériences (tests de Berkes/Qu,
# partie II) pour ne pas dupliquer la lecture.

import os

import numpy as np
import pandas as pd

from hurst.estimation_segment import series_du_segment


def utilisateurs(dossier="sortie"):
    """Liste triee des identifiants d'utilisateurs ayant un CSV non vide dans `dossier`."""
    sortie = []
    for f in sorted(os.listdir(dossier)):
        if f.endswith(".csv") and os.path.getsize(os.path.join(dossier, f)) > 0:
            sortie.append(f[:-4])
    return sortie


def segments_utilisateur(user_id, dossier="sortie"):
    """
    Generateur de segments d'un utilisateur.
    Rend, pour chaque segment, un dict :
      utilisateur, segment_id, points (n x 3 : lat, lon, timestamp), d (distances en m), dt (s), v (km/h),
      suspect (bool par point : suspect_dist ou suspect_vitesse).
    Les segments de moins de 2 points sont ignores.
    """
    chemin = os.path.join(dossier, f"{user_id}.csv")
    df = pd.read_csv(chemin)
    if df.empty:
        return
    for col in ("suspect_dist", "suspect_vitesse"):
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].astype(str).str.strip().str.lower().eq("true")
    for seg_id, g in df.groupby("segment_id", sort=False):
        if len(g) < 2:
            continue
        pts = g[["lat", "lon", "timestamp"]].to_numpy(dtype=float)
        s = series_du_segment(pts)
        yield dict(utilisateur=user_id, segment_id=int(seg_id), points=pts, d=s["d"], dt=s["dt"], v=s["v"],
                   suspect=(g["suspect_dist"] | g["suspect_vitesse"]).to_numpy())


def tous_les_segments(dossier="sortie", max_utilisateurs=None):
    """Generateur sur tous les segments de tous les utilisateurs (ordre lexicographique, reproductible)."""
    for i, u in enumerate(utilisateurs(dossier)):
        if max_utilisateurs is not None and i >= max_utilisateurs:
            return
        yield from segments_utilisateur(u, dossier)

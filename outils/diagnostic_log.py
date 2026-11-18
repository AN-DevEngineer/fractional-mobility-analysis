#!/usr/bin/env python3
"""
diagnostic_log.py

Outil reutilisable pour visualiser les problemes decrits dans un fichier
log produit par le pretraitement (format "LOG TAXI xxxxx" / "LOG DU
SEGMENT x") : artefacts GPS isoles (sauts qui reviennent), depot/zones
d'immobilite, et repartition des gaps temporels.

C'est un script DETERMINISTE (parsing + calculs fixes), pas un appel a
un modele -- pensé pour pouvoir etre lance par n'importe qui (ou par
l'agent local, plus tard, comme commande deterministe et non comme du
code genere a la volee).

Usage :
    python3 diagnostic_log.py chemin/vers/log_00001.txt [--sortie fig.png]

Sortie : une figure PNG a 3 panneaux :
    1. Trajectoire (lon, lat) -- les sauts GPS isoles ressortent comme
       des lignes qui partent loin du nuage de points puis reviennent.
    2. Vitesse instantanee dans le temps (echelle log) -- pics isoles
       = artefacts, plateaux bas = immobilite/depot.
    3. Repartition des intervalles de temps (dt) entre points
       consecutifs -- separe les gaps "depot" (distance quasi nulle)
       des gaps "trou d'echantillonnage" (distance significative).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine du projet : permet `python outils/<script>.py`

import argparse
import re
import os

import numpy as np
import matplotlib.pyplot as plt


RE_POINT = re.compile(
    r"\[\s*(\d+)\]\s+(\S+) (\S+) \| lon=([\-\d.]+) lat=([\-\d.]+)"
    r"(?:\s*\|\s*dt=(\d+)s\s*\|\s*dist=([\d.]+)m\s*\|\s*v=([\d.]+)km/h)?"
)

SEUIL_VITESSE_SUSPECTE = 200.0   # km/h -- coherent avec MAX_SPEED du pipeline
SEUIL_DEPOT_DIST       = 5.0     # m -- distance quasi nulle = candidat "immobile"


def parser_log(chemin):
    """
    Parse un fichier log (format point par point avec dt/dist/v) et
    retourne la liste des points sous forme de dict.

    Entree :
        chemin (str) : chemin du fichier log

    Sortie :
        list of dict : un par point, cles "idx","date","heure","lon",
                       "lat","dt","dist","v" (None pour le premier
                       point du segment, qui n'a pas de predecesseur)
    """
    with open(chemin, "r", encoding="utf-8", errors="replace") as f:
        lignes = f.readlines()

    points = []
    for l in lignes:
        l = l.rstrip()
        if not l.startswith("["):
            continue
        m = RE_POINT.match(l)
        if not m:
            continue
        idx, date, heure, lon, lat, dt, dist, v = m.groups()
        points.append({
            "idx": int(idx), "date": date, "heure": heure,
            "lon": float(lon), "lat": float(lat),
            "dt": float(dt) if dt else np.nan,
            "dist": float(dist) if dist else np.nan,
            "v": float(v) if v else np.nan,
        })
    return points


def diagnostiquer(points):
    """
    Calcule les indicateurs de diagnostic a partir des points parses.

    Sortie : dict avec tableaux numpy (lon, lat, dt, dist, v) et les
             masques booleens "suspect" (vitesse > seuil) et "depot"
             (distance quasi nulle, point a l'arret).
    """
    lon  = np.array([p["lon"] for p in points])
    lat  = np.array([p["lat"] for p in points])
    dt   = np.array([p["dt"] for p in points])
    dist = np.array([p["dist"] for p in points])
    v    = np.array([p["v"] for p in points])

    suspect = np.nan_to_num(v, nan=0.0) > SEUIL_VITESSE_SUSPECTE
    depot   = np.nan_to_num(dist, nan=999.0) < SEUIL_DEPOT_DIST

    return {
        "lon": lon, "lat": lat, "dt": dt, "dist": dist, "v": v,
        "suspect": suspect, "depot": depot,
    }


def tracer_diagnostic(points, diag, titre, chemin_sortie):
    """
    Construit la figure a 3 panneaux et l'enregistre en PNG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(titre, fontsize=12, fontweight="bold")

    lon, lat, v, dt = diag["lon"], diag["lat"], diag["v"], diag["dt"]
    suspect, depot = diag["suspect"], diag["depot"]

    # --- Panneau 1 : trajectoire lon/lat ---
    ax = axes[0]
    ax.plot(lon, lat, "-", color="0.75", linewidth=0.6, zorder=1)
    ax.scatter(lon[~depot & ~suspect], lat[~depot & ~suspect], s=4,
               color="seagreen", label="en mouvement", zorder=2)
    ax.scatter(lon[depot], lat[depot], s=14, color="steelblue",
               label="immobile (dist<5m)", zorder=3)
    ax.scatter(lon[suspect], lat[suspect], s=70, color="crimson",
               marker="x", linewidths=2, label="suspect (v>200km/h)", zorder=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Trajectoire (lon, lat)")
    ax.legend(fontsize=7, loc="best")

    # --- Panneau 2 : vitesse dans le temps (echelle log) ---
    ax = axes[1]
    idxs = np.arange(len(v))
    v_plot = np.where(np.nan_to_num(v, nan=0.0) > 0, v, 0.01)  # eviter log(0)
    ax.plot(idxs, v_plot, "-", color="0.6", linewidth=0.5, zorder=1)
    ax.scatter(idxs[suspect], v_plot[suspect], s=40, color="crimson",
               marker="x", linewidths=2, zorder=3, label="suspect")
    ax.axhline(SEUIL_VITESSE_SUSPECTE, color="orange", linestyle="--",
               linewidth=1, label=f"seuil {SEUIL_VITESSE_SUSPECTE:.0f} km/h")
    ax.set_yscale("log")
    ax.set_xlabel("Index du point")
    ax.set_ylabel("Vitesse instantanee (km/h, echelle log)")
    ax.set_title("Vitesse dans le temps")
    ax.legend(fontsize=7, loc="best")

    # --- Panneau 3 : repartition des dt (gaps) ---
    ax = axes[2]
    dt_valide = dt[~np.isnan(dt)]
    dist_valide = diag["dist"][~np.isnan(dt)]
    est_depot_gap = dist_valide < SEUIL_DEPOT_DIST

    bins = np.logspace(0, np.log10(max(dt_valide.max(), 10)), 40)
    ax.hist(dt_valide[~est_depot_gap], bins=bins, color="seagreen",
            alpha=0.7, label="gap + déplacement réel")
    ax.hist(dt_valide[est_depot_gap], bins=bins, color="steelblue",
            alpha=0.7, label="gap + quasi immobile (dépôt ?)")
    ax.axvline(600, color="orange", linestyle="--", linewidth=1,
               label="SEUIL_GAP_NOTABLE (600s)")
    ax.set_xscale("log")
    ax.set_xlabel("Intervalle de temps dt (s, echelle log)")
    ax.set_ylabel("Nombre de points")
    ax.set_title("Répartition des intervalles de temps")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(chemin_sortie, dpi=130)
    plt.close(fig)


def resumer(points, diag):
    """Retourne un court resume texte des anomalies detectees."""
    n = len(points)
    n_suspect = int(diag["suspect"].sum())
    n_depot   = int(diag["depot"].sum())
    n_gaps    = int((diag["dt"] > 600).sum())

    lignes = [
        f"Points              : {n}",
        f"Points suspects (v>{SEUIL_VITESSE_SUSPECTE:.0f}km/h) : {n_suspect}",
        f"Points quasi immobiles (dist<{SEUIL_DEPOT_DIST:.0f}m) : {n_depot} ({100*n_depot/n:.1f}%)",
        f"Gaps notables (>600s)            : {n_gaps}",
    ]
    if n_suspect > 0:
        v = diag["v"]
        idx_max = int(np.nanargmax(v))
        lignes.append(f"Vitesse max          : {v[idx_max]:.1f} km/h (point idx={points[idx_max]['idx']})")

    return "\n".join(lignes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chemin_log", help="Chemin du fichier log a diagnostiquer")
    parser.add_argument("--sortie", default=None, help="Chemin du PNG de sortie (defaut: <nom>_diagnostic.png)")
    args = parser.parse_args()

    points = parser_log(args.chemin_log)
    if not points:
        print(f"[ERREUR] Aucun point parse dans {args.chemin_log}")
        return

    diag = diagnostiquer(points)

    nom_base = os.path.splitext(os.path.basename(args.chemin_log))[0]
    sortie = args.sortie or f"./images/{nom_base}_diagnostic.png"
    titre = f"Diagnostic visuel -- {nom_base}"

    tracer_diagnostic(points, diag, titre, sortie)

    print(resumer(points, diag))
    print(f"\nFigure enregistree : {sortie}")


if __name__ == "__main__":
    main()
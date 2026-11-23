#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
outils/exporter_pour_analyse.py

Compacte les gros dossiers du projet (sortie/ ~ 1,5 Go ; resultats_H/ ~ 320 Mo) en quelques petits fichiers
transmissibles (< ~25 Mo chacun), SANS perte utile pour les analyses :

  resume   resultats_H/<user>_H.csv  ->  export/resume_H.csv.gz      (une ligne par segment, toutes les colonnes)
           + copie des petits CSV de resultats_H/lois_de_puissance/ et resultats_H/validation/
           Suffit pour : comparaison R/S brut / corrige, audit des tableaux de la partie 3.1, filtre de fiabilite.

  series   sortie/<user>.csv  ->  export/series_000.npz, series_001.npz ...
           Pour chaque segment : la serie d_i des distances (float32, m). Suffit pour les tests de Berkes et de Qu.
           float32 : erreur relative < 1e-7 sur d_i, sans effet sur les statistiques.

Usage :
  python outils/exporter_pour_analyse.py resume --resultats resultats_H --export export
  python outils/exporter_pour_analyse.py series --dossier sortie --export export --mo-max 22
Taille attendue : resume ~ quelques Mo ; series ~ 50 a 60 Mo au total, en 3 fichiers de ~ 20 Mo.
"""

import argparse
import glob
import gzip
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def exporter_resume(resultats, export):
    import pandas as pd
    os.makedirs(export, exist_ok=True)
    fichiers = sorted(glob.glob(os.path.join(resultats, "*_H.csv")))
    if not fichiers:
        sys.exit(f"Aucun *_H.csv dans {resultats}")
    morceaux = []
    for f in fichiers:
        df = pd.read_csv(f)
        df.insert(0, "utilisateur", os.path.basename(f)[:-len("_H.csv")])
        morceaux.append(df)
    tout = pd.concat(morceaux, ignore_index=True)
    sortie = os.path.join(export, "resume_H.csv.gz")
    tout.to_csv(sortie, index=False, compression="gzip")
    print(f"{len(fichiers)} utilisateurs, {len(tout)} segments -> {sortie} ({os.path.getsize(sortie) / 1e6:.1f} Mo)")
    # petits CSV annexes (lois de puissance, validation) : copies telles quelles si < 25 Mo
    for sous in ("lois_de_puissance", "validation"):
        for f in glob.glob(os.path.join(resultats, sous, "*.csv")):
            if os.path.getsize(f) < 25e6:
                dest = os.path.join(export, sous)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(f, dest)
                print(f"copie {f} ({os.path.getsize(f) / 1e6:.2f} Mo)")
            else:
                print(f"IGNORE (> 25 Mo) : {f}")


def exporter_series(dossier, export, mo_max=22.0):
    from outils.chargement_segments import utilisateurs, segments_utilisateur
    os.makedirs(export, exist_ok=True)
    limite = int(mo_max * 1e6 / 4)                       # nb de float32 par fichier
    tampon_d, tampon_u, tampon_s, tampon_n, total = [], [], [], [], 0
    nb_fichiers, nb_seg = 0, 0

    def vider():
        nonlocal tampon_d, tampon_u, tampon_s, tampon_n, total, nb_fichiers
        if not tampon_n:
            return
        chemin = os.path.join(export, f"series_{nb_fichiers:03d}.npz")
        np.savez_compressed(chemin, d=np.concatenate(tampon_d).astype(np.float32),
                            longueurs=np.array(tampon_n, dtype=np.int64),
                            utilisateurs=np.array(tampon_u), segment_ids=np.array(tampon_s, dtype=np.int64))
        print(f"{chemin} : {len(tampon_n)} segments, {total} points, {os.path.getsize(chemin) / 1e6:.1f} Mo")
        nb_fichiers += 1
        tampon_d, tampon_u, tampon_s, tampon_n, total = [], [], [], [], 0

    for u in utilisateurs(dossier):
        for seg in segments_utilisateur(u, dossier):
            tampon_d.append(seg["d"])
            tampon_u.append(u)
            tampon_s.append(seg["segment_id"])
            tampon_n.append(len(seg["d"]))
            total += len(seg["d"])
            nb_seg += 1
            if total >= limite:
                vider()
    vider()
    print(f"Total : {nb_seg} segments en {nb_fichiers} fichier(s) -> {export}/")


def lire_series_npz(export):
    """Generateur de dicts (utilisateur, segment_id, d) depuis export/series_*.npz (meme ordre que l'export)."""
    for chemin in sorted(glob.glob(os.path.join(export, "series_*.npz"))):
        z = np.load(chemin, allow_pickle=False)
        bornes = np.r_[0, np.cumsum(z["longueurs"])]
        for i in range(len(z["longueurs"])):
            yield dict(utilisateur=str(z["utilisateurs"][i]), segment_id=int(z["segment_ids"][i]),
                       d=z["d"][bornes[i]:bornes[i + 1]].astype(float))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("resume")
    a.add_argument("--resultats", default="resultats_H")
    a.add_argument("--export", default="export")
    b = sp.add_parser("series")
    b.add_argument("--dossier", default="sortie")
    b.add_argument("--export", default="export")
    b.add_argument("--mo-max", type=float, default=22.0, help="taille maximale visee par fichier (Mo, avant compression)")
    args = p.parse_args()
    if args.cmd == "resume":
        exporter_resume(args.resultats, args.export)
    else:
        exporter_series(args.dossier, args.export, args.mo_max)


if __name__ == "__main__":
    main()

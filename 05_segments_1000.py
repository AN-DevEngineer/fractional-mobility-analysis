#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_segments_1000.py

Extrait les segments de resultats_H/*.csv avec n_points >= 1000.
AUCUN filtre sur H : on garde tous les segments de plus de 1000 points.

Sortie :
  - resultats_H_1000/<user>_H_1000.csv   (une ligne par segment)
  - resultats_H_1000/segments_1000.csv   (concatenation de tous les users)
  - resultats_H_1000/synthese_1000.csv   (une ligne par user)
  - resultats_H_1000/bilan.txt           (stats globales imprimees)

Usage :
  python 05_segments_1000.py
  python 05_segments_1000.py --seuil 2000
"""

import os
import csv
import argparse
from datetime import datetime

SOURCE_DIR = "resultats_H"
SORTIE_DIR = "resultats_H_1000"
SEUIL_N = 1000

COLONNES_SEGMENT = [
    "user_id", "segment_id", "n_points",
    "h_dfa", "h_rs", "h_whittle", "divergence",
    "h_rs_corrige", "dfa_bas", "dfa_haut", "whittle_m",
    "nul_p", "whittle_a_la_borne",
    "mode_echantillonnage", "pas_median_s", "serie",
]


def _float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v


def _int(x):
    v = _float(x)
    return None if v is None else int(round(v))


def _moy(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _moy_pond(vals, poids):
    couples = [(v, p) for v, p in zip(vals, poids) if v is not None and p]
    if not couples:
        return None
    return round(sum(v * p for v, p in couples) / sum(p for _, p in couples), 6)


def extraire(seuil_n, dossier_source=SOURCE_DIR, dossier_sortie=SORTIE_DIR):
    os.makedirs(dossier_sortie, exist_ok=True)

    fichiers = sorted(
        f for f in os.listdir(dossier_source)
        if f.endswith("_H.csv") and not f.startswith("synthese")
    )
    print(f"Fichiers source : {len(fichiers)} dans {dossier_source}/")
    print(f"Seuil n_points : >= {seuil_n}")
    print()

    lignes_toutes = []          # pour segments_1000.csv
    lignes_synthese = []        # pour synthese_1000.csv
    n_total = 0
    n_garde = 0

    for fichier in fichiers:
        user_id = fichier[:-len("_H.csv")]
        chemin = os.path.join(dossier_source, fichier)
        try:
            with open(chemin, "r", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError as e:
            print(f"[ERREUR] {chemin} : {e}")
            continue

        n_total += len(rows)
        gardes_user = []
        for row in rows:
            n = _int(row.get("n_points"))
            if n is None or n < seuil_n:
                continue
            h_dfa = _float(row.get("h_dfa"))
            h_rs = _float(row.get("h_rs"))
            h_wh = _float(row.get("h_whittle"))
            div = _float(row.get("divergence"))
            ligne = {
                "user_id": user_id,
                "segment_id": _int(row.get("segment_id")),
                "n_points": n,
                "h_dfa": h_dfa,
                "h_rs": h_rs,
                "h_whittle": h_wh,
                "divergence": div,
                "h_rs_corrige": _float(row.get("h_rs_corrige")),
                "dfa_bas": _float(row.get("dfa_bas")),
                "dfa_haut": _float(row.get("dfa_haut")),
                "whittle_m": _int(row.get("whittle_m")),
                "nul_p": _float(row.get("nul_p")),
                "whittle_a_la_borne": row.get("whittle_a_la_borne", ""),
                "mode_echantillonnage": row.get("mode_echantillonnage", ""),
                "pas_median_s": _float(row.get("pas_median_s")),
                "serie": row.get("serie", ""),
            }
            lignes_toutes.append(ligne)
            gardes_user.append(ligne)
            n_garde += 1

        # Une ligne de synthese par user (meme si 0 segment garde)
        h_dfa_u = [l["h_dfa"] for l in gardes_user]
        h_rs_u = [l["h_rs"] for l in gardes_user]
        h_wh_u = [l["h_whittle"] for l in gardes_user]
        poids_u = [l["n_points"] for l in gardes_user]

        lignes_synthese.append({
            "user_id": user_id,
            "n_segments_total": len(rows),
            "n_segments_1000": len(gardes_user),
            "h_dfa": _moy(h_dfa_u),
            "h_rs": _moy(h_rs_u),
            "h_whittle": _moy(h_wh_u),
            "h_dfa_wavg": _moy_pond(h_dfa_u, poids_u),
        })

        if gardes_user:
            print(f"  {user_id:<10} | {len(rows):>4} seg. | {len(gardes_user):>3} >= {seuil_n} pts")

    # --- Ecriture segments_1000.csv ---
    chemin_seg = os.path.join(dossier_sortie, "segments_1000.csv")
    with open(chemin_seg, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLONNES_SEGMENT)
        for l in lignes_toutes:
            w.writerow(["" if l.get(c) is None else l.get(c) for c in COLONNES_SEGMENT])
    print(f"\n[OK] {chemin_seg} ({len(lignes_toutes)} segments)")

    # --- Ecriture synthese_1000.csv ---
    chemin_syn = os.path.join(dossier_sortie, "synthese_1000.csv")
    with open(chemin_syn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "n_segments_total", "n_segments_1000",
                    "h_dfa", "h_rs", "h_whittle", "h_dfa_wavg"])
        for l in lignes_synthese:
            w.writerow([l["user_id"], l["n_segments_total"], l["n_segments_1000"],
                        l["h_dfa"], l["h_rs"], l["h_whittle"], l["h_dfa_wavg"]])
    print(f"[OK] {chemin_syn} ({len(lignes_synthese)} utilisateurs)")

    # --- Bilan imprime ---
    h_dfa_all = [l["h_dfa"] for l in lignes_toutes]
    h_rs_all = [l["h_rs"] for l in lignes_toutes]
    h_wh_all = [l["h_whittle"] for l in lignes_toutes]
    poids_all = [l["n_points"] for l in lignes_toutes]
    h_dfa_wavg = _moy_pond(h_dfa_all, poids_all)

    lignes_bilan = []
    lignes_bilan.append("=" * 60)
    lignes_bilan.append(f"BILAN  (seuil n_points >= {seuil_n})")
    lignes_bilan.append("=" * 60)
    lignes_bilan.append(f"  Segments totaux     : {n_total}")
    lignes_bilan.append(f"  Segments >= {seuil_n} pts : {n_garde}  ({100*n_garde/n_total:.1f}%)")
    lignes_bilan.append(f"  Utilisateurs avec >= 1 segment : "
                        f"{sum(1 for l in lignes_synthese if l['n_segments_1000'] > 0)}")
    lignes_bilan.append("")
    lignes_bilan.append(f"  H_DFA   (arithm.)  : {_moy(h_dfa_all)}")
    lignes_bilan.append(f"  H_DFA   (pondere)  : {h_dfa_wavg}")
    lignes_bilan.append(f"  H_RS    (arithm.)  : {_moy(h_rs_all)}")
    lignes_bilan.append(f"  H_WH    (arithm.)  : {_moy(h_wh_all)}")
    lignes_bilan.append("")
    lignes_bilan.append("  A comparer avec le corpus complet :")
    lignes_bilan.append("    H_DFA_gardes pondere = 0.743 (cf. synthese_H.csv)")
    lignes_bilan.append("=" * 60)

    for l in lignes_bilan:
        print(l)

    chemin_bilan = os.path.join(dossier_sortie, "bilan.txt")
    with open(chemin_bilan, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes_bilan) + "\n")
    print(f"[OK] {chemin_bilan}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extrait les segments >= N points")
    p.add_argument("--seuil", type=int, default=SEUIL_N,
                   help=f"seuil n_points (defaut : {SEUIL_N})")
    p.add_argument("--source", default=SOURCE_DIR)
    p.add_argument("--sortie", default=SORTIE_DIR)
    args = p.parse_args()
    debut = datetime.now()
    extraire(args.seuil, args.source, args.sortie)
    print(f"\nDuree : {(datetime.now()-debut).total_seconds():.1f} s")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparer_tdrive.py

Convertit T-Drive au format attendu par le pipeline lois de puissance :
    sortie_tdrive/NNNNN.csv avec colonnes segment_id, timestamp, lat, lon

Regles :
  - filtre les taxis avec pas median aberrant (> 700 s ou < 30 s)
  - segmentation a GAP_S = 3600 s
  - deduplique les timestamps
  - INVERSE lon/lat (T-Drive : lon puis lat ; format interne : lat puis lon)
  - AUCUN filtre MIN_POINTS (c'est un seuil DFA, inutile pour LP)

Usage :
    python3.13 preparer_tdrive.py
"""

import argparse
import csv
import glob
import os
from datetime import datetime

import numpy as np

SRC_DEF = "dataset/T-drive Taxi Trajectories/release/taxi_log_2008_by_id"
DST_DEF = "sortie_tdrive"
GAP_S = 3600
PAS_MIN_S = 30
PAS_MAX_S = 700
MIN_POINTS_FICHIER = 50


def lire_taxi(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                t = datetime.strptime(parts[1], "%Y-%m-%d %H:%M:%S").timestamp()
                lon = float(parts[2])
                lat = float(parts[3])
            except (ValueError, IndexError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180 and t > 0):
                continue
            rows.append((t, lat, lon))
    if len(rows) < MIN_POINTS_FICHIER:
        return None
    rows.sort()
    unique = [rows[0]]
    for r in rows[1:]:
        if r[0] != unique[-1][0]:
            unique.append(r)
    return unique


def ecrire_segments(rows, out_path):
    t = np.array([r[0] for r in rows])
    coupures = np.where(np.diff(t) > GAP_S)[0]
    bornes = [0] + [c + 1 for c in coupures] + [len(rows)]
    out_rows = []
    for seg_id, (a, b) in enumerate(zip(bornes[:-1], bornes[1:])):
        if b - a < 2:
            continue
        for k in range(a, b):
            t_k, lat_k, lon_k = rows[k]
            out_rows.append((seg_id, t_k, lat_k, lon_k))
    if not out_rows:
        return False
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "timestamp", "lat", "lon"])
        w.writerows(out_rows)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEF)
    ap.add_argument("--dst", default=DST_DEF)
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"[ERREUR] source introuvable : {args.src}")
        return 1
    os.makedirs(args.dst, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, "*.txt")))
    print(f"{len(files)} fichiers T-Drive dans {args.src}")

    traites, ignores, vides = 0, 0, 0
    for path in files:
        taxi = os.path.splitext(os.path.basename(path))[0]
        rows = lire_taxi(path)
        if rows is None:
            ignores += 1
            continue
        t = np.array([r[0] for r in rows])
        pas = float(np.median(np.diff(t)))
        if pas < PAS_MIN_S or pas > PAS_MAX_S:
            ignores += 1
            continue
        out_path = os.path.join(args.dst, f"{taxi.zfill(5)}.csv")
        if not ecrire_segments(rows, out_path):
            vides += 1
            continue
        traites += 1
        if traites % 500 == 0:
            print(f"  {traites} traites...")

    print(f"\nResultat : {traites} traites, {ignores} ignores, {vides} sans segment")
    print(f"Sortie : {args.dst}/")


if __name__ == "__main__":
    raise SystemExit(main())
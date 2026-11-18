#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiences/sensibilite_pretraitement.py        (hypothese H3)

Question : H depend-il des choix de pretraitement, du mode d'enregistrement et de la serie choisie ?
Si H bouge beaucoup quand on change un seuil raisonnable, alors H mesure en partie le pretraitement
(ou le logger) plutot que le deplacement : c'est ce que ce script quantifie.

Il part des CSV de ./sortie/ (segments deja marques par la Phase 3) et recalcule H sous plusieurs
VARIANTES, sans refaire le pretraitement :

    base               d_i sur le segment entier                       (comme la V4)
    coupure_artefacts  segment coupe aux points "artefact"/suspect_dist, SANS raccord
    coupure_placebo    PLACEBO : meme nombre de points retires, mais a des positions ALEATOIRES.
                       Sert a separer l'effet des artefacts de l'effet du simple raccourcissement.
    coupure_trous_60s  segment coupe aux trous de temps > 60 s
    coupure_dt_3x      segment coupe aux intervalles > 3 x le pas median
    vitesse            v_i = d_i / dt_i au lieu de d_i
    echelle_min_10     DFA avec s_min = 10 au lieu de 4

Pour une variante qui produit plusieurs morceaux, H est la moyenne ponderee par la longueur.
Sorties : resultats_H/experiences/sensibilite_pretraitement.csv (une ligne par segment et variante) et
          ..._resume.csv (effet de chaque variante par rapport a la base, et H selon le mode).

Usage :  python experiences/sensibilite_pretraitement.py --sortie ./sortie --utilisateurs 000 001
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from hurst.dfa import dfa_h
from hurst.estimation_segment import series_du_segment
from hurst.rs import rs_complet
from hurst.whittle import whittle_local
from pretraitement.echantillonnage import profil_echantillonnage
from pretraitement.segmentation import couper_aux_points_marques

MIN_POINTS_MORCEAU = 50
VARIANTES = ["base", "coupure_artefacts", "coupure_placebo", "coupure_trous_60s", "coupure_dt_3x", "vitesse", "echelle_min_10"]


def lire_sortie(chemin):
    """Lit un CSV de sortie de la V4 ; renvoie {segment_id: DataFrame} avec les colonnes de marquage."""
    df = pd.read_csv(chemin)
    if "type_suspect_vitesse" not in df.columns:
        df["type_suspect_vitesse"] = ""
    df["type_suspect_vitesse"] = df["type_suspect_vitesse"].fillna("")
    if "suspect_dist" not in df.columns:
        df["suspect_dist"] = False
    return {sid: g.reset_index(drop=True) for sid, g in df.groupby("segment_id", sort=True)}


def couper_aux_intervalles(df, seuil_s=None, facteur=None, min_points=MIN_POINTS_MORCEAU):
    """Morceaux continus, coupes la ou dt depasse seuil_s (secondes) ou facteur x pas median. Sans raccord."""
    dt = np.diff(df["timestamp"].to_numpy(float))
    limite = seuil_s if seuil_s is not None else facteur * np.median(dt)
    coupures = np.flatnonzero(dt > limite) + 1
    bornes = np.concatenate([[0], coupures, [len(df)]])
    morceaux = [df.iloc[a:b] for a, b in zip(bornes[:-1], bornes[1:])]
    return [m for m in morceaux if len(m) >= min_points]


def couper_aux_indices(df, indices, min_points=MIN_POINTS_MORCEAU):
    """Retire les points d'indices donnes et renvoie les morceaux continus restants (sans raccord)."""
    morceaux, debut = [], 0
    for i in sorted(indices):
        if i > debut:
            morceaux.append(df.iloc[debut:i])
        debut = i + 1
    if debut < len(df):
        morceaux.append(df.iloc[debut:])
    return [m for m in morceaux if len(m) >= min_points]


def couper_placebo(df, min_points=MIN_POINTS_MORCEAU):
    """
    Meme nombre de points retires que la coupure aux artefacts, mais tires AU HASARD (graine liee
    au segment : resultat reproductible). Si aucun artefact : segment inchange.
    """
    k = len(df) - sum(len(m) for m in couper_aux_points_marques(df, 1))
    if k == 0:
        return [df] if len(df) >= min_points else []
    rng = np.random.default_rng([len(df), k, int(df["timestamp"].iloc[0]) % 2**31])
    return couper_aux_indices(df, rng.choice(len(df), size=k, replace=False), min_points)


def morceaux_de_la_variante(df, variante):
    """Liste de DataFrames (morceaux) sur lesquels on calcule H pour cette variante."""
    if variante in ("base", "vitesse", "echelle_min_10"):
        return [df]
    if variante == "coupure_artefacts":
        return couper_aux_points_marques(df, MIN_POINTS_MORCEAU)
    if variante == "coupure_placebo":
        return couper_placebo(df)
    if variante == "coupure_trous_60s":
        return couper_aux_intervalles(df, seuil_s=60.0)
    if variante == "coupure_dt_3x":
        return couper_aux_intervalles(df, facteur=3.0)
    raise ValueError(f"variante inconnue : {variante!r}")


def h_du_morceau(df, variante):
    pts = df[["lat", "lon", "timestamp"]].to_numpy(float)
    s = series_du_segment(pts)
    if variante == "vitesse":
        x = s["v"][np.isfinite(s["v"])]
    else:
        x = s["d"]
    kw = {"echelle_min": 10} if variante == "echelle_min_10" else {}
    return dict(h_dfa=dfa_h(x, **kw), h_rs_corrige=rs_complet(x).h_corrige, h_whittle=whittle_local(x).h)


def h_de_la_variante(df, variante):
    """H (ponderee par la longueur des morceaux) pour DFA, R/S corrige et Whittle."""
    morceaux = morceaux_de_la_variante(df, variante)
    resultats = [(len(m), h_du_morceau(m, variante)) for m in morceaux if len(m) >= 16]
    sortie = dict(n_morceaux=len(resultats), n_points_utilises=int(sum(n for n, _ in resultats)))
    for methode in ("h_dfa", "h_rs_corrige", "h_whittle"):
        valeurs = [(n, r[methode]) for n, r in resultats if np.isfinite(r[methode])]
        sortie[methode] = float(np.average([v for _, v in valeurs], weights=[n for n, _ in valeurs])) if valeurs else np.nan
    return sortie


def analyser(dossier_sortie, utilisateurs=None, variantes=VARIANTES):
    lignes = []
    for nom in sorted(os.listdir(dossier_sortie)):
        if not (nom.endswith(".csv") and nom[:3].isdigit()):
            continue
        if utilisateurs and nom[:3] not in utilisateurs:
            continue
        for sid, df in lire_sortie(os.path.join(dossier_sortie, nom)).items():
            if len(df) < 30:
                continue
            pts = df[["lat", "lon", "timestamp"]].to_numpy(float)
            s = series_du_segment(pts)
            mode = profil_echantillonnage(s["dt"], s["d"])["mode"]
            for v in variantes:
                lignes.append(dict(utilisateur=nom[:3], segment=sid, n_points=len(df), mode=mode, variante=v,
                                   **h_de_la_variante(df, v)))
    return pd.DataFrame(lignes)


def resumer(df):
    """Effet de chaque variante par rapport a la base, sur les segments ou les deux sont calculables."""
    base = df[df["variante"] == "base"].set_index(["utilisateur", "segment"])
    lignes = []
    for v, g in df.groupby("variante"):
        g = g.set_index(["utilisateur", "segment"])
        comm = g.index.intersection(base.index)
        for m in ("h_dfa", "h_whittle"):
            a, b = g.loc[comm, m], base.loc[comm, m]
            ok = a.notna() & b.notna()
            lignes.append(dict(variante=v, methode=m, n_segments=int(ok.sum()),
                               h_mediane=float(a[ok].median()) if ok.any() else np.nan,
                               ecart_abs_median_vs_base=float((a[ok] - b[ok]).abs().median()) if ok.any() else np.nan,
                               part_ecart_sup_0_1=float(((a[ok] - b[ok]).abs() > 0.1).mean()) if ok.any() else np.nan,
                               part_dfa_sup_1=float((a[ok] > 1).mean()) if (ok.any() and m == "h_dfa") else np.nan))
    return pd.DataFrame(lignes)


def h_selon_le_mode(df):
    base = df[df["variante"] == "base"]
    return base.groupby("mode").agg(n_segments=("segment", "count"), h_dfa_mediane=("h_dfa", "median"),
                                    h_whittle_mediane=("h_whittle", "median"), n_points_mediane=("n_points", "median")).reset_index()


def main():
    analyseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analyseur.add_argument("--sortie", default="sortie")
    analyseur.add_argument("--utilisateurs", nargs="*", default=None)
    analyseur.add_argument("--resultats", default=os.path.join("resultats_H", "experiences"))
    args = analyseur.parse_args()
    df = analyser(args.sortie, args.utilisateurs)
    if df.empty:
        print("Aucun segment trouve dans", args.sortie)
        return
    os.makedirs(args.resultats, exist_ok=True)
    df.to_csv(os.path.join(args.resultats, "sensibilite_pretraitement.csv"), index=False)
    resume = resumer(df)
    resume.to_csv(os.path.join(args.resultats, "sensibilite_pretraitement_resume.csv"), index=False)
    print(f"{df.groupby(['utilisateur', 'segment']).ngroups} segments, {len(VARIANTES)} variantes\n")
    print("Effet de chaque variante sur H (par rapport a la base) :")
    print(resume.round(3).to_string(index=False))
    print("\nH de la base selon le mode d'enregistrement :")
    print(h_selon_le_mode(df).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

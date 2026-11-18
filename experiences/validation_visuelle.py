#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiences/validation_visuelle.py

Votre lecture des courbes, mesuree. Vous dites deviner H en regardant une courbe, et tomber juste dans la
plupart des cas : c'est une connaissance d'expert, et on peut la transformer en resultat.

PRINCIPE (a l'aveugle) :
  1. `preparer` tire des segments reels et fabrique des images NEUTRES : meme style que courbe.py mais
     sans titre, sans statistique, sans point rouge. Elle y MELE des LEURRES, sans vous le dire :
       - "melange" : les memes valeurs d_i dans un ordre aleatoire (aucune structure temporelle, H = 0.5)
       - "ar"      : memes valeurs, mais avec une memoire COURTE (modele AR) : H apparent ~ 0.5 a grande echelle
  2. Vous remplissez feuille_reponses.csv : votre H pour chaque image (ne regardez PAS cle_secrete.csv).
  3. `evaluer` compare vos reponses aux estimateurs et mesure si vous distinguez le reel des leurres.

CE QUE LES CHIFFRES DISENT :
  - Accord eleve avec les estimateurs SUR LES SEGMENTS REELS : votre oeil et les estimateurs voient la meme chose.
  - Mais si votre H est aussi eleve sur les leurres "ar" que sur le reel (AUC proche de 0.5), votre oeil lit
    surtout l'AMPLITUDE et les paliers, pas la dependance longue : meme limite que DFA (Maraun et al., 2004).
  - Si votre H est bas sur les leurres et haut sur le reel (AUC proche de 1), votre oeil detecte une
    vraie dependance temporelle. C'est un resultat.
Les deux issues sont utilisables dans le memoire.

Usage :
    python experiences/validation_visuelle.py preparer --sortie ./sortie --n 60 --leurres 20
    (remplir experiences/visuel/feuille_reponses.csv)
    python experiences/validation_visuelle.py evaluer
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from hurst.dfa import dfa_h
from hurst.estimation_segment import series_du_segment
from hurst.rs import rs_complet
from hurst.whittle import whittle_local
from pretraitement.courbe import tracer_courbe_serie
from validation.modeles_nuls import surrogats

DOSSIER_PAR_DEFAUT = os.path.join("experiences", "visuel")
MIN_POINTS_VISUEL = 150


def lister_segments(dossier_sortie, utilisateurs=None, min_points=MIN_POINTS_VISUEL):
    """Tous les segments assez longs : liste de (utilisateur, segment_id, DataFrame)."""
    segments = []
    for nom in sorted(os.listdir(dossier_sortie)):
        if not (nom.endswith(".csv") and nom[:3].isdigit()):
            continue
        if utilisateurs and nom[:3] not in utilisateurs:
            continue
        df = pd.read_csv(os.path.join(dossier_sortie, nom))
        for sid, g in df.groupby("segment_id", sort=True):
            if len(g) >= min_points:
                segments.append((nom[:3], int(sid), g.reset_index(drop=True)))
    return segments


def _estimations(x):
    return dict(h_dfa=dfa_h(x), h_rs_corrige=rs_complet(x).h_corrige, h_whittle=whittle_local(x).h)


def preparer(dossier_sortie, dossier=DOSSIER_PAR_DEFAUT, n_reels=60, n_leurres=20, graine=7, utilisateurs=None):
    """Fabrique images neutres, cle secrete et feuille de reponses. Renvoie la cle (DataFrame)."""
    rng = np.random.default_rng(graine)
    candidats = lister_segments(dossier_sortie, utilisateurs)
    if len(candidats) < n_reels:
        raise ValueError(f"seulement {len(candidats)} segments d'au moins {MIN_POINTS_VISUEL} points pour {n_reels} demandes")
    choisis = [candidats[i] for i in rng.choice(len(candidats), size=n_reels, replace=False)]

    elements = []   # (type, utilisateur, segment, t_rel, serie)
    series = {}
    for user, sid, df in choisis:
        s = series_du_segment(df[["lat", "lon", "timestamp"]].to_numpy(float))
        series[(user, sid)] = (s["t"][1:] - s["t"][0], s["d"])
        elements.append(("reel", user, sid, *series[(user, sid)]))
    cles = list(series)
    for k in range(n_leurres):
        user, sid = cles[int(rng.integers(len(cles)))]
        t_rel, d = series[(user, sid)]
        type_leurre = "melange" if k % 2 == 0 else "ar"
        faux = rng.permutation(d) if type_leurre == "melange" else surrogats(d, rng, modele="ar", nb=1)[0]
        elements.append((type_leurre, user, sid, t_rel, faux))

    ordre = rng.permutation(len(elements))
    os.makedirs(os.path.join(dossier, "images"), exist_ok=True)
    lignes = []
    for rang, i in enumerate(ordre, start=1):
        type_, user, sid, t_rel, x = elements[i]
        code = f"img_{rang:03d}"
        tracer_courbe_serie(t_rel, x, os.path.join(dossier, "images", f"{code}.png"))
        lignes.append(dict(code=code, type=type_, utilisateur=user, segment=sid, n_points=len(x), **_estimations(x)))
    cle = pd.DataFrame(lignes)
    cle.to_csv(os.path.join(dossier, "cle_secrete.csv"), index=False)
    pd.DataFrame({"code": cle["code"], "h_visuel": "", "commentaire": ""}).to_csv(
        os.path.join(dossier, "feuille_reponses.csv"), index=False)
    with open(os.path.join(dossier, "LISEZ_MOI.txt"), "w", encoding="utf-8") as f:
        f.write(f"""VALIDATION VISUELLE A L'AVEUGLE -- {len(cle)} images dans images/

1. Ouvrez les images dans l'ordre (img_001, img_002, ...).
2. Pour chacune, ecrivez dans feuille_reponses.csv (colonne h_visuel) le H que vous predisez.
   Meme echelle que les estimateurs (0.5 = pas de memoire, 1 = tres persistant ; vous pouvez depasser 1).
3. NE REGARDEZ PAS cle_secrete.csv avant d'avoir fini : elle contient les H estimes et la nature de chaque image.
4. Quand c'est fini :  python experiences/validation_visuelle.py evaluer
Certaines images sont des leurres : ne cherchez pas a les deviner, notez seulement le H que vous lisez.
""")
    return cle


def evaluer(dossier=DOSSIER_PAR_DEFAUT):
    """Compare feuille_reponses.csv a la cle ; renvoie un DataFrame de resume."""
    cle = pd.read_csv(os.path.join(dossier, "cle_secrete.csv"))
    rep = pd.read_csv(os.path.join(dossier, "feuille_reponses.csv"))
    rep["h_visuel"] = pd.to_numeric(rep["h_visuel"], errors="coerce")
    df = cle.merge(rep[["code", "h_visuel"]], on="code").dropna(subset=["h_visuel"])
    if len(df) < 5:
        raise ValueError(f"seulement {len(df)} reponses exploitables : remplissez feuille_reponses.csv")
    lignes = []
    for type_, g in df.groupby("type"):
        lignes.append(dict(rubrique=f"h_visuel moyen ({type_})", valeur=float(g["h_visuel"].mean()), n=len(g)))
    reels = df[df["type"] == "reel"]
    for m in ("h_dfa", "h_rs_corrige", "h_whittle"):
        ok = reels[[m, "h_visuel"]].dropna()
        if len(ok) >= 5:
            lignes.append(dict(rubrique=f"Spearman visuel vs {m} (reel)", valeur=float(spearmanr(ok["h_visuel"], ok[m])[0]), n=len(ok)))
            lignes.append(dict(rubrique=f"erreur absolue moyenne vs {m} (reel)", valeur=float((ok["h_visuel"] - ok[m]).abs().mean()), n=len(ok)))
            lignes.append(dict(rubrique=f"part |ecart| <= 0.1 vs {m} (reel)", valeur=float(((ok["h_visuel"] - ok[m]).abs() <= 0.1).mean()), n=len(ok)))
    leurres = df[df["type"] != "reel"]
    if len(reels) >= 3 and len(leurres) >= 3:
        u = mannwhitneyu(reels["h_visuel"], leurres["h_visuel"], alternative="two-sided")
        lignes.append(dict(rubrique="AUC reel vs leurres (0.5 = indiscernable)", valeur=float(u.statistic / (len(reels) * len(leurres))), n=len(df)))
    resume = pd.DataFrame(lignes)
    resume.to_csv(os.path.join(dossier, "evaluation.csv"), index=False)
    return resume


def main():
    analyseur = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = analyseur.add_subparsers(dest="commande", required=True)
    p = sous.add_parser("preparer")
    p.add_argument("--sortie", default="sortie")
    p.add_argument("--dossier", default=DOSSIER_PAR_DEFAUT)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--leurres", type=int, default=20)
    p.add_argument("--graine", type=int, default=7)
    p.add_argument("--utilisateurs", nargs="*", default=None)
    e = sous.add_parser("evaluer")
    e.add_argument("--dossier", default=DOSSIER_PAR_DEFAUT)
    args = analyseur.parse_args()
    if args.commande == "preparer":
        cle = preparer(args.sortie, args.dossier, args.n, args.leurres, args.graine, args.utilisateurs)
        print(f"{len(cle)} images dans {os.path.join(args.dossier, 'images')}/ -- lisez {os.path.join(args.dossier, 'LISEZ_MOI.txt')}")
    else:
        print(evaluer(args.dossier).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

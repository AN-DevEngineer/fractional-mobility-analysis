#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
02_validation_methodes.py

Validation Monte-Carlo : exactitude des estimateurs (DFA, R/S, Whittle local) et specificite du
test nul a memoire courte, sur des series de verite connue. AUCUNE DONNEE REQUISE.

Usage :
    python 02_validation_methodes.py --n-rep 500 --null-rep 200

Ecrit dans resultats_H/validation/ :
    exactitude.csv          une ligne par replication
    exactitude_resume.csv   biais, ecart-type, RMSE par (H0, N, methode)
    tableau_exactitude.tex  a inclure dans le memoire par \input{}
    taux_rejet_nul.csv      faux positifs et puissance du test nul (AR(1) et AR(<=5))
    config.txt              empreinte de config.py et parametres

Les tableaux du memoire viennent d'ici, jamais de la main (c'est ce qui a produit les
incoherences de la V4 : tableau 6.1 non reproductible).
"""

import argparse
import os
import time

import numpy as np

import config
from validation.monte_carlo import resumer, monte_carlo, tableau_latex_exactitude, taux_rejet_nul


def main():
    analyseur = argparse.ArgumentParser(description="Validation Monte-Carlo des estimateurs et du test nul")
    analyseur.add_argument("--h", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    analyseur.add_argument("--n", type=int, nargs="+", default=[300, 500, 1000, 2000, 5000])
    analyseur.add_argument("--n-rep", type=int, default=200, help="replications par cellule (exactitude)")
    analyseur.add_argument("--null-rep", type=int, default=100, help="replications du test nul (0 = ne pas le faire)")
    analyseur.add_argument("--null-n", type=int, nargs="+", default=[300, 1000])
    analyseur.add_argument("--modeles", nargs="+",
                           default=["iid", "ar1_0.5", "ar1_0.8", "regimes", "onoff_1.5", "fgn_0.6", "fgn_0.7", "fgn_0.9"])
    analyseur.add_argument("--sortie", default=os.path.join("resultats_H", "validation"))
    args = analyseur.parse_args()

    os.makedirs(args.sortie, exist_ok=True)
    rng = np.random.default_rng(config.GRAINE)

    t0 = time.time()
    mc = monte_carlo(args.h, args.n, args.n_rep, rng)
    mc.to_csv(os.path.join(args.sortie, "exactitude.csv"), index=False)
    resume = resumer(mc)
    resume.to_csv(os.path.join(args.sortie, "exactitude_resume.csv"), index=False)
    legende = f"Biais et RMSE des estimateurs sur fGn ({args.n_rep} r\u00e9alisations par cellule, graine {config.GRAINE})."
    with open(os.path.join(args.sortie, "tableau_exactitude.tex"), "w", encoding="utf-8") as f:
        f.write(tableau_latex_exactitude(resume, legende, "tab:validation_synthetique"))
    print(f"exactitude : {len(mc)} series en {time.time() - t0:.0f} s  (RMSE)")
    print(resume.pivot(index=["h0", "n"], columns="methode", values="rmse").round(3).to_string())

    if args.null_rep > 0:
        t0 = time.time()
        taux = taux_rejet_nul(args.modeles, args.null_n, args.null_rep, rng)
        taux.to_csv(os.path.join(args.sortie, "taux_rejet_nul.csv"), index=False)
        print(f"\ntest nul : {len(taux)} cellules en {time.time() - t0:.0f} s  (taux de rejet a alpha = {config.NUL_ALPHA})")
        print(taux.pivot_table(index=["modele", "n"], columns="ordre_ar_max", values="taux_rejet").round(2).to_string())

    with open(os.path.join(args.sortie, "config.txt"), "w", encoding="utf-8") as f:
        f.write(f"empreinte_config = {config.empreinte_config()}\nGRAINE = {config.GRAINE}\n"
                f"n_rep = {args.n_rep}\nnull_rep = {args.null_rep}\n")


if __name__ == "__main__":
    main()

# ./validation/monte_carlo.py
#
# Validation Monte-Carlo des estimateurs et du test nul sur des series de verite connue.
#
# Deux questions distinctes :
#   1. EXACTITUDE  : sur un fGn de H connu, quel biais et quelle dispersion (par estimateur, H, N) ?
#   2. SPECIFICITE : le test de depassement du modele nul rejette-t-il a tort des series a memoire
#      courte (faux positifs) et detecte-t-il une vraie memoire longue (puissance) ?
# Les tableaux du memoire sont produits par ce module, jamais recopies a la main.

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from hurst.dfa import dfa_h
from hurst.rs import rs_complet
from hurst.whittle import whittle_local
from simulation import catalogue_modeles
from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire
from validation.modeles_nuls import evaluer_depassement_dfa

METHODES = ("dfa", "rs_brut", "rs_corrige", "whittle")
NOMS_METHODES = {"dfa": "DFA", "rs_brut": "R/S brut", "rs_corrige": "R/S corrig\u00e9", "whittle": "Whittle local"}


def estimer_tout(x):
    """H estime par les quatre variantes sur une serie. nan si non calculable."""
    r = rs_complet(x)
    return dict(dfa=dfa_h(x), rs_brut=r.h_brut, rs_corrige=r.h_corrige, whittle=whittle_local(x).h)


def monte_carlo(valeurs_h, valeurs_n, nb_rep, rng):
    """Estimations sur nb_rep fGn pour chaque couple (H0, N) ; une ligne par replication."""
    lignes = []
    for h0 in valeurs_h:
        for n in valeurs_n:
            for rep in range(nb_rep):
                lignes.append(dict(h0=h0, n=n, rep=rep, **estimer_tout(bruit_gaussien_fractionnaire(n, h0, rng))))
    return pd.DataFrame(lignes)


def resumer(df):
    """Biais, ecart-type et RMSE par (H0, N, methode)."""
    sortie = []
    for (h0, n), g in df.groupby(["h0", "n"]):
        for m in METHODES:
            e = g[m].dropna() - h0
            sortie.append(dict(h0=h0, n=n, methode=m, n_valides=len(e), biais=e.mean(),
                               ecart_type=e.std(ddof=1), rmse=np.sqrt((e ** 2).mean())))
    return pd.DataFrame(sortie)


def taux_rejet_nul(modeles, valeurs_n, nb_rep, rng, ordres_ar=(1, 5)):
    """
    Taux de rejet du test nul par modele, N et ordre AR maximal du modele nul.

    Lecture : pour un modele a memoire courte le taux devrait etre proche de alpha ; pour un fGn de
    H > 0.5 il mesure la puissance ; pour 'regimes' ou 'onoff', il montre que le test ne separe pas
    memoire longue et changements de regime (il dit seulement "non explique par un AR court").
    """
    catalogue = catalogue_modeles()
    lignes = []
    for nom in modeles:
        generer = catalogue[nom]
        for n in valeurs_n:
            for p in ordres_ar:
                rejets, hs = 0, []
                for _ in range(nb_rep):
                    t = evaluer_depassement_dfa(generer(n, rng), rng, ordre_ar_max=p)
                    rejets += bool(t["p"] <= config.NUL_ALPHA)
                    hs.append(t["h_obs"])
                lignes.append(dict(modele=nom, n=n, ordre_ar_max=p, taux_rejet=rejets / nb_rep,
                                   h_dfa_moyen=float(np.nanmean(hs)), nb_rep=nb_rep))
    return pd.DataFrame(lignes)


def tableau_latex_exactitude(resume, legende, etiquette):
    """Tableau LaTeX (biais / RMSE) genere depuis les resultats, pret a etre inclus par \\input{}."""
    lignes = [r"\begin{table}[ht]", r"\centering", r"\small",
              r"\begin{tabular}{rr" + "rr" * len(METHODES) + "}", r"\hline",
              r"$H_0$ & $N$ & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{NOMS_METHODES[m]}}}" for m in METHODES) + r" \\",
              " & & " + " & ".join("biais & RMSE" for _ in METHODES) + r" \\", r"\hline"]
    for (h0, n), g in resume.groupby(["h0", "n"]):
        cases = []
        for m in METHODES:
            r = g[g["methode"] == m].iloc[0]
            cases.append(f"{r.biais:+.3f} & {r.rmse:.3f}")
        lignes.append(f"{h0:.1f} & {n} & " + " & ".join(cases) + r" \\")
    lignes += [r"\hline", r"\end{tabular}", rf"\caption{{{legende}}}", rf"\label{{{etiquette}}}", r"\end{table}"]
    return "\n".join(lignes)

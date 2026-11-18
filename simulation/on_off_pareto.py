# ./simulation/on_off_pareto.py
#
# Processus ON/OFF a durees de Pareto : alternance mouvement / arret dont les durees ont une queue
# en loi de puissance. Pour un exposant de queue alpha dans ]1, 2[, la serie presente de la
# dependance longue avec H = (3 - alpha) / 2 (Taqqu, Willinger & Sherman, 1997 -- formule a
# confronter a la source avant publication). C'est l'hypothese H2 du memoire : la memoire longue
# apparente de d_i viendrait de l'alternance mouvement/arret aux durees a queue lourde.

import numpy as np


def durees_pareto(alpha, taille, rng, xm=1.0):
    """Durees de Pareto : P(T > t) = (xm / t)^alpha pour t >= xm."""
    return xm * (1.0 - rng.random(taille)) ** (-1.0 / alpha)


def on_off(n, alpha_on, alpha_off, rng, rodage=2000, xm=1.0):
    """
    Indicateur 0/1 d'un processus de renouvellement alterne ON/OFF aux durees de Pareto.
    Pas de temps discret : les durees sont arrondies a l'entier superieur.
    `rodage` points sont generes puis jetes pour oublier la condition initiale.
    `xm` est la duree minimale (en pas de temps) des deux etats : avec des arrets d'au moins 5 min
    echantillonnes toutes les 5 s, xm vaut 60. Il ne change pas l'exposant asymptotique, mais change
    le biais a taille finie, d'ou son importance pour comparer a des donnees reelles.
    """
    total = n + rodage
    lot = max(1000, total // 4)
    blocs, longueur, etat = [], 0, int(rng.integers(2))
    while longueur < total:
        durees = np.ceil(durees_pareto(alpha_on if etat else alpha_off, lot, rng, xm)).astype(int)
        for d in durees:
            blocs.append((etat, d))
            longueur += d
            etat ^= 1
            if longueur >= total:
                break
    etats = np.concatenate([np.full(d, s, dtype=float) for s, d in blocs])
    return etats[rodage:rodage + n]


def h_theorique_on_off(alpha_on, alpha_off):
    """H theorique d'un ON/OFF a queues lourdes ; hors ]1, 2[ la prediction ne s'applique pas (nan)."""
    a = min(alpha_on, alpha_off)       # la queue la plus lourde domine
    return (3.0 - a) / 2.0 if 1.0 < a < 2.0 else float("nan")

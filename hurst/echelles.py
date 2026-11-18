# ./hurst/echelles.py
#
# Outils communs aux versions rapides des estimateurs : grille d'echelles et
# regressions log-log (moindres carres ordinaires).

from math import exp, floor, log

import numpy as np


def grille_echelles(bas, haut, nombre):
    """
    Echelles entieres regulierement espacees en log, de `bas` a `haut`.

    Reproduit EXACTEMENT la grille de la V4 (generer_valeurs_s / generer_valeurs_m) :
    floor(exp(...)) puis suppression des doublons en conservant l'ordre. C'est ce qui
    garantit que les versions rapides retrouvent les valeurs de la V4.

    Sortie : tableau d'entiers ; vide si haut <= bas.
    """
    if haut <= bas or nombre < 2:
        return np.array([], dtype=int)
    echelles = []
    for j in range(1, nombre + 1):
        s = floor(exp(log(bas) + (j - 1) / (nombre - 1) * (log(haut) - log(bas))))
        if s not in echelles:
            echelles.append(s)
    return np.array(echelles, dtype=int)


def pente_mco(x, y):
    """Pente et ordonnee a l'origine des moindres carres ordinaires ; (nan, nan) si moins de 2 points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    den = ((x - xm) ** 2).sum()
    if den == 0:
        return float("nan"), float("nan")
    pente = ((x - xm) * (y - ym)).sum() / den
    return float(pente), float(ym - pente * xm)


def pentes_par_ligne(log_x, log_y):
    """
    Pente MCO de chaque ligne de log_y contre log_x ; les NaN de log_y sont ignores.

    log_x : (k,)   log_y : (nb_series, k)   ->   (nb_series,), NaN si moins de 2 points valides.
    Sert a traiter d'un seul coup un lot de series (surrogats du modele nul).
    """
    log_y = np.atleast_2d(log_y)
    masque = np.isfinite(log_y)
    x = np.broadcast_to(np.asarray(log_x, float), log_y.shape)
    n = masque.sum(axis=1).astype(float)
    xm = np.where(masque, x, 0.0)
    ym = np.where(masque, log_y, 0.0)
    sx, sy = xm.sum(1), ym.sum(1)
    sxy, sxx = (xm * ym).sum(1), (xm * xm).sum(1)
    den = n * sxx - sx ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        pente = (n * sxy - sx * sy) / den
    pente[(n < 2) | (den <= 0)] = np.nan
    return pente


def pentes_glissantes(log_x, log_y, fenetre):
    """
    Pentes locales sur fenetres glissantes de `fenetre` points consecutifs.
    Sortie : (centres, pentes) ; centres = moyenne des log_x de chaque fenetre. Vide si trop peu de points.
    """
    log_x = np.asarray(log_x, float)
    log_y = np.asarray(log_y, float)
    k = len(log_x)
    if fenetre < 2 or k < fenetre:
        return np.array([]), np.array([])
    centres, pentes = [], []
    for i in range(k - fenetre + 1):
        p, _ = pente_mco(log_x[i:i + fenetre], log_y[i:i + fenetre])
        centres.append(log_x[i:i + fenetre].mean())
        pentes.append(p)
    return np.array(centres), np.array(pentes)

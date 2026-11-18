# ./simulation/bruit_fractionnaire.py
#
# Series synthetiques de dependance longue et de H CONNU : sert a valider les estimateurs.
# Toutes les fonctions prennent un numpy.random.Generator explicite (aucune graine cachee).

import numpy as np


def bruit_gaussien_fractionnaire(n, h, rng):
    """
    Bruit gaussien fractionnaire (fGn) EXACT, de variance 1, par plongement circulant (Davies-Harte).
    Stationnaire. Autocovariance : gamma(k) = 0.5 * (|k+1|^2H - 2|k|^2H + |k-1|^2H).

    Entree : n (longueur), h dans ]0, 1[, rng (numpy.random.Generator)
    Sortie : tableau numpy de longueur n
    """
    if not 0 < h < 1:
        raise ValueError("H doit etre dans ]0, 1[ pour un fGn")
    k = np.arange(0, n + 1, dtype=float)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    ligne = np.concatenate([gamma, gamma[-2:0:-1]])      # premiere ligne de la matrice circulante (taille 2n)
    lam = np.fft.fft(ligne).real
    if lam.min() < -1e-8 * lam.max():
        raise ValueError(f"plongement circulant non defini positif (H={h}, n={n})")
    lam = np.maximum(lam, 0.0)
    m = len(ligne)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(np.sqrt(lam / m) * z).real[:n]


def mouvement_brownien_fractionnaire(n, h, rng):
    """
    Mouvement brownien fractionnaire (fBm, NON stationnaire) : somme cumulee d'un fGn.
    La DFA y lit H + 1 (et non H) : c'est ce qui rend un exposant DFA > 1 diagnostique.
    """
    return np.cumsum(bruit_gaussien_fractionnaire(n, h, rng))

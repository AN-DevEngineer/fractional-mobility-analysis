# ./profilage/clustering.py
#
# K-means avec distance de Minkowski d_p, 1 <= p <= 2 (chapitre 5.2).
#
# POINT TECHNIQUE (objection previsible du jury) : pour p != 2, la moyenne N'EST PAS le minimiseur de
# sum_i |x_i - c|^p. Un K-means de Lloyd qui garde la mise a jour par la moyenne ne decroit plus son critere et
# n'a pas de garantie de convergence. Ici le centroide est le VRAI minimiseur coordonnee par coordonnee
# (mediane pour p = 1, moyenne pour p = 2, moindres carres repondérés iteres (IRLS) entre les deux), si bien que
# le critere sum_i d_p(x_i, c_{z_i})^p est monotone decroissant (verifie dans tests/test_clustering.py).
# Restreindre p a [1, 2] garantit la convexite du critere en c, pas "la convergence de K-means" en general.

import numpy as np
from sklearn.metrics import adjusted_rand_score


def distance_minkowski(X, C, p):
    """Matrice (n, k) des distances de Minkowski d_p entre lignes de X et de C."""
    return (np.abs(X[:, None, :] - C[None, :, :]) ** p).sum(axis=2) ** (1.0 / p)


def centroide(points, p, iterations=30, eps=1e-8):
    """Minimiseur coordonnee par coordonnee de sum_i |x_i - c|^p (mediane si p = 1, moyenne si p = 2, IRLS sinon)."""
    if p == 2:
        return points.mean(axis=0)
    if p == 1:
        return np.median(points, axis=0)
    c = np.median(points, axis=0)
    for _ in range(iterations):
        w = np.maximum(np.abs(points - c), eps) ** (p - 2.0)
        c_nouveau = (w * points).sum(axis=0) / w.sum(axis=0)
        if np.max(np.abs(c_nouveau - c)) < 1e-9:
            c = c_nouveau
            break
        c = c_nouveau
    return c


def init_kmeanspp(X, k, p, rng):
    """K-means++ avec distances d_p^p."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = (distance_minkowski(X, np.array(centres), p) ** p).min(axis=1)
        s = d2.sum()
        probas = d2 / s if s > 0 else np.full(len(X), 1.0 / len(X))
        centres.append(X[rng.choice(len(X), p=probas)])
    return np.array(centres)


def kmeans_minkowski(X, k, p=2.0, n_init=10, max_iter=100, rng=None):
    """
    K-means de Minkowski, meilleur de `n_init` initialisations K-means++.
    Sortie (dict) : etiquettes, centres, critere (sum d_p^p), historique (critere a chaque iteration de la meilleure
    initialisation), n_iter.
    """
    if not 1.0 <= p <= 2.0:
        raise ValueError("p doit etre dans [1, 2]")
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    meilleur = None
    for _ in range(n_init):
        C = init_kmeanspp(X, k, p, rng)
        historique, z_prec = [], None
        for it in range(max_iter):
            D = distance_minkowski(X, C, p)
            z = D.argmin(axis=1)
            historique.append(float((D[np.arange(len(X)), z] ** p).sum()))
            if z_prec is not None and np.array_equal(z, z_prec):
                break
            z_prec = z
            for j in range(k):
                pts = X[z == j]
                if len(pts) == 0:                       # cluster vide : on reprend le point le plus eloigne
                    C[j] = X[np.argmax(D[np.arange(len(X)), z])]
                else:
                    C[j] = centroide(pts, p)
        if meilleur is None or historique[-1] < meilleur["critere"]:
            meilleur = dict(etiquettes=z, centres=C.copy(), critere=historique[-1], historique=historique, n_iter=len(historique))
    return meilleur


def stabilite_bootstrap(X, k, p, n_boot=30, rng=None, n_init=5):
    """
    Stabilite (section 5.5) : indice de Rand ajuste MOYEN entre le partitionnement de reference et celui obtenu en
    ajustant sur un echantillon bootstrap puis en affectant TOUS les points au centre le plus proche.
    Sortie : (ari_moyen, ari_ecart_type).
    """
    rng = np.random.default_rng() if rng is None else rng
    ref = kmeans_minkowski(X, k, p, n_init=n_init, rng=rng)
    aris = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        m = kmeans_minkowski(X[idx], k, p, n_init=n_init, rng=rng)
        z = distance_minkowski(X, m["centres"], p).argmin(axis=1)
        aris.append(adjusted_rand_score(ref["etiquettes"], z))
    return float(np.mean(aris)), float(np.std(aris))

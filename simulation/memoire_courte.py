# ./simulation/memoire_courte.py
#
# Series SANS memoire longue, pour mesurer les faux positifs : un estimateur ou un test qui
# declare de la memoire longue sur ces series se trompe.

import numpy as np
from scipy.signal import lfilter


def ar1(n, phi, rng, rodage=200):
    """Processus AR(1) stationnaire, variance d'innovation 1. Memoire courte : correlation phi^k."""
    eps = rng.standard_normal(n + rodage)
    return lfilter([1.0], [1.0, -phi], eps)[rodage:]


def changements_de_regime(n, rng, niveaux=(0.0, 1.0, 3.0), duree_moyenne=60.0, bruit=1.0):
    """
    Moyenne constante par morceaux (chaine de Markov a durees geometriques) + bruit i.i.d.

    Mime les changements de regime (arret, marche, bus). Piege classique de la FAUSSE memoire
    longue (Diebold & Inoue, 2001) : de longs paliers ressemblent a de la dependance longue.
    """
    niveaux = np.asarray(niveaux, dtype=float)
    sortie = np.empty(n)
    i, etat = 0, int(rng.integers(len(niveaux)))
    while i < n:
        d = int(rng.geometric(1.0 / duree_moyenne))
        sortie[i:i + d] = niveaux[etat]
        i += d
        etat = int(rng.choice([s for s in range(len(niveaux)) if s != etat]))
    return sortie + bruit * rng.standard_normal(n)

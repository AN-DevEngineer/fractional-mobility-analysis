# ./simulation/marginale_lognormale.py
#
# Rend une serie gaussienne positive et asymetrique comme les distances GPS, en conservant sa memoire.

import numpy as np


def rendre_positive(x, sigma):
    """Marginale log-normale de moyenne 1 : x -> exp(sigma * z - sigma^2 / 2), z standardise."""
    z = (x - x.mean()) / x.std()
    return np.exp(sigma * z - sigma ** 2 / 2)

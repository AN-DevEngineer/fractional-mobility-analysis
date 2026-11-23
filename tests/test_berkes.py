import numpy as np
import pytest

from simulation import ar1, bruit_gaussien_fractionnaire
from validation.berkes import (kolmogorov_cdf, valeur_critique, variance_long_terme, cusum, largeur_bartlett,
                               statistique_T, test_berkes as _test_berkes)


def test_valeurs_critiques_du_papier():
    # section 3 du papier, u = 2 : 1.36 / 1.48 / 1.72 pour 10 % / 5 % / 1 % (arrondis du papier)
    assert valeur_critique(0.10, 2) == pytest.approx(1.36, abs=0.02)
    assert valeur_critique(0.05, 2) == pytest.approx(1.48, abs=0.02)
    assert valeur_critique(0.01, 2) == pytest.approx(1.72, abs=0.02)
    assert kolmogorov_cdf(1.358) == pytest.approx(0.95, abs=1e-3)


def test_cusum_retrouve_la_rupture():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    x[400:] += 2.0
    _, k = cusum(x)
    assert abs(k - 400) <= 10


def test_variance_long_terme_iid():
    x = np.random.default_rng(1).standard_normal(20000)
    assert variance_long_terme(x, 10) == pytest.approx(1.0, abs=0.1)


def test_largeur_bartlett_bornee():
    assert 1 <= largeur_bartlett(50) < 50
    assert largeur_bartlett(1000, 7) == 7


def test_une_rupture_nette_n_est_pas_de_la_memoire_longue():
    rng = np.random.default_rng(2)
    rej = [_test_berkes(np.r_[rng.standard_normal(500), 1.5 + rng.standard_normal(500)], u_max=1)["rejette"]
           for _ in range(60)]
    assert np.mean(rej) < 0.2          # H0 vraie : peu de rejets


def test_fgn_fort_est_detecte_plus_souvent_que_iid():
    rng = np.random.default_rng(3)
    lg = np.mean([_test_berkes(bruit_gaussien_fractionnaire(2000, 0.9, rng), u_max=1)["rejette"] for _ in range(60)])
    ii = np.mean([_test_berkes(rng.standard_normal(2000), u_max=1)["rejette"] for _ in range(60)])
    assert lg > ii + 0.2


def test_serie_trop_courte_non_concluante():
    assert _test_berkes(np.arange(10.0))["rejette"] is None

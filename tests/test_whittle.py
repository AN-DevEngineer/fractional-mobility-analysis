import numpy as np
import pytest

from hurst.whittle import Whittle, frequences, periodogramme, periodogramme_rapide, whittle_local
from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire
from simulation.marginale_lognormale import rendre_positive
from simulation.memoire_courte import ar1


@pytest.mark.parametrize("n", [120, 300, 411])
def test_parite_avec_whittle_v4(n, rng):
    series = [bruit_gaussien_fractionnaire(n, 0.75, rng) + 10.0,
              rendre_positive(ar1(n, 0.6, rng), 1.0) * 5.0,
              rng.standard_normal(n)]
    for x in series:
        assert whittle_local(x).h == pytest.approx(Whittle(list(map(float, x))), abs=1e-12)


def test_periodogramme_rapide_egal_a_la_version_v4(rng):
    x = rng.standard_normal(64)
    lam = frequences(64)
    I_v4, lam_v4 = periodogramme(list(x), lam)            # la V4 renvoie (periodogramme, frequences)
    lam_rapide, I_rapide = periodogramme_rapide(x)
    assert np.allclose(I_rapide, I_v4, rtol=1e-9) and np.allclose(lam_rapide, lam_v4)


def test_la_borne_de_la_grille_est_signalee_sur_une_serie_non_stationnaire(rng):
    r = whittle_local(np.cumsum(rng.standard_normal(600)))      # marche aleatoire
    assert r.a_la_borne and r.h >= 0.999


def test_pas_de_borne_sur_un_bruit_blanc(rng):
    assert not whittle_local(rng.standard_normal(600)).a_la_borne


def test_ecart_type_conforme_a_la_theorie_asymptotique(rng):
    """Variance asymptotique 1/(4m) : c'est une limite de la methode, pas un defaut du code."""
    n = 1000
    valeurs = [whittle_local(rng.standard_normal(n)).h for _ in range(150)]
    m = whittle_local(rng.standard_normal(n)).m
    assert np.std(valeurs, ddof=1) == pytest.approx(1 / (2 * np.sqrt(m)), rel=0.25)


def test_series_courtes_donnent_nan_ou_none():
    assert np.isnan(whittle_local(np.arange(10.0)).h)
    assert Whittle(list(np.arange(10.0))) is None

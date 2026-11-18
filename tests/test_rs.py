import numpy as np
import pytest

from hurst.echelles import grille_echelles
from hurst.rs import RS, esperance_rs, generer_valeurs_m, rs_complet, rs_moyen
from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire
from simulation.marginale_lognormale import rendre_positive
from simulation.memoire_courte import ar1


@pytest.mark.parametrize("n", [120, 300, 411])
def test_parite_avec_rs_v4(n, rng):
    series = [bruit_gaussien_fractionnaire(n, 0.75, rng) + 10.0,
              rendre_positive(ar1(n, 0.6, rng), 1.0) * 5.0,
              rng.standard_normal(n)]
    for x in series:
        assert rs_complet(x).h_brut == pytest.approx(RS(list(map(float, x))), abs=1e-9)


@pytest.mark.parametrize("n", [100, 300, 1000])
def test_grille_de_tailles_identique_a_la_v4(n):
    assert list(grille_echelles(10, n // 2, 20)) == list(generer_valeurs_m(n, 20))


def test_esperance_rs_correspond_a_monte_carlo(rng):
    """Formule exacte d'Anis-Lloyd, SANS facteur de Peters : verifie par simulation (le facteur sous-estime de 5 % a n = 10)."""
    for n in (10, 50, 200):
        mc = np.mean([rs_moyen(rng.standard_normal(n), n) for _ in range(3000)])
        assert mc / esperance_rs(n) == pytest.approx(1.0, abs=0.02)


def test_correction_retire_le_biais_de_petit_echantillon_sur_bruit_blanc(rng):
    res = [rs_complet(rng.standard_normal(500)) for _ in range(100)]
    brut, corrige = np.mean([r.h_brut for r in res]), np.mean([r.h_corrige for r in res])
    assert brut > 0.53                                   # biais connu du R/S brut sur serie courte
    assert corrige == pytest.approx(0.5, abs=0.03)


def test_series_courtes_donnent_nan_ou_none():
    assert np.isnan(rs_complet(np.arange(15.0)).h_brut)
    assert RS(list(np.arange(15.0))) is None


def test_serie_constante_donne_nan():
    assert np.isnan(rs_complet(np.ones(200)).h_brut)

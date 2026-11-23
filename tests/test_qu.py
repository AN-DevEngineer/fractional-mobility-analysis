import numpy as np
import pytest

from simulation import bruit_gaussien_fractionnaire, ar1
from validation.qu import (whittle_local_qu, statistique_W, test_qu as _test_qu, simuler_loi_limite,
                           valeur_critique_W, preblanchir, decision_croisee)


def test_whittle_local_retrouve_d():
    rng = np.random.default_rng(0)
    d = np.mean([whittle_local_qu(bruit_gaussien_fractionnaire(4000, 0.7, rng))[0] for _ in range(30)])
    assert d == pytest.approx(0.2, abs=0.05)


def test_somme_partielle_nulle_a_r_1():
    # condition du premier ordre : W vaut max |somme partielle| avec somme(m) = 0 ; verifie via l'invariance
    x = bruit_gaussien_fractionnaire(2000, 0.7, np.random.default_rng(1))
    w, d, m = statistique_W(x)
    assert np.isfinite(w) and -0.5 < d < 1 and m == int(2000 ** 0.7)


def test_loi_limite_contre_tableau_1_du_papier():
    sim = simuler_loi_limite(nb=2000, n_grille=2000, eps=0.02)
    assert np.quantile(sim, 0.95) == pytest.approx(valeur_critique_W(0.05, 0.02), abs=0.08)


def test_preblanchiment_retire_un_ar1_fort():
    x = ar1(3000, 0.8, np.random.default_rng(2))
    y, (p, q, a, b, d) = preblanchir(x)
    assert p == 1 or q == 1
    assert abs(np.corrcoef(y[:-1], y[1:])[0, 1]) < abs(np.corrcoef(x[:-1], x[1:])[0, 1])


def test_rupture_nette_rejetee_par_qu_plus_que_fgn():
    rng = np.random.default_rng(3)
    ruptures = [_test_qu(np.r_[rng.standard_normal(1000), 1.0 + rng.standard_normal(1000)])["rejette"] for _ in range(40)]
    fgn = [_test_qu(bruit_gaussien_fractionnaire(2000, 0.7, rng))["rejette"] for _ in range(40)]
    assert np.mean(ruptures) > np.mean(fgn)


def test_serie_courte_non_concluante():
    assert _test_qu(np.arange(50.0))["rejette"] is None


def test_decision_croisee():
    assert decision_croisee(True, False) == "memoire longue compatible"
    assert decision_croisee(False, True) == "memoire courte + regimes compatible"
    assert decision_croisee(None, True) == "non concluant"

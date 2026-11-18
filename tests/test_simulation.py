import numpy as np
import pytest

from simulation import catalogue_modeles
from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire, mouvement_brownien_fractionnaire
from simulation.marginale_lognormale import rendre_positive
from simulation.memoire_courte import ar1, changements_de_regime
from simulation.on_off_pareto import h_theorique_on_off, on_off


def test_autocovariance_du_fgn_conforme_a_la_theorie(rng):
    h, n, reps = 0.8, 2048, 150
    acv = np.zeros(4)
    for _ in range(reps):
        x = bruit_gaussien_fractionnaire(n, h, rng)
        acv += np.array([np.mean(x[:n - k] * x[k:]) for k in range(4)]) / reps
    theorie = [0.5 * (abs(k + 1) ** (2 * h) - 2 * abs(k) ** (2 * h) + abs(k - 1) ** (2 * h)) for k in range(4)]
    assert acv == pytest.approx(theorie, abs=0.03)


def test_reproductibilite_avec_la_meme_graine():
    a = bruit_gaussien_fractionnaire(300, 0.7, np.random.default_rng(5))
    b = bruit_gaussien_fractionnaire(300, 0.7, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_h_invalide_est_refuse(rng):
    for h in (0.0, 1.0, 1.4):
        with pytest.raises(ValueError):
            bruit_gaussien_fractionnaire(100, h, rng)


def test_fbm_est_la_somme_cumulee_du_fgn():
    a = mouvement_brownien_fractionnaire(200, 0.6, np.random.default_rng(1))
    b = np.cumsum(bruit_gaussien_fractionnaire(200, 0.6, np.random.default_rng(1)))
    assert np.allclose(a, b)


def test_ar1_correlation_au_retard_1(rng):
    x = ar1(20000, 0.6, rng)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.6, abs=0.03)


def test_on_off_est_binaire_et_la_theorie_est_bien_definie(rng):
    x = on_off(3000, 1.5, 1.5, rng)
    assert len(x) == 3000 and set(np.unique(x)) <= {0.0, 1.0}
    assert h_theorique_on_off(1.5, 1.5) == pytest.approx(0.75)
    assert np.isnan(h_theorique_on_off(2.5, 2.5)) and np.isnan(h_theorique_on_off(1.0, 1.5))
    assert h_theorique_on_off(1.5, 1.8) == pytest.approx(0.75)      # la queue la plus lourde domine


def test_changements_de_regime_ont_des_paliers(rng):
    x = changements_de_regime(3000, rng, duree_moyenne=100.0, bruit=0.1)
    assert len(x) == 3000 and np.corrcoef(x[:-1], x[1:])[0, 1] > 0.9


def test_rendre_positive_moyenne_un_et_valeurs_positives(rng):
    y = rendre_positive(rng.standard_normal(200000), 0.8)
    assert y.min() > 0 and y.mean() == pytest.approx(1.0, abs=0.02)


def test_le_catalogue_fournit_des_series_de_la_bonne_longueur(rng):
    for nom, generer in catalogue_modeles().items():
        assert len(generer(400, rng)) == 400, nom

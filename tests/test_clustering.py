import numpy as np
import pytest

from profilage.clustering import centroide, kmeans_minkowski, distance_minkowski, stabilite_bootstrap
from profilage.caracteristiques import poids_gl, derivee_gl, entropie_renyi, dimension_boites, ordre_fractionnaire


def _melange(rng, n=60):
    return np.vstack([rng.normal(m, 0.4, (n, 3)) for m in (0.0, 3.0, 6.0)])


def test_centroide_est_le_minimiseur():
    rng = np.random.default_rng(0)
    pts = rng.standard_cauchy((50, 2))
    for p in (1.0, 1.5, 2.0):
        c = centroide(pts, p)
        f = lambda z: (np.abs(pts - z) ** p).sum()
        for _ in range(20):
            assert f(c) <= f(c + 0.05 * rng.standard_normal(2)) + 1e-9
    assert np.allclose(centroide(pts, 2.0), pts.mean(axis=0))
    assert np.allclose(centroide(pts, 1.0), np.median(pts, axis=0))


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0])
def test_critere_monotone_decroissant(p):
    X = _melange(np.random.default_rng(1))
    r = kmeans_minkowski(X, 3, p, n_init=3, rng=np.random.default_rng(2))
    h = np.array(r["historique"])
    assert np.all(np.diff(h) <= 1e-9)


def test_retrouve_trois_groupes_et_stabilite():
    from sklearn.metrics import adjusted_rand_score
    X = _melange(np.random.default_rng(3))
    y = np.repeat([0, 1, 2], 60)
    r = kmeans_minkowski(X, 3, 1.5, rng=np.random.default_rng(4))
    assert adjusted_rand_score(y, r["etiquettes"]) > 0.95
    ari, _ = stabilite_bootstrap(X, 3, 1.5, n_boot=8, rng=np.random.default_rng(5))
    assert ari > 0.9


def test_p_hors_domaine():
    with pytest.raises(ValueError):
        kmeans_minkowski(np.zeros((5, 2)), 2, p=0.5)


def test_grunwald_letnikov():
    assert np.allclose(poids_gl(1.0, 3), [1, -1, 0, 0])
    assert np.allclose(poids_gl(0.0, 3), [1, 0, 0, 0])
    x = np.cumsum(np.random.default_rng(0).standard_normal(40))
    assert np.allclose(derivee_gl(x, 1.0, 5)[1:], np.diff(x))
    k = np.arange(20, 400)
    w = np.abs(poids_gl(0.8, 399))[20:]
    assert np.polyfit(np.log(k), np.log(w), 1)[0] == pytest.approx(-1.8, abs=0.05)    # |w_k| ~ k^-(1+alpha), asymptotique


def test_renyi_et_dimension():
    assert entropie_renyi(np.ones(8) / 8, 2.0) == pytest.approx(np.log(8))
    assert entropie_renyi(np.ones(8) / 8, 1.0) == pytest.approx(np.log(8))
    t = np.linspace(0, 1, 2000)
    assert dimension_boites(48 + 0 * t, 2 + 0.01 * t) == pytest.approx(1.0, abs=0.05)


def test_ordre_fractionnaire_dans_la_grille():
    a, err = ordre_fractionnaire(np.random.default_rng(0).standard_normal(300))
    assert 0.1 <= a <= 0.9 and len(err) == 9

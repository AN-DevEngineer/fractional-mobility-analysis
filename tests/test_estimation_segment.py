import numpy as np
import pytest

import config
from conftest import points_gps, segment_marque
from hurst.estimation_segment import (estimer_segment, estimer_serie, graine_segment, h_dfa_par_morceaux,
                                      series_du_segment)
from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire


def _points(rng, n=500):
    t = 1e9 + np.cumsum(rng.uniform(1, 5, n))
    lat = 39.9 + np.cumsum(np.abs(bruit_gaussien_fractionnaire(n, 0.7, rng)) * 1e-5)
    return points_gps(t, lat, np.full(n, 116.3))


def test_series_du_segment_distances_en_metres_et_vitesses_en_kmh():
    pts = points_gps([0.0, 2.0, 4.0], [39.9, 39.9 + 2.0 / 111194.9, 39.9 + 4.0 / 111194.9], [116.3] * 3)
    s = series_du_segment(pts)
    assert s["d"] == pytest.approx([2.0, 2.0], rel=1e-3) and s["v"] == pytest.approx([3.6, 3.6], rel=1e-3)
    assert list(s["dt"]) == [2.0, 2.0]


def test_estimer_serie_cles_et_option_sans_nul(rng):
    sortie = estimer_serie(bruit_gaussien_fractionnaire(800, 0.8, rng), rng, avec_nul=True)
    for cle in ("h_dfa", "h_rs_brut", "h_rs_corrige", "h_whittle", "whittle_a_la_borne", "nul_p", "nul_exces", "ecart_pentes"):
        assert cle in sortie
    assert "nul_p" not in estimer_serie(bruit_gaussien_fractionnaire(800, 0.8, rng), rng, avec_nul=False)


def test_la_ligne_est_tracable_et_deterministe(rng):
    pts = _points(rng)
    a, b = estimer_segment(pts, "000", 3), estimer_segment(pts, "000", 3)
    assert a == b                                       # le resultat ne depend que de (graine, utilisateur, segment)
    assert a["empreinte_config"] == config.empreinte_config() and a["serie"] == "distance"
    assert a["n_points"] == 500 and a["mode_echantillonnage"] in {"temps", "distance", "mixte"}


def test_la_graine_depend_de_l_identite_du_segment_seulement():
    g = lambda u, s: np.random.default_rng(graine_segment(u, s)).random()
    assert g("000", 1) == g("000", 1) and g("000", 1) != g("000", 2) and g("000", 1) != g("001", 1)


def test_serie_vitesse_et_serie_inconnue(rng):
    pts = _points(rng)
    assert estimer_segment(pts, avec_nul=False, serie="vitesse")["serie"] == "vitesse"
    with pytest.raises(ValueError):
        estimer_segment(pts, serie="autre")


def test_h_par_morceaux_coupe_le_pic_sans_raccorder():
    rng = np.random.default_rng(2)
    v = np.abs(1.4 + 0.2 * rng.standard_normal(400))
    v[200] = 60.0
    df = segment_marque(v, rng.uniform(1, 3, 400))
    h, nb, conserves = h_dfa_par_morceaux(df, min_points=50)
    assert nb == 2 and h is not None and np.isfinite(h)
    assert conserves < len(df)                          # les points artefacts ne sont pas dans les morceaux


def test_h_par_morceaux_sans_artefact_egale_le_h_du_segment_entier():
    rng = np.random.default_rng(2)
    df = segment_marque(np.abs(1.4 + 0.2 * rng.standard_normal(400)), rng.uniform(1, 3, 400))
    h, nb, _ = h_dfa_par_morceaux(df, min_points=50)
    pts = df[["lat", "lon", "timestamp"]].to_numpy()
    assert nb == 1 and h == pytest.approx(estimer_segment(pts.tolist(), avec_nul=False)["h_dfa"])


def test_l_ordre_ar_est_un_parametre_et_est_trace(rng):
    pts = _points(rng)
    a = estimer_segment(pts, "000", 1, ordre_ar_max=1)
    b = estimer_segment(pts, "000", 1, ordre_ar_max=5)
    assert a["nul_ordre_ar"] == 1 and b["nul_ordre_ar"] == 5
    assert a["h_dfa"] == b["h_dfa"]                     # l'ordre ne change que le test nul, pas H
    assert a["nul_seuil"] != b["nul_seuil"]

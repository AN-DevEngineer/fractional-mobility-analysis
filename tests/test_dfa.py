import numpy as np
import pytest

from conftest import marche, points_gps
from hurst.dfa import (DFA, dfa_complet, dfa_h, echelles_dfa, fonction_fluctuation, generer_valeurs_s)
from hurst.echelles import grille_echelles
from hurst.estimation_segment import series_du_segment
from simulation.bruit_fractionnaire import bruit_gaussien_fractionnaire, mouvement_brownien_fractionnaire
from simulation.marginale_lognormale import rendre_positive
from simulation.memoire_courte import ar1


# ---------------------------------------------------------------- parite avec la reference V4 (DFA() ecrite a la main)
@pytest.mark.parametrize("n", [120, 300, 411])
def test_parite_avec_dfa_v4(n, rng):
    series = [bruit_gaussien_fractionnaire(n, 0.75, rng) + 10.0,
              rendre_positive(ar1(n, 0.6, rng), 1.0) * 5.0,
              rng.standard_normal(n)]
    for x in series:
        assert dfa_complet(x).h == pytest.approx(DFA(list(map(float, x))), abs=1e-9)
        assert dfa_h(x) == pytest.approx(DFA(list(map(float, x))), abs=1e-9)


def test_parite_sur_points_gps(rng):
    t, lat, lon = marche(350)
    lat = lat + np.cumsum(np.abs(rng.standard_normal(350))) * 1e-6
    pts = points_gps(t, lat, lon)
    assert dfa_h(series_du_segment(pts)["d"]) == pytest.approx(DFA(pts), abs=1e-9)


@pytest.mark.parametrize("n", [80, 300, 1000, 5000])
def test_grille_d_echelles_identique_a_la_v4(n):
    assert list(grille_echelles(4, n // 4, 20)) == list(generer_valeurs_s(n, 20))


# ---------------------------------------------------------------- exactitude statistique
def test_dfa_sur_bruit_blanc_et_fgn(rng):
    blanc = np.mean([dfa_h(rng.standard_normal(1000)) for _ in range(40)])
    persistant = np.mean([dfa_h(bruit_gaussien_fractionnaire(1000, 0.7, rng)) for _ in range(40)])
    assert blanc == pytest.approx(0.5, abs=0.04)
    assert persistant == pytest.approx(0.7, abs=0.05)


def test_dfa_lit_h_plus_un_sur_un_fbm(rng):
    """Sur un fBm (non stationnaire) l'exposant DFA vaut H + 1 : c'est ce qui rend alpha > 1 diagnostique."""
    est = np.mean([dfa_h(mouvement_brownien_fractionnaire(2000, 0.6, rng)) for _ in range(30)])
    assert est == pytest.approx(1.6, abs=0.08)


def test_dfa2_absorbe_mieux_une_tendance_que_dfa1(rng):
    x = rng.standard_normal(1500) + 0.01 * np.arange(1500)
    assert abs(dfa_h(x, ordre=2) - 0.5) < abs(dfa_h(x, ordre=1) - 0.5)


def test_echelle_min_est_un_parametre(rng):
    x = bruit_gaussien_fractionnaire(1000, 0.7, rng)
    assert dfa_h(x, echelle_min=4) != dfa_h(x, echelle_min=10)


def test_lot_de_series_donne_les_memes_resultats_que_une_par_une(rng):
    lot = np.stack([rng.standard_normal(500) for _ in range(5)])
    ech = echelles_dfa(500)
    F = fonction_fluctuation(lot, ech)
    for i in range(5):
        assert np.allclose(F[i], fonction_fluctuation(lot[i], ech)[0])


# ---------------------------------------------------------------- bords
def test_series_courtes_donnent_nan_ou_none():
    assert np.isnan(dfa_h(np.arange(10.0))) and np.isnan(dfa_complet(np.arange(10.0)).h)
    assert DFA(list(np.arange(10.0))) is None            # convention de la V4 : None = non calculable


def test_serie_constante_ne_plante_pas():
    x = np.ones(200)
    assert np.isnan(dfa_h(x)) or np.isfinite(dfa_h(x))


def test_resultat_expose_les_pentes_locales(rng):
    r = dfa_complet(bruit_gaussien_fractionnaire(2000, 0.7, rng))
    assert len(r.pentes_locales) == len(r.echelles_locales) > 3
    assert np.isfinite(r.ecart_pentes) and np.isfinite(r.h_bas) and np.isfinite(r.h_haut)

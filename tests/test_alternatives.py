"""
Tests de lois_de_puissance/alternatives.py : lois alternatives, rapports de vraisemblance, poids d'Akaike.

Deux familles de tests :
  - EXACTITUDE : chaque loi retrouve ses parametres sur des donnees simulees, et chaque densite
    s'integre bien a 1 (verification independante de la normalisation) ;
  - LOGIQUE DE COMPARAISON : signes, seuils et cas limites, sur des vraisemblances construites a la main.
Les ajustements qui demandent une optimisation numerique de la loi tronquee sont marques `lent`.
"""
import numpy as np
import pytest
from scipy import integrate, stats

from lois_de_puissance.alternatives import (NOMS_ALTERNATIVES, NOMS_MODELES, AjustementAlternatif,
                                            _log_normalisation_puissance_tronquee, _logv_puissance_tronquee,
                                            ajuster_alternatives, ajuster_exponentielle, ajuster_lognormale,
                                            ajuster_puissance_tronquee, ajuster_weibull, comparer_modeles,
                                            logv_loi_de_puissance, poids_aicc, rapport_de_vraisemblance,
                                            verdict_comparaison)
from lois_de_puissance.clauset import exposant_vraisemblance_max, simuler_loi_de_puissance


# ---------------------------------------------------------------- les densites s'integrent a 1
def test_la_loi_de_puissance_s_integre_a_un():
    xmin, alpha = 2.0, 2.5
    valeur, _ = integrate.quad(lambda x: np.exp(logv_loi_de_puissance(np.array([x]), xmin, alpha)[0]), xmin, np.inf)
    assert valeur == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("alpha,lam,xmin", [(2.0, 0.1, 1.0), (1.5, 0.01, 5.0), (2.5, 1e-5, 1.0), (3.0, 0.5, 2.0)])
def test_normalisation_de_la_loi_tronquee_egale_l_integrale_directe(alpha, lam, xmin):
    """Verifie le changement de variable en log de _log_normalisation_puissance_tronquee."""
    directe, _ = integrate.quad(lambda x: x ** (-alpha) * np.exp(-lam * x), xmin, np.inf, limit=200)
    assert np.exp(_log_normalisation_puissance_tronquee(alpha, lam, xmin)) == pytest.approx(directe, rel=1e-5)


def test_la_loi_tronquee_s_integre_a_un():
    xmin, alpha, lam = 2.0, 2.0, 0.1
    valeur, _ = integrate.quad(lambda x: np.exp(_logv_puissance_tronquee(np.array([x]), xmin, alpha, lam)[0]),
                               xmin, np.inf, limit=200)
    assert valeur == pytest.approx(1.0, abs=1e-5)


def test_normalisation_refuse_une_coupure_nulle():
    assert np.isnan(_log_normalisation_puissance_tronquee(2.0, 0.0, 1.0))


# ---------------------------------------------------------------- exactitude des ajustements
def test_exponentielle_retrouve_son_parametre(rng):
    x = 3.0 + rng.exponential(2.0, 5000)                    # lambda = 1 / 2 = 0,5
    aj = ajuster_exponentielle(x, 3.0)
    assert aj.nom == "exponentielle" and aj.nb_parametres == 1
    assert aj.parametres["lambda"] == pytest.approx(0.5, abs=0.03)
    assert aj.logv_points.shape == x.shape and np.all(np.isfinite(aj.logv_points))


def test_exponentielle_degeneree():
    assert ajuster_exponentielle([2.0, 2.0, 2.0], 2.0) is None


def test_lognormale_retrouve_ses_parametres_malgre_la_troncature(rng):
    complet = rng.lognormal(1.0, 0.5, 6000)
    xmin = float(np.quantile(complet, 0.2))
    aj = ajuster_lognormale(complet[complet >= xmin], xmin)
    assert aj.nb_parametres == 2
    assert aj.parametres["mu"] == pytest.approx(1.0, abs=0.15)
    assert aj.parametres["sigma"] == pytest.approx(0.5, abs=0.1)


def test_weibull_retrouve_sa_forme(rng):
    complet = 2.0 * rng.weibull(0.7, 6000)
    xmin = float(np.quantile(complet, 0.1))
    aj = ajuster_weibull(complet[complet >= xmin], xmin)
    assert aj.parametres["k"] == pytest.approx(0.7, abs=0.1)
    assert aj.parametres["echelle"] == pytest.approx(2.0, rel=0.25)


@pytest.mark.lent
def test_puissance_tronquee_retrouve_ses_parametres(rng):
    """Echantillon par rejet : on tire une loi de puissance (alpha = 2) et on garde x avec probabilite exp(-0,1 (x - 1))."""
    brut = simuler_loi_de_puissance(20000, 2.0, 1.0, rng)
    garde = rng.random(20000) < np.exp(-0.1 * (brut - 1.0))
    queue = brut[garde]
    aj = ajuster_puissance_tronquee(queue, 1.0, alpha_depart=2.0)
    assert aj.parametres["alpha"] == pytest.approx(2.0, abs=0.25)
    assert aj.parametres["lambda"] == pytest.approx(0.1, abs=0.04)


@pytest.mark.lent
def test_ajuster_alternatives_renvoie_au_plus_les_quatre_modeles(rng):
    queue = simuler_loi_de_puissance(500, 2.5, 1.0, rng)
    resultat = ajuster_alternatives(queue, 1.0, 2.5)
    assert set(resultat) <= set(NOMS_ALTERNATIVES) and len(resultat) >= 3
    assert all(isinstance(a, AjustementAlternatif) for a in resultat.values())


# ---------------------------------------------------------------- logique de comparaison
def test_verdict_de_comparaison():
    assert verdict_comparaison(5.0, 0.01, "lognormale") == "loi_de_puissance"
    assert verdict_comparaison(-5.0, 0.01, "lognormale") == "lognormale"
    assert verdict_comparaison(-5.0, 0.5, "lognormale") == "indecidable"       # non significatif
    assert verdict_comparaison(5.0, float("nan"), "lognormale") == "indecidable"
    assert verdict_comparaison(-5.0, 0.04, "weibull", seuil=0.01) == "indecidable"   # le seuil est un parametre


def test_rapport_de_vraisemblance_non_emboite_utilise_vuong():
    """logv_puissance = 0 ; alternative de logv -0,05 a -0,15 : R = +10, la loi de puissance gagne nettement."""
    alt = AjustementAlternatif("lognormale", {}, -np.linspace(0.05, 0.15, 100), 2)
    R, p = rapport_de_vraisemblance(np.zeros(100), alt)
    assert R == pytest.approx(10.0)
    assert p < 1e-10


def test_rapport_de_vraisemblance_sans_variance_donne_p_inconnue():
    alt = AjustementAlternatif("lognormale", {}, np.full(100, -0.1), 2)
    R, p = rapport_de_vraisemblance(np.zeros(100), alt)
    assert R == pytest.approx(10.0) and np.isnan(p)


def test_rapport_de_vraisemblance_emboite_utilise_le_chi2():
    """La loi tronquee contient la loi de puissance : R = -5 -> p = chi2(1 ddl) sur 2|R| = 10."""
    alt = AjustementAlternatif("puissance_tronquee", {}, np.full(100, 0.05), 2)
    R, p = rapport_de_vraisemblance(np.zeros(100), alt)
    assert R == pytest.approx(-5.0)
    assert p == pytest.approx(stats.chi2.sf(10.0, df=1))


def test_poids_d_akaike_somment_a_un_et_favorisent_la_meilleure_vraisemblance():
    w = poids_aicc({"a": -100.0, "b": -90.0}, {"a": 1, "b": 2}, n=1000)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["b"] > 0.99


def test_poids_d_akaike_penalisent_les_parametres_superflus():
    """Meme vraisemblance : le modele a 2 parametres pese moins que celui a 1 parametre."""
    w = poids_aicc({"simple": -100.0, "complexe": -100.0}, {"simple": 1, "complexe": 2}, n=500)
    assert w["simple"] > w["complexe"]


def test_poids_d_akaike_inconnus_si_l_echantillon_est_trop_petit():
    w = poids_aicc({"a": -10.0, "b": -9.0}, {"a": 1, "b": 2}, n=3)       # n - k - 1 = 0 pour k = 2
    assert all(np.isnan(v) for v in w.values())


# ---------------------------------------------------------------- comparaison complete
@pytest.mark.lent
def test_une_exponentielle_est_preferee_a_la_loi_de_puissance(rng):
    queue = 1.0 + rng.exponential(1.0, 4000)
    alpha, _ = exposant_vraisemblance_max(queue, 1.0)
    c = comparer_modeles(queue, 1.0, alpha)
    assert c["comparaisons"]["exponentielle"]["favorise"] == "exponentielle"
    assert c["meilleur_modele"] != "loi_de_puissance"
    assert sum(c["poids_aicc"].values()) == pytest.approx(1.0)


@pytest.mark.lent
def test_une_vraie_loi_de_puissance_n_est_pas_battue_par_la_lognormale(rng):
    queue = simuler_loi_de_puissance(3000, 2.5, 1.0, rng)
    alpha, _ = exposant_vraisemblance_max(queue, 1.0)
    c = comparer_modeles(queue, 1.0, alpha)
    assert c["comparaisons"]["lognormale"]["favorise"] in ("loi_de_puissance", "indecidable")
    assert set(c["poids_aicc"]) <= set(NOMS_MODELES)

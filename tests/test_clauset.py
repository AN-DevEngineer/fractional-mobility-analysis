"""
Tests de lois_de_puissance/clauset.py : exposant, distance de Kolmogorov-Smirnov, choix de xmin, p-valeur.

Les donnees sont SIMULEES avec un exposant connu : on verifie que la methode le retrouve. Les
tolerances sont de l'ordre de 4 a 5 ecarts-types pour qu'un test ne echoue pas par hasard.
Le test contre le paquet `powerlaw` (optionnel) valide l'implementation a la main.
"""
import numpy as np
import pytest

from lois_de_puissance.clauset import (AjustementPuissance, ajuster_loi_de_puissance, distance_ks,
                                       exposant_vraisemblance_max, p_valeur_adequation,
                                       simuler_loi_de_puissance)


# ---------------------------------------------------------------- exposant a xmin fixe
def test_formule_de_l_exposant_sur_un_exemple_a_la_main():
    """alpha = 1 + n / somme(ln(x/xmin)) : pour x = 2, 4, 8 et xmin = 1, somme = 6 ln 2."""
    alpha, sigma = exposant_vraisemblance_max([2.0, 4.0, 8.0], 1.0)
    assert alpha == pytest.approx(1.0 + 3.0 / (6.0 * np.log(2.0)))
    assert sigma == pytest.approx((alpha - 1.0) / np.sqrt(3.0))


def test_l_exposant_est_retrouve_sur_des_donnees_simulees(rng):
    x = simuler_loi_de_puissance(5000, 2.5, 1.0, rng)
    alpha, sigma = exposant_vraisemblance_max(x, 1.0)
    assert alpha == pytest.approx(2.5, abs=0.1)                        # ecart-type attendu ~ 0.02
    assert sigma == pytest.approx(1.5 / np.sqrt(5000), rel=0.1)


def test_exposant_cas_degeneres():
    assert np.isnan(exposant_vraisemblance_max([5.0], 1.0)[0])         # un seul point
    assert np.isnan(exposant_vraisemblance_max([2.0, 2.0, 2.0], 2.0)[0])   # tout egal a xmin : somme nulle


# ---------------------------------------------------------------- simulation
def test_simulation_respecte_xmin_et_la_fonction_de_survie(rng):
    x = simuler_loi_de_puissance(20000, 2.0, 3.0, rng)
    assert x.min() >= 3.0
    # P(X > 6) = (6/3)^(1 - 2) = 0.5
    assert np.mean(x > 6.0) == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------- distance KS
def test_ks_faible_pour_le_bon_modele_et_forte_pour_un_mauvais(rng):
    x = np.sort(simuler_loi_de_puissance(5000, 2.5, 1.0, rng))
    assert distance_ks(x, 2.5, 1.0) < 0.04                              # ~ 0.9 / racine(5000) = 0.013 en moyenne
    assert distance_ks(x, 1.5, 1.0) > 0.2                               # exposant tres different : ecart marque


def test_ks_est_compris_entre_zero_et_un(rng):
    x = np.sort(simuler_loi_de_puissance(300, 2.2, 5.0, rng))
    assert 0.0 <= distance_ks(x, 3.5, 5.0) <= 1.0


# ---------------------------------------------------------------- ajustement complet avec choix de xmin
def test_ajustement_sur_une_loi_de_puissance_pure(rng):
    x = simuler_loi_de_puissance(5000, 2.5, 1.0, rng)
    aj = ajuster_loi_de_puissance(x)
    assert isinstance(aj, AjustementPuissance)
    assert aj.alpha == pytest.approx(2.5, abs=0.35)                    # la queue retenue peut etre petite : alpha plus bruite
    assert aj.n == 5000 and aj.n_queue <= 5000 and aj.n_queue >= 10
    assert aj.ks < 0.06


def test_ajustement_retrouve_le_seuil_quand_un_corps_precede_la_queue(rng):
    """3000 points uniformes sur [1, 10] (corps) puis 3000 points de loi de puissance a partir de 10."""
    corps = rng.uniform(1.0, 10.0, 3000)
    queue = simuler_loi_de_puissance(3000, 2.2, 10.0, rng)
    aj = ajuster_loi_de_puissance(np.concatenate([corps, queue]))
    assert 8.0 <= aj.xmin <= 25.0                                       # le vrai seuil est 10 ; la methode est bruitee
    assert aj.alpha == pytest.approx(2.2, abs=0.3)


def test_ajustement_impossible_avec_trop_peu_de_donnees():
    assert ajuster_loi_de_puissance([1.0, 2.0, 3.0]) is None
    assert ajuster_loi_de_puissance([]) is None


def test_ajustement_ignore_les_valeurs_non_positives(rng):
    x = simuler_loi_de_puissance(400, 2.5, 1.0, rng)
    avec_zeros = np.concatenate([x, [0.0, -3.0]])
    assert ajuster_loi_de_puissance(avec_zeros).n == 400


def test_le_nombre_de_candidats_limite_les_seuils_essayes(rng):
    x = simuler_loi_de_puissance(2000, 2.5, 1.0, rng)
    a = ajuster_loi_de_puissance(x, nb_candidats=20)
    assert a is not None and a.alpha == pytest.approx(2.5, abs=0.35)


# ---------------------------------------------------------------- p-valeur d'adequation
def test_p_valeur_d_une_vraie_loi_de_puissance_n_est_pas_petite(rng):
    x = simuler_loi_de_puissance(500, 2.5, 1.0, rng)
    aj = ajuster_loi_de_puissance(x)
    p = p_valeur_adequation(x, aj, rng, nb_bootstrap=60)
    assert 0.0 <= p <= 1.0
    assert p > 0.01                    # sous H0, p est uniforme : 1 % de risque de faux rejet avec cette graine


def test_p_valeur_est_reproductible_avec_la_meme_graine(rng):
    x = simuler_loi_de_puissance(300, 2.5, 1.0, np.random.default_rng(1))
    aj = ajuster_loi_de_puissance(x)
    p1 = p_valeur_adequation(x, aj, np.random.default_rng(42), nb_bootstrap=20)
    p2 = p_valeur_adequation(x, aj, np.random.default_rng(42), nb_bootstrap=20)
    assert p1 == p2


def test_p_valeur_sans_corps_ne_plante_pas(rng):
    """Cas ou xmin est la plus petite valeur : le corps (x < xmin) est vide."""
    x = simuler_loi_de_puissance(200, 2.5, 1.0, rng)
    aj = AjustementPuissance(n=200, xmin=float(x.min()), alpha=2.5, sigma_alpha=0.1, ks=0.05, n_queue=200)
    p = p_valeur_adequation(x, aj, rng, nb_bootstrap=5)
    assert np.isnan(p) or 0.0 <= p <= 1.0


@pytest.mark.lent
def test_une_exponentielle_est_rejetee(rng):
    x = rng.exponential(1.0, 3000)
    aj = ajuster_loi_de_puissance(x)
    assert p_valeur_adequation(x, aj, rng, nb_bootstrap=60) < 0.2


# ---------------------------------------------------------------- validation contre le paquet de reference
@pytest.mark.lent
def test_accord_avec_le_paquet_powerlaw(rng):
    """
    Valide l'implementation a la main contre `powerlaw` (Alstott et al., 2014) : meme donnees, memes
    exposant et seuil (a la tolerance pres : notre recherche de xmin se limite a LP_NB_CANDIDATS_XMIN
    valeurs). Ignore si le paquet n'est pas installe (pip install powerlaw).
    """
    powerlaw = pytest.importorskip("powerlaw")
    x = np.concatenate([rng.uniform(1.0, 10.0, 2000), simuler_loi_de_puissance(3000, 2.4, 10.0, rng)])
    ref = powerlaw.Fit(x, verbose=False)
    nous = ajuster_loi_de_puissance(x, nb_candidats=5000)
    assert nous.alpha == pytest.approx(ref.power_law.alpha, abs=0.05)
    assert nous.xmin == pytest.approx(ref.power_law.xmin, rel=0.1)

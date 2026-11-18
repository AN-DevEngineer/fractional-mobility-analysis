"""
Tests de lois_de_puissance/analyse.py : analyser_distribution(), son verdict et le domaine du lien ON/OFF.

On teste surtout la LOGIQUE de l'assemblage (colonnes, coherence du verdict avec la p-valeur, domaine
de validite de H = (3 - a) / 2), sur des donnees simulees d'exposant connu. Les calculs statistiques
eux-memes sont testes dans test_clauset.py et test_alternatives.py.
"""
import numpy as np
import pytest

from config import LP_SEUIL_P
from lois_de_puissance.alternatives import NOMS_ALTERNATIVES, NOMS_MODELES
from lois_de_puissance.analyse import analyser_distribution
from lois_de_puissance.clauset import simuler_loi_de_puissance


# ---------------------------------------------------------------- forme du resultat
def test_donnees_insuffisantes_donnent_un_verdict_insuffisant(rng):
    r = analyser_distribution([1.0, 2.0, 3.0], rng)
    assert r["verdict"] == "insuffisant" and r["n"] == 3
    assert np.isnan(r["alpha"]) and not r["dans_domaine_on_off"]


def test_les_colonnes_sont_toujours_les_memes(rng):
    """Un CSV doit rester rectangulaire, qu'un ajustement ait eu lieu ou non."""
    vide = analyser_distribution([1.0, 2.0, 3.0], rng)
    plein = analyser_distribution(simuler_loi_de_puissance(400, 2.5, 1.0, rng), rng, nb_bootstrap=0, avec_alternatives=False)
    assert set(vide) == set(plein)
    for nom in NOMS_MODELES:
        assert f"poids_aicc_{nom}" in vide
    for nom in NOMS_ALTERNATIVES:
        assert {f"R_{nom}", f"p_{nom}", f"favorise_{nom}"} <= set(vide)


def test_les_valeurs_non_finies_ou_negatives_sont_ignorees(rng):
    x = simuler_loi_de_puissance(300, 2.5, 1.0, rng)
    sale = np.concatenate([x, [np.nan, np.inf, -1.0, 0.0]])
    assert analyser_distribution(sale, rng, nb_bootstrap=0, avec_alternatives=False)["n"] == 300


def test_champs_de_queue_coherents(rng):
    x = simuler_loi_de_puissance(1000, 2.5, 1.0, rng)
    r = analyser_distribution(x, rng, nb_bootstrap=0, avec_alternatives=False)
    assert r["xmin"] > 0 and 10 <= r["n_queue"] <= r["n"] == 1000
    assert r["part_queue"] == pytest.approx(r["n_queue"] / r["n"])
    assert r["exposant_survie"] == pytest.approx(r["alpha"] - 1.0)
    assert 0.0 <= r["ks"] <= 1.0


# ---------------------------------------------------------------- verdict
def test_sans_bootstrap_le_verdict_est_indetermine(rng):
    r = analyser_distribution(simuler_loi_de_puissance(400, 2.5, 1.0, rng), rng, nb_bootstrap=0, avec_alternatives=False)
    assert r["verdict"] == "indetermine" and np.isnan(r["p_bootstrap"]) and r["nb_bootstrap"] == 0


def test_le_verdict_est_coherent_avec_la_p_valeur(rng):
    """Rejet si et seulement si p <= seuil ; sinon plausible ou alternative_meilleure."""
    r = analyser_distribution(simuler_loi_de_puissance(400, 2.5, 1.0, rng), rng, nb_bootstrap=30, avec_alternatives=False)
    assert 0.0 <= r["p_bootstrap"] <= 1.0 and r["nb_bootstrap"] == 30
    assert (r["verdict"] == "rejetee") == (r["p_bootstrap"] <= LP_SEUIL_P)
    if r["verdict"] != "rejetee":
        assert r["verdict"] in ("plausible", "alternative_meilleure")


def test_le_resultat_ne_depend_que_de_la_graine():
    x = simuler_loi_de_puissance(300, 2.5, 1.0, np.random.default_rng(5))
    a = analyser_distribution(x, np.random.default_rng(11), nb_bootstrap=10, avec_alternatives=False)
    b = analyser_distribution(x, np.random.default_rng(11), nb_bootstrap=10, avec_alternatives=False)
    for cle in ("alpha", "xmin", "ks", "p_bootstrap", "n_queue"):
        assert a[cle] == b[cle]


# ---------------------------------------------------------------- domaine du lien avec H (ON/OFF)
def test_exposant_de_densite_entre_2_et_3_est_dans_le_domaine(rng):
    """alpha de densite = 2,5 <=> exposant de survie 1,5 : H = (3 - 1,5) / 2 = 0,75 = (4 - alpha) / 2."""
    r = analyser_distribution(simuler_loi_de_puissance(3000, 2.5, 1.0, rng), rng, nb_bootstrap=0, avec_alternatives=False)
    assert r["dans_domaine_on_off"]
    assert r["h_on_off_predit"] == pytest.approx((4.0 - r["alpha"]) / 2.0)


def test_exposant_de_densite_sous_2_est_hors_domaine(rng):
    """alpha de densite = 1,5 <=> exposant de survie 0,5 < 1 : duree moyenne infinie, la formule ne s'applique pas."""
    r = analyser_distribution(simuler_loi_de_puissance(3000, 1.5, 1.0, rng), rng, nb_bootstrap=0, avec_alternatives=False)
    assert not r["dans_domaine_on_off"] and np.isnan(r["h_on_off_predit"])


@pytest.mark.lent
def test_analyse_complete_avec_alternatives(rng):
    r = analyser_distribution(simuler_loi_de_puissance(800, 2.5, 1.0, rng), rng, nb_bootstrap=40, avec_alternatives=True)
    poids = [r[f"poids_aicc_{nom}"] for nom in NOMS_MODELES if np.isfinite(r[f"poids_aicc_{nom}"])]
    assert sum(poids) == pytest.approx(1.0)
    assert r["meilleur_modele"] in NOMS_MODELES
    for nom in NOMS_ALTERNATIVES:
        assert r[f"favorise_{nom}"] in ("", "loi_de_puissance", "indecidable", nom)
